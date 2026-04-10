from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.ocr import ocr_from_image

import numpy as np
import os, sys, shutil, json, traceback, gc, re, tempfile, subprocess
from typing import List
from PIL import Image

import torch
from vllm import LLM, SamplingParams

# =========================
# 🔹 APP INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# =========================
# 🔹 PATHS
# =========================
sys.path.insert(0, '/content/Uni-MuMER')

UNIMER_WORKER  = "/content/ScriptSense/backend/run_unimer.py"
QWEN_MODEL_PATH = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"

# =========================
# 🔹 GLOBAL QWEN STATE
# Qwen lives across requests. Because Uni-MuMER runs in a subprocess,
# there is no VRAM conflict — the subprocess exits and fully releases GPU
# before Qwen runs. No manual teardown needed.
# =========================
_qwen_llm      = None
_qwen_sampling = None

def get_qwen():
    global _qwen_llm, _qwen_sampling
    if _qwen_llm is None:
        print("🔄 Loading Qwen...")
        _qwen_llm = LLM(
            model=QWEN_MODEL_PATH,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.75,
            max_model_len=2048,  # increased so Qwen can output 18+ lines
        )
        # max_tokens=2048 gives enough room for 18-25 lines of cleaned text
        _qwen_sampling = SamplingParams(temperature=0, max_tokens=2048)
        print("✅ Qwen loaded and cached")
    else:
        print("✅ Qwen already loaded, reusing")
    return _qwen_llm, _qwen_sampling

# =========================
# 🔹 OCR LINE COUNT
# Reads the texts/ folder written by segment_lines_and_find_diagrams.
# Takes the median across pages to ignore outlier pages (diagrams, blank).
# =========================
def get_ocr_line_count(sheet_dir: str) -> int:
    text_folder = os.path.join(sheet_dir, "texts")
    counts = []

    if os.path.exists(text_folder):
        for filename in sorted(os.listdir(text_folder)):
            filepath = os.path.join(text_folder, filename)
            with open(filepath, "rb") as f:
                txt = ocr_from_image(f.read())
            lines = [l for l in txt.split("\n") if l.strip()]
            count = len(lines)
            print(f"  📋 OCR [{filename}]: {count} lines")
            counts.append(count)

    print(f"📊 OCR counts per image: {counts}")

    if not counts:
        return 5

    filtered = [c for c in counts if c >= 5]
    if not filtered:
        filtered = counts
    filtered.sort()
    mid = len(filtered) // 2
    median = (filtered[mid - 1] + filtered[mid]) // 2 if len(filtered) % 2 == 0 else filtered[mid]
    result = max(3, min(median, 40))
    print(f"📊 Final expected line count: {result}")
    return result

# =========================
# 🔹 RUN Uni-MuMER IN SUBPROCESS
# Saves page images to disk, calls run_unimer.py as a child process,
# reads back the JSON result. When the subprocess exits, all its CUDA
# allocations are released by the OS — no vLLM teardown hacks needed.
# =========================
def run_unimer_subprocess(images: list, sheet_dir: str) -> dict:
    tmp_dir = os.path.join(sheet_dir, "pages")
    os.makedirs(tmp_dir, exist_ok=True)

    # Save page images so the subprocess can read them
    page_paths = []
    for i, img in enumerate(images):
        path = os.path.join(tmp_dir, f"page_{i}.png")
        img.save(path)
        page_paths.append(path)

    input_json  = os.path.join(sheet_dir, "unimer_input.json")
    output_json = os.path.join(sheet_dir, "unimer_output.json")

    with open(input_json, "w") as f:
        json.dump({"pages": page_paths, "sheet_dir": sheet_dir}, f)

    print("🔄 Starting Uni-MuMER subprocess...")
    result = subprocess.run(
        [sys.executable, UNIMER_WORKER, input_json, output_json],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        timeout=600,  # 10 min max
    )

    if result.returncode != 0:
        raise RuntimeError(f"Uni-MuMER subprocess failed with code {result.returncode}")

    with open(output_json) as f:
        data = json.load(f)

    print("✅ Uni-MuMER subprocess done, GPU fully released")
    return data  # {"raw_text": "...", "page_line_counts": [...]}

# =========================
# 🔹 QWEN PASS 1 — CLEAN + SPLIT
# Takes the raw Uni-MuMER text blob and produces exactly `expected_lines` lines.
# max_tokens=2048 ensures Qwen doesn't truncate mid-output on long answers.
# =========================
def qwen_clean_and_split(raw_text: str, expected_lines: int) -> list:
    if not raw_text.strip():
        return []

    qwen, sampling = get_qwen()

    system_msg = (
        "You are an OCR post-processor for handwritten answer sheets. "
        "You receive a block of OCR text with spacing issues and merged words. "
        f"Fix all spacing and broken words, then split the corrected text into "
        f"exactly {expected_lines} lines that reflect the natural sentence and "
        "paragraph breaks of the original handwriting. "
        "Output ONLY the fixed text. "
        "Each line of output must correspond to one line of handwriting. "
        "No labels, no numbering, no explanations. "
        f"You MUST output exactly {expected_lines} lines — no more, no less."
    )

    user_msg = f"Fix and split into exactly {expected_lines} lines:\n\n{raw_text}"

    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    result = qwen.generate([prompt], sampling)[0].outputs[0].text
    result = result.replace("<|im_end|>", "").strip()

    lines = [l.strip() for l in result.split("\n") if l.strip()]

    if len(lines) > expected_lines:
        lines = lines[:expected_lines]
    while len(lines) < expected_lines:
        lines.append("")

    print("\n==============================")
    print("🧹 QWEN PASS 1: CLEANED + SPLIT")
    print("==============================")
    for i, line in enumerate(lines):
        print(f"Line {i+1}: {line}")
    print("==============================\n")

    return lines

# =========================
# 🔹 QWEN PASS 2 — EVALUATE
# For each {keyAnswer, marks} question: Qwen reads the full student text
# and returns {"score": x, "reason": "..."}.
# =========================
def qwen_evaluate(structured_lines: list, questions_data: list) -> dict:
    qwen, sampling = get_qwen()

    student_text = "\n".join(l for l in structured_lines if l.strip())
    total_score  = 0.0
    max_total    = 0
    per_question = []

    system_msg = (
        "You are a strict but fair exam evaluator. "
        "You will be given a key answer and a student's handwritten answer. "
        "Evaluate the student's answer based on conceptual correctness and "
        "coverage of key points. "
        "Respond with ONLY a valid JSON object — no other text:\n"
        '{"score": <number>, "reason": "<one concise sentence>"}'
    )

    print("\n==============================")
    print("📝 QWEN PASS 2: EVALUATION")
    print("==============================")

    for i, q in enumerate(questions_data):
        key_answer = q.get("keyAnswer", "")
        marks      = q.get("marks", 1)
        max_total += marks

        user_msg = (
            f"Key Answer: {key_answer}\n\n"
            f"Student's Answer:\n{student_text}\n\n"
            f"Maximum marks: {marks}\n"
            f"Give a score between 0 and {marks}."
        )

        prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        raw = ""
        try:
            raw = qwen.generate([prompt], sampling)[0].outputs[0].text
            raw = raw.replace("<|im_end|>", "").strip()
            raw_clean = re.sub(r"```[a-z]*|```", "", raw).strip()

            parsed = json.loads(raw_clean)
            score  = float(parsed.get("score", 0))
            reason = parsed.get("reason", "")
            score  = max(0.0, min(float(marks), score))

            print(f"  Q{i+1}: {score}/{marks} — {reason}")
            per_question.append({"score": score, "marks": marks, "reason": reason})
            total_score += score

        except Exception as e:
            print(f"  ⚠️ Q{i+1} eval failed: {e} | raw='{raw}'")
            per_question.append({"score": 0, "marks": marks, "reason": "Evaluation failed"})

    print(f"\n✅ Final Score: {total_score}/{max_total}")
    print("==============================\n")

    return {
        "score": round(total_score, 2),
        "total": max_total,
        "per_question": per_question,
    }

# =========================
# 🔹 HEALTH
# =========================
@app.get("/health")
async def health():
    return {"status": "running"}

# =========================
# 🔹 MAIN API
#
# Flow per request:
#   1. Save page images to disk
#   2. Run Uni-MuMER in a subprocess → subprocess exits → GPU fully freed
#   3. Read OCR line count from texts/ folder
#   4. Load Qwen (or reuse cached)
#   5. Qwen Pass 1: fix spacing + split into OCR line count
#   6. Qwen Pass 2: evaluate each question against key answer
#   7. Return score, per_question breakdown, and structured lines
# =========================
@app.post("/similarity")
async def similarity(
    request: Request,
    questions: str = Form(...),
    answer_sheets: List[UploadFile] = File(...)
):
    try:
        print("\n🚀 New request received")

        questions_data = json.loads(questions)

        if not answer_sheets:
            return {"error": "No answer sheets uploaded"}

        results = []

        for sheet_idx, sheet in enumerate(answer_sheets):

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(sheet_dir, exist_ok=True)

            content = await sheet.read()
            images  = convert_from_bytes(content)

            # ── STAGE 1: Uni-MuMER in subprocess ────────────────────────────
            unimer_result = run_unimer_subprocess(images, sheet_dir)
            raw_text      = unimer_result.get("raw_text", "")

            print(f"\n📝 Raw text blob ({len(raw_text)} chars):\n{raw_text[:300]}...\n")

            # ── STAGE 2: OCR line count ──────────────────────────────────────
            expected_lines = get_ocr_line_count(sheet_dir)

            # ── STAGE 3: Qwen Pass 1 — clean + split ────────────────────────
            print("🧠 Qwen Pass 1: Cleaning and splitting into lines...")
            structured_lines = qwen_clean_and_split(raw_text, expected_lines)

            # ── STAGE 4: Qwen Pass 2 — evaluate ─────────────────────────────
            print("🧠 Qwen Pass 2: Evaluating against answer key...")
            eval_result = qwen_evaluate(structured_lines, questions_data)

            results.append({
                "score":        eval_result["score"],
                "total":        eval_result["total"],
                "per_question": eval_result["per_question"],
                "lines":        structured_lines,
            })

            shutil.rmtree(sheet_dir, ignore_errors=True)

        return results[0] if results else {"score": 0, "total": 0}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}