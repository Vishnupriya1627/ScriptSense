from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.ocr import ocr_from_image

import numpy as np
import os, sys, shutil, json, traceback, gc, re
from typing import List

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

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

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"

# =========================
# 🔹 GLOBAL MODEL STATE
# Only one model lives in GPU at a time.
# Qwen persists across requests; unloaded only when Uni-MuMER needs to load.
# =========================
_qwen_llm      = None
_qwen_sampling = None

# =========================
# 🔹 FULL vLLM TEARDOWN
# torch.cuda.empty_cache() alone does NOT release vLLM's internal CUDA pool.
# destroy_model_parallel() is required before the next model can load.
# =========================
def destroy_llm(llm):
    try:
        destroy_model_parallel()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("🗑️  Model destroyed, GPU freed")

# =========================
# 🔹 LOAD Uni-MuMER
# Always unloads Qwen first if alive.
# =========================
def load_unimer():
    global _qwen_llm, _qwen_sampling

    if _qwen_llm is not None:
        print("🗑️  Unloading Qwen before Uni-MuMER...")
        q = _qwen_llm
        _qwen_llm = None
        _qwen_sampling = None
        destroy_llm(q)

    print("🔄 Loading Uni-MuMER...")
    llm = LLM(
        model=UNIMER_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    print("✅ Uni-MuMER loaded")
    return llm, SamplingParams(temperature=0, max_tokens=512)

# =========================
# 🔹 GET QWEN (singleton)
# Loads once after Uni-MuMER is destroyed, stays alive for all subsequent passes.
# =========================
def get_qwen():
    global _qwen_llm, _qwen_sampling
    if _qwen_llm is None:
        print("🔄 Loading Qwen...")
        _qwen_llm = LLM(
            model=QWEN_MODEL_PATH,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.75,
            max_model_len=1024,
        )
        _qwen_sampling = SamplingParams(temperature=0, max_tokens=1024)
        print("✅ Qwen loaded and cached")
    else:
        print("✅ Qwen already loaded, reusing")
    return _qwen_llm, _qwen_sampling

# =========================
# 🔹 CLEAN Uni-MuMER OUTPUT
# Merges spaced-out single characters: "G r a d i e n t" → "Gradient"
# =========================
def clean_unimumer_output(text: str) -> str:
    if not text:
        return text

    text = re.sub(
        r'(?<!\S)((?:[A-Za-z] )+[A-Za-z])(?!\S)',
        lambda m: m.group(0).replace(' ', ''),
        text
    )
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)\.', r'\1. ', text)
    text = text.replace(" .", ".").replace(" ,", ",")

    return text.strip()

# =========================
# 🔹 OCR LINE COUNT
# Reads the texts/ folder written by segment_lines_and_find_diagrams.
# Takes the median across pages to ignore outliers (diagrams, blank pages).
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
# 🔹 QWEN PASS 1 — CLEAN + SPLIT
# Takes the raw Uni-MuMER text blob (merged words, no spacing)
# and produces exactly `expected_lines` clean lines matching the handwriting layout.
# =========================
def qwen_clean_and_split(raw_text: str, expected_lines: int) -> list:
    if not raw_text.strip():
        return []

    qwen, sampling = get_qwen()

    system_msg = (
        "You are an OCR post-processor for handwritten answer sheets. "
        "You receive a block of OCR text with spacing issues and merged words. "
        "Your job: fix all spacing/broken words, then split the corrected text "
        f"into exactly {expected_lines} lines that reflect the natural sentence "
        "and paragraph breaks in the original handwriting. "
        "Output ONLY the fixed text — one line of output per line of handwriting. "
        "No labels, no numbering, no explanations, no extra lines."
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
# For each question: send key answer + full student text to Qwen.
# Qwen returns {"score": x, "reason": "..."} JSON.
# Scores are summed across all questions.
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
        "Evaluate the student's answer based on conceptual correctness and coverage of key points. "
        "Respond with ONLY a valid JSON object in this exact format with no other text: "
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
            f"Maximum marks for this question: {marks}\n"
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

            # Clamp to valid range
            score = max(0.0, min(float(marks), score))

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
#   1. Unload Qwen if alive (so Uni-MuMER has full VRAM)
#   2. Load Uni-MuMER → recognize text from all pages → merge into one blob
#   3. Read OCR line count from texts/ folder (written by segmentation)
#   4. Destroy Uni-MuMER (full vLLM teardown)
#   5. Load Qwen (or reuse if already cached)
#   6. Qwen Pass 1: fix spacing + split into OCR line count
#   7. Qwen Pass 2: evaluate each question against key answer
#   8. Return score, per_question breakdown, and structured lines
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

            # ── STAGE 1: Uni-MuMER recognition ──────────────────────────────
            unimer, u_sampling = load_unimer()

            content = await sheet.read()
            images  = convert_from_bytes(content)

            for page_idx, img in enumerate(images):
                arr = np.array(img.convert("RGB"))
                count, _ = segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer,
                    sampling_params=u_sampling,
                )
                print(f"📄 Page {page_idx+1}: Uni-MuMER lines={count}")

            # Merge all Uni-MuMER output into one text blob
            raw_text  = ""
            latex_file = os.path.join(sheet_dir, "formulas_latex.json")

            if os.path.exists(latex_file):
                with open(latex_file, "r") as f:
                    data = json.load(f)

                print("\n==============================")
                print("📝 Uni-MuMER RAW OUTPUT")
                print("==============================")

                raw_parts = []
                for i, v in enumerate(data.values()):
                    cleaned = clean_unimumer_output(v)
                    if cleaned.strip():
                        raw_parts.append(cleaned)
                        print(f"Line {i+1}: {cleaned}")

                # Join as a single blob — Qwen will re-split by line count
                raw_text = " ".join(raw_parts)
                print("==============================\n")

            # ── STAGE 2: OCR line count (texts/ folder still exists here) ───
            expected_lines = get_ocr_line_count(sheet_dir)

            # ── STAGE 3: Destroy Uni-MuMER ───────────────────────────────────
            destroy_llm(unimer)
            del unimer

            # ── STAGE 4: Qwen Pass 1 — clean + split ─────────────────────────
            print("🧠 Qwen Pass 1: Cleaning and splitting into lines...")
            structured_lines = qwen_clean_and_split(raw_text, expected_lines)

            # ── STAGE 5: Qwen Pass 2 — evaluate ──────────────────────────────
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