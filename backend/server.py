from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes

import numpy as np
import os, sys, shutil, json, traceback, re, subprocess
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

UNIMER_WORKER   = "/content/ScriptSense/backend/run_unimer.py"
QWEN_MODEL_PATH = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"

# =========================
# 🔹 GLOBAL QWEN STATE
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
            max_model_len=2048,
        )
        _qwen_sampling = SamplingParams(temperature=0, max_tokens=2048)
        print("✅ Qwen loaded and cached")
    else:
        print("✅ Qwen already loaded, reusing")
    return _qwen_llm, _qwen_sampling

# =========================
# 🔹 RUN Uni-MuMER IN SUBPROCESS
# =========================
def run_unimer_subprocess(images: list, sheet_dir: str) -> dict:
    tmp_dir = os.path.join(sheet_dir, "pages")
    os.makedirs(tmp_dir, exist_ok=True)

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
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Uni-MuMER subprocess failed (code {result.returncode})")

    with open(output_json) as f:
        data = json.load(f)

    print("✅ Uni-MuMER subprocess done, GPU fully released")
    return data

# =========================
# 🔹 HELPER: wrap text into lines of ~8 words each
# =========================
def wrap_into_lines(text: str, words_per_line: int = 8) -> list:
    words = text.split()
    lines = []
    for i in range(0, len(words), words_per_line):
        lines.append(" ".join(words[i:i + words_per_line]))
    return lines

# =========================
# 🔹 QWEN PASS 1 — CLEAN TEXT (for display to professors)
# =========================
def qwen_clean_text(raw_text: str) -> str:
    if not raw_text.strip():
        return ""

    qwen, sampling = get_qwen()

    system_msg = (
        "You are an OCR post-processor for handwritten exam answer sheets. "
        "You receive garbled OCR text with merged words, missing spaces, and spelling noise. "
        "Fix all spacing, word-merging, and obvious OCR errors. "
        "Preserve the original meaning exactly — do NOT add, remove, or rephrase content. "
        "Output ONLY the corrected text as clean prose. No labels, no explanation."
    )

    user_msg = f"Fix the OCR errors in this text:\n\n{raw_text}"

    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    result = qwen.generate([prompt], sampling)[0].outputs[0].text
    result = result.replace("<|im_end|>", "").strip()

    print("\n==============================")
    print("🧹 QWEN PASS 1: CLEANED TEXT")
    print("==============================")
    print(result)
    print("==============================\n")

    return result

# =========================
# 🔹 QWEN PASS 2 — EXTRACT KEY POINTS
# =========================
def qwen_extract_key_points(cleaned_text: str) -> list:
    if not cleaned_text.strip():
        return []

    qwen, sampling = get_qwen()

    system_msg = (
        "You are an exam answer analyser. "
        "Extract the distinct key points/facts/steps from the student's answer. "
        "Express each key point as a SHORT phrase of 5-8 words maximum. "
        "Return ONLY a JSON array of strings — one string per key point. "
        "Example: "
        '[\"Gradient descent minimizes the loss function\", '
        '\"Weights updated via backpropagation\"] '
        "No explanation, no labels — ONLY the JSON array."
    )

    user_msg = f"Extract key points:\n\n{cleaned_text}"

    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    raw = qwen.generate([prompt], sampling)[0].outputs[0].text
    raw = raw.replace("<|im_end|>", "").strip()
    raw = re.sub(r"```[a-z]*|```", "", raw).strip()

    try:
        points = json.loads(raw)
        if not isinstance(points, list):
            raise ValueError("Not a list")
        points = [str(p).strip() for p in points if str(p).strip()]
    except Exception as e:
        print(f"⚠️ Key point parse failed ({e}), falling back to sentence split")
        sentences = re.split(r'[.;]\s*', cleaned_text)
        points = [s.strip() for s in sentences if len(s.strip()) > 4]

    print("\n==============================")
    print("🔑 QWEN PASS 2: KEY POINTS")
    print("==============================")
    for i, p in enumerate(points):
        print(f"  {i+1}. {p}")
    print("==============================\n")

    return points

# =========================
# 🔹 QWEN PASS 3 — EVALUATE
# =========================
def qwen_evaluate(key_points: list, questions_data: list) -> dict:
    qwen, sampling = get_qwen()

    student_answer_for_eval = (
        "\n".join(f"- {p}" for p in key_points)
        if key_points else "(no answer detected)"
    )

    total_score  = 0.0
    max_total    = 0
    per_question = []

    system_msg = (
        "You are a strict but fair exam evaluator. "
        "You are given a key answer and the student's answer as extracted key points. "
        "Evaluate how well the student's key points cover the key answer. "
        "Respond with ONLY valid JSON — no other text:\n"
        '{"score": <number>, "remarks": "<one or two concise sentences>"}'
    )

    print("\n==============================")
    print("📝 QWEN PASS 3: EVALUATION")
    print("==============================")

    for i, q in enumerate(questions_data):
        key_answer = q.get("keyAnswer", "")
        marks      = q.get("marks", 1)
        max_total += marks

        user_msg = (
            f"Key Answer: {key_answer}\n\n"
            f"Student's Key Points:\n{student_answer_for_eval}\n\n"
            f"Maximum marks: {marks}\n"
            f"Award a score between 0 and {marks}."
        )

        prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        raw = ""
        try:
            raw     = qwen.generate([prompt], sampling)[0].outputs[0].text
            raw     = raw.replace("<|im_end|>", "").strip()
            raw     = re.sub(r"```[a-z]*|```", "", raw).strip()
            parsed  = json.loads(raw)
            score   = max(0.0, min(float(marks), float(parsed.get("score", 0))))
            remarks = parsed.get("remarks", "")

            print(f"  Q{i+1}: {score}/{marks} — {remarks}")
            per_question.append({"score": score, "marks": marks, "remarks": remarks})
            total_score += score

        except Exception as e:
            print(f"  ⚠️ Q{i+1} eval failed: {e} | raw='{raw}'")
            per_question.append({"score": 0, "marks": marks, "remarks": "Evaluation failed"})

    print(f"\n✅ Final Score: {total_score}/{max_total}")
    print("==============================\n")

    return {
        "score":        round(total_score, 2),
        "total":        max_total,
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

            # ── STAGE 1: Uni-MuMER (subprocess) ─────────────────────────────
            unimer_result = run_unimer_subprocess(images, sheet_dir)
            raw_text      = unimer_result.get("raw_text", "")

            print(f"\n📝 Raw blob ({len(raw_text)} chars): {raw_text[:300]}\n")

            # ── STAGE 2: Qwen Pass 1 — clean text ───────────────────────────
            print("🧠 Qwen Pass 1: Cleaning OCR text...")
            cleaned_text = qwen_clean_text(raw_text)

            # Wrap into ~8-word lines for professor display (done in Python, not Qwen)
            student_answer_lines = wrap_into_lines(cleaned_text, words_per_line=8)

            # ── STAGE 3: Qwen Pass 2 — extract key points ───────────────────
            print("🧠 Qwen Pass 2: Extracting key points...")
            key_points = qwen_extract_key_points(cleaned_text)

            # ── STAGE 4: Qwen Pass 3 — evaluate ─────────────────────────────
            print("🧠 Qwen Pass 3: Evaluating against answer key...")
            eval_result = qwen_evaluate(key_points, questions_data)

            final_result = {
                "score":          eval_result["score"],
                "total":          eval_result["total"],
                "student_answer": student_answer_lines,
                "key_points":     key_points,
                "per_question":   eval_result["per_question"],
            }

            # ── FULL JSON LOG ────────────────────────────────────────────────
            print("\n" + "=" * 60)
            print("📦 FINAL RESPONSE JSON")
            print("=" * 60)
            print(json.dumps(final_result, indent=2))
            print("=" * 60 + "\n")

            results.append(final_result)
            shutil.rmtree(sheet_dir, ignore_errors=True)

        return results[0] if results else {"score": 0, "total": 0}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}