from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.similarity import text_similarity
from Utils.ocr import ocr_from_image

import numpy as np
import os, sys, shutil, json, traceback, gc, re
from typing import List

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

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"

# =========================
# 🔹 GLOBAL MODEL STATE
# Key idea: Uni-MuMER loads fresh per request (too big to keep alongside Qwen),
# but Qwen is loaded ONCE after first use and kept alive for all subsequent requests.
# This avoids the ~40s Qwen reload on every request which was killing ngrok.
# =========================
_qwen_llm = None      # persists across requests
_qwen_sampling = None

# =========================
# 🔹 GPU CLEAN
# =========================
def clear_gpu():
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        pass

# =========================
# 🔹 OCR‑BASED LINE COUNT
# =========================
def get_clean_line_count(ocr_counts):
    if not ocr_counts:
        return 3

    filtered = [c for c in ocr_counts if c >= 5]
    if not filtered:
        filtered = ocr_counts

    filtered.sort()
    mid = len(filtered) // 2

    if len(filtered) % 2 == 0:
        median = (filtered[mid - 1] + filtered[mid]) // 2
    else:
        median = filtered[mid]

    return max(3, min(median, 25))

# =========================
# 🔹 CLEAN Uni-MuMER OUTPUT
# =========================
def clean_unimumer_output(text: str) -> str:
    if not text:
        return text

    # merge spaced-out single characters e.g. "G r a d i e n t" → "Gradient"
    text = re.sub(
        r'(?<!\S)((?:[A-Za-z] )+[A-Za-z])(?!\S)',
        lambda m: m.group(0).replace(' ', ''),
        text
    )

    # fix missing spaces between camelCase-merged words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)\.', r'\1. ', text)

    text = text.replace(" .", ".").replace(" ,", ",")

    return text.strip()

# =========================
# 🔹 LOAD Uni-MuMER (per-request, unloaded before Qwen)
# =========================
def load_unimer():
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
# 🔹 GET QWEN (singleton — loads once, stays alive)
# =========================
def get_qwen():
    global _qwen_llm, _qwen_sampling
    if _qwen_llm is None:
        print("🔄 Loading Qwen (first time)...")
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
# 🔹 QWEN CLEANING
# FIX: use proper chat format so model doesn't echo the prompt back
# =========================
def restructure_with_qwen(raw_lines, expected_lines):
    if not raw_lines:
        return []

    try:
        qwen, sampling = get_qwen()

        text = "\n".join(raw_lines)

        # Use chat-style formatting with clear system/user separation.
        # The previous single-block prompt caused the model to echo instructions.
        # With chat format, the model only sees "assistant turn starts here" and fills it.
        system_msg = (
            "You are an OCR post-processor. "
            "Fix spacing and broken words in the input text. "
            "Output ONLY the fixed text. "
            "Do NOT repeat instructions, rules, labels, or any explanation. "
            "Do NOT add new content. "
            "Preserve the original meaning exactly."
        )

        user_msg = f"Fix the spacing in this OCR output:\n\n{text}"

        # vLLM chat template format
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

        result = qwen.generate([prompt], sampling)[0].outputs[0].text.strip()

        # Strip any trailing <|im_end|> tokens vLLM may leave
        result = result.replace("<|im_end|>", "").strip()

        lines = [l.strip() for l in result.split("\n") if l.strip()]

        # Enforce line count
        if len(lines) > expected_lines:
            lines = lines[:expected_lines]
        while len(lines) < expected_lines:
            lines.append("")

        print("\n==============================")
        print("🧠 QWEN CLEANED OUTPUT")
        print("==============================")
        for i, line in enumerate(lines):
            print(f"Line {i+1}: {line}")
        print("==============================\n")

        return lines

    except Exception as e:
        print("⚠️ Qwen error:", e)
        traceback.print_exc()
        return raw_lines

    # ✅ Do NOT del qwen here — we want it to stay alive for next request

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

            # 🔥 Load Uni-MuMER fresh for this sheet
            unimer, sampling = load_unimer()

            content = await sheet.read()
            images  = convert_from_bytes(content)

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(sheet_dir, exist_ok=True)

            raw_lines = []
            ocr_counts = []

            # 🔹 Process each page with Uni-MuMER + OCR line count
            for page_idx, img in enumerate(images):
                arr = np.array(img.convert("RGB"))

                count, _ = segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer,
                    sampling_params=sampling,
                )
                print(f"📄 Page {page_idx+1}: Uni-MuMER lines={count}")

                # OCR line count collected per page image here
                # (kept same logic as original — reads from texts/ folder below)

            # 🔹 Read Uni-MuMER output
            latex_file = os.path.join(sheet_dir, "formulas_latex.json")

            if os.path.exists(latex_file):
                with open(latex_file, "r") as f:
                    data = json.load(f)

                print("\n==============================")
                print("📝 Uni-MuMER RAW OUTPUT")
                print("==============================")

                for i, v in enumerate(data.values()):
                    cleaned = clean_unimumer_output(v)
                    raw_lines.append(cleaned)
                    print(f"Line {i+1}: {cleaned}")

                print("==============================\n")

            # 🔥 UNLOAD Uni-MuMER before loading Qwen (free ~7GB)
            del unimer
            clear_gpu()
            print("🗑️ Uni-MuMER unloaded, GPU freed")

            # 🔹 OCR-based line count
            text_folder = os.path.join(sheet_dir, "texts")
            if os.path.exists(text_folder):
                for filename in os.listdir(text_folder):
                    filepath = os.path.join(text_folder, filename)
                    with open(filepath, "rb") as f:
                        txt = ocr_from_image(f.read())
                        if txt.strip():
                            ocr_counts.append(len(txt.split("\n")))

            expected_lines = get_clean_line_count(ocr_counts)

            print(f"📊 OCR line counts per image: {ocr_counts}")
            print(f"📊 Final expected lines: {expected_lines}")

            # 🔥 Qwen cleaning (singleton — fast after first load)
            structured_lines = restructure_with_qwen(raw_lines, expected_lines)

            # 🔹 Scoring
            total = 0
            max_total = 0

            for q in questions_data:
                sim = 0
                for line in structured_lines:
                    sim = max(sim, text_similarity(q["keyAnswer"], line))

                total += sim * q["marks"]
                max_total += q["marks"]

            results.append({
                "score": round(total, 2),
                "total": max_total,
                "lines": structured_lines
            })

            shutil.rmtree(sheet_dir, ignore_errors=True)

        return results[0] if results else {"score": 0, "total": 0}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}