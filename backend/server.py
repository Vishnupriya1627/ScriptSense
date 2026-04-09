from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.similarity import text_similarity

import numpy as np
import os, sys, shutil, json, traceback, gc, re
from typing import List

import torch
import cv2
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
)

# =========================
# 🔹 PATHS
# =========================
sys.path.insert(0, '/content/Uni-MuMER')

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"

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
# 🔥 LINE DETECTION (IMAGE BASED)
# =========================
def detect_handwritten_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    projection = np.sum(thresh, axis=1)
    threshold = np.max(projection) * 0.2

    lines = 0
    in_line = False

    for val in projection:
        if val > threshold and not in_line:
            lines += 1
            in_line = True
        elif val <= threshold:
            in_line = False

    return max(1, lines)

# =========================
# 🔹 CLEAN Uni-MuMER OUTPUT
# =========================
def clean_unimumer_output(text: str) -> str:
    if not text:
        return text

    # merge spaced characters
    text = re.sub(
        r'(?<!\S)((?:[A-Za-z] )+[A-Za-z])(?!\S)',
        lambda m: m.group(0).replace(' ', ''),
        text
    )

    # fix spacing issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)\.', r'\1. ', text)

    text = text.replace(" .", ".").replace(" ,", ",")

    return text.strip()

# =========================
# 🔹 LOAD Uni-MuMER
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
# 🔹 LOAD QWEN
# =========================
def load_qwen():
    print("🔄 Loading Qwen...")
    llm = LLM(
        model=QWEN_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.75,
        max_model_len=1024,
    )
    print("✅ Qwen loaded")
    return llm

# =========================
# 🔹 QWEN CLEANING ONLY
# =========================
def restructure_with_qwen(raw_lines, expected_lines):
    if not raw_lines:
        return []

    qwen = None
    try:
        qwen = load_qwen()

        text = "\n".join(raw_lines)

        prompt = f"""
You are an OCR post-processor.

TASK:
Fix spacing and broken words in the text.

STRICT RULES:
- Do NOT add new content
- Do NOT explain anything
- Do NOT repeat the prompt
- Do NOT include words like "RULES", "Answer", etc.
- Keep SAME number of lines: {expected_lines}
- Preserve original meaning exactly

INPUT:
{text}

OUTPUT (only cleaned text):
"""

        sampling = SamplingParams(temperature=0, max_tokens=1024)
        result = qwen.generate([prompt], sampling)[0].outputs[0].text.strip()

        lines = [l.strip() for l in result.split("\n") if l.strip()]

        # enforce exact line count
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
        return raw_lines

    finally:
        try:
            del qwen
        except:
            pass
        clear_gpu()

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

            # 🔥 LOAD Uni-MuMER
            unimer, sampling = load_unimer()

            content = await sheet.read()
            images  = convert_from_bytes(content)

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(sheet_dir, exist_ok=True)

            raw_lines = []
            detected_lines_all = []

            # 🔹 Process pages
            for page_idx, img in enumerate(images):
                arr = np.array(img.convert("RGB"))

                # 🔥 real line detection
                detected_lines_all.append(detect_handwritten_lines(arr))

                count, _ = segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer,
                    sampling_params=sampling,
                )

                print(f"📄 Page {page_idx+1}: Uni-MuMER lines={count}")

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

            # 🔥 determine expected lines
            expected_lines = int(np.mean(detected_lines_all)) if detected_lines_all else len(raw_lines)
            expected_lines = max(1, expected_lines)

            print("📊 Detected lines:", detected_lines_all)
            print("📊 Expected lines:", expected_lines)

            # 🔥 unload Uni-MuMER
            del unimer
            clear_gpu()

            # 🔥 Qwen cleaning
            structured_lines = restructure_with_qwen(raw_lines, expected_lines)

            # 🔹 scoring
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