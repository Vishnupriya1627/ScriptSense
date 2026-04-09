from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.ocr import ocr_from_image
from Utils.similarity import text_similarity

import numpy as np
import os, sys, shutil, json, traceback, gc
from typing import List

import torch
from vllm import LLM, SamplingParams

app = FastAPI()

# =========================
# 🔹 CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔹 PATH SETUP
# =========================
sys.path.insert(0, '/content/Uni-MuMER')

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/models/Qwen2.5-3B-Instruct"

# =========================
# 🔹 GPU CLEAN
# =========================
def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

# =========================
# 🔹 LOADERS
# =========================
def load_unimer():
    print("🔄 Loading Uni-MuMER...")
    llm = LLM(
        model=UNIMER_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.95,
        max_model_len=1024
    )
    return llm, SamplingParams(temperature=0, max_tokens=512)

def load_qwen():
    print("🔄 Loading Qwen...")
    llm = LLM(
        model=QWEN_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.50,
        max_model_len=1024
    )
    return llm

# =========================
# 🔹 CLEAN Uni-MuMER TEXT
# =========================
def clean_unimumer_output(text):
    tokens = text.split(' ')
    result = []
    i = 0

    while i < len(tokens):
        if len(tokens[i]) == 1:
            word = tokens[i]
            while i + 1 < len(tokens) and len(tokens[i + 1]) == 1:
                word += tokens[i + 1]
                i += 1
            result.append(word)
        else:
            result.append(tokens[i])
        i += 1

    return ' '.join(result)

# =========================
# 🔹 SMART LINE SPLIT (🔥 CORE FIX)
# =========================
def split_into_lines(text, n_lines):
    words = text.split()
    if not words:
        return []

    avg = max(1, len(words) // n_lines)

    lines = []
    for i in range(n_lines):
        start = i * avg
        end = (i + 1) * avg if i < n_lines - 1 else len(words)
        lines.append(" ".join(words[start:end]))

    return lines

# =========================
# 🔹 QWEN CLEAN ONLY
# =========================
def clean_with_qwen(lines):
    qwen_llm = None
    try:
        qwen_llm = load_qwen()

        text = "\n".join(lines)

        prompt = f"""
Fix spacing and broken words ONLY.

Rules:
- Do NOT change meaning
- Do NOT add anything
- Do NOT repeat prompt
- Output same lines

{text}
"""

        sampling = SamplingParams(temperature=0, max_tokens=512)
        outputs = qwen_llm.generate([prompt], sampling)

        result = outputs[0].outputs[0].text.strip()
        cleaned = [l.strip() for l in result.split("\n") if l.strip()]

        return cleaned

    except Exception as e:
        print("⚠️ Qwen error:", e)
        return lines

    finally:
        if qwen_llm:
            del qwen_llm
        clear_gpu()

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
        results = []

        # 🔥 Load Uni-MuMER
        unimer_llm, unimer_sampling = load_unimer()

        for sheet_idx, sheet in enumerate(answer_sheets):

            content = await sheet.read()
            images = convert_from_bytes(content)

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(sheet_dir, exist_ok=True)

            page_line_counts = []

            for img in images:
                arr = np.array(img.convert("RGB"))

                count, _ = segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer_llm,
                    sampling_params=unimer_sampling,
                )
                page_line_counts.append(count)

            # 🔥 FREE Uni-MuMER
            del unimer_llm
            clear_gpu()

            # =========================
            # 🔹 READ Uni-MuMER OUTPUT
            # =========================
            latex_file = f"{sheet_dir}/formulas_latex.json"

            raw_text = ""
            if os.path.exists(latex_file):
                data = json.load(open(latex_file))

                for v in data.values():
                    if v:
                        raw_text += " " + clean_unimumer_output(v)

            # =========================
            # 🔹 OCR LINE COUNT (CLAMPED)
            # =========================
            ocr_counts = []

            for f in os.listdir(f"{sheet_dir}/texts"):
                with open(f"{sheet_dir}/texts/{f}", "rb") as file:
                    txt = ocr_from_image(file.read())
                    if txt.strip():
                        ocr_counts.append(len(txt.split("\n")))

            ocr_lines = max(ocr_counts) if ocr_counts else 3

            # 🔥 CRITICAL FIX
            ocr_lines = min(max(ocr_lines, 2), 12)

            print(f"📊 Final line count used: {ocr_lines}")

            # =========================
            # 🔹 SPLIT TEXT
            # =========================
            structured_lines = split_into_lines(raw_text, ocr_lines)

            # =========================
            # 🔹 CLEAN WITH QWEN
            # =========================
            structured_lines = clean_with_qwen(structured_lines)

            # =========================
            # 🔹 SCORING
            # =========================
            total = 0
            max_total = 0

            for q in questions_data:
                sim = 0
                if q.get("keyAnswer"):
                    for line in structured_lines:
                        sim = max(sim, text_similarity(q["keyAnswer"], line))

                marks = q.get("marks", 0)
                total += sim * marks
                max_total += marks

            results.append({
                "score": round(total, 2),
                "total": max_total,
                "lines": structured_lines
            })

            shutil.rmtree(sheet_dir, ignore_errors=True)

        return results[0]

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}