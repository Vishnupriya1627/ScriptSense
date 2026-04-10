from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.ocr import ocr_from_image
from Utils.similarity import text_similarity

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
)

sys.path.insert(0, '/content/Uni-MuMER')

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"

# =========================
# 🔹 GLOBAL MODELS
# =========================
unimer_llm = None
unimer_sampling = None
qwen_llm = None

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
# 🔹 LOADERS
# =========================
def load_unimer():
    print("🔄 Loading Uni-MuMER...")
    llm = LLM(
        model=UNIMER_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=2048
    )
    print("✅ Uni-MuMER loaded")
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
    print("✅ Qwen loaded")
    return llm

# =========================
# 🔹 STARTUP LOAD
# =========================
@app.on_event("startup")
def startup_event():
    global unimer_llm, unimer_sampling, qwen_llm

    clear_gpu()
    unimer_llm, unimer_sampling = load_unimer()
    qwen_llm = load_qwen()

# =========================
# 🔹 CLEAN Uni-MuMER TEXT
# =========================
def clean_unimumer_output(text):
    if not text:
        return ""

    text = re.sub(
        r'(?<!\S)(?:[A-Za-z]\s){2,}[A-Za-z](?!\S)',
        lambda m: m.group(0).replace(" ", ""),
        text
    )

    text = re.sub(r'\s+', ' ', text)
    text = text.replace(" .", ".").replace(" ,", ",")

    return text.strip()

# =========================
# 🔹 OCR LINE COUNT
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
# 🔹 SPLIT LINES
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
# 🔹 QWEN RESTRUCTURE
# =========================
def restructure_with_qwen(lines):
    global qwen_llm
    try:
        text = "\n".join(lines)

        prompt = f"""
Rearrange text into natural handwritten-style lines.

Rules:
- Do NOT add content
- Do NOT remove content
- Keep meaning same
- Keep similar number of lines
- No explanations

{text}
"""

        sampling = SamplingParams(temperature=0, max_tokens=512)
        result = qwen_llm.generate([prompt], sampling)[0].outputs[0].text.strip()

        new_lines = [l.strip() for l in result.split("\n") if l.strip()]
        return new_lines if new_lines else lines

    except Exception as e:
        print("⚠️ Qwen restructure error:", e)
        return lines

# =========================
# 🔹 QWEN CLEAN
# =========================
def clean_with_qwen(lines):
    global qwen_llm
    try:
        text = "\n".join(lines)

        prompt = f"""
Fix spacing and broken words ONLY.

Rules:
- Do NOT change meaning
- Do NOT add content
- Keep SAME number of lines

{text}
"""

        sampling = SamplingParams(temperature=0, max_tokens=512)
        result = qwen_llm.generate([prompt], sampling)[0].outputs[0].text.strip()

        cleaned = [l.strip() for l in result.split("\n")]

        if len(cleaned) != len(lines):
            return lines

        return cleaned

    except Exception as e:
        print("⚠️ Qwen clean error:", e)
        return lines

# =========================
# 🔹 HEALTH
# =========================
@app.get("/health")
def health():
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
        print("\n🚀 New request")

        global unimer_llm, unimer_sampling

        questions_data = json.loads(questions)
        results = []

        for sheet_idx, sheet in enumerate(answer_sheets):

            content = await sheet.read()
            images = convert_from_bytes(content)

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(sheet_dir, exist_ok=True)

            raw_text = ""

            # =========================
            # 🔹 PROCESS PAGES
            # =========================
            for img in images:
                arr = np.array(img.convert("RGB"))

                segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer_llm,
                    sampling_params=unimer_sampling,
                )

            # =========================
            # 🔹 READ Uni-MuMER OUTPUT
            # =========================
            latex_file = f"{sheet_dir}/formulas_latex.json"

            if os.path.exists(latex_file):
                data = json.load(open(latex_file))
                for v in data.values():
                    raw_text += " " + clean_unimumer_output(v)

            # =========================
            # 🔹 OCR LINE COUNT
            # =========================
            ocr_counts = []

            text_folder = f"{sheet_dir}/texts"
            if os.path.exists(text_folder):
                for f in os.listdir(text_folder):
                    with open(f"{text_folder}/{f}", "rb") as file:
                        txt = ocr_from_image(file.read())
                        if txt.strip():
                            ocr_counts.append(len(txt.split("\n")))

            final_lines = get_clean_line_count(ocr_counts)

            print(f"📊 OCR counts: {ocr_counts}")
            print(f"📊 Final line count: {final_lines}")

            # =========================
            # 🔹 STRUCTURE PIPELINE
            # =========================
            structured_lines = split_into_lines(raw_text, final_lines)
            structured_lines = restructure_with_qwen(structured_lines)
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