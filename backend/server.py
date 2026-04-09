from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.ocr import ocr_from_image
from Utils.similarity import text_similarity
import numpy as np
import os, sys, shutil, json, traceback
from typing import List
import torch, gc

# =========================
# 🔹 APP
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
# 🔹 PATH
# =========================
sys.path.insert(0, '/content/Uni-MuMER')

from vllm import LLM, SamplingParams

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH = "/content/models/Qwen2.5-3B-Instruct"

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

    print("✅ Uni-MuMER loaded")
    return llm, SamplingParams(temperature=0, max_tokens=512)


def load_qwen():
    print("🔄 Loading Qwen...")

    llm = LLM(
        model=QWEN_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.55,   # safe balance
        max_model_len=2048
    )

    print("✅ Qwen loaded")
    return llm

# =========================
# 🔹 CLEAN OCR TEXT
# =========================
def clean_unimumer_output(text):
    tokens = text.split(' ')
    result = []
    i = 0

    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
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
# 🔹 QWEN STRUCTURING
# =========================
def restructure_with_qwen(raw_lines: list, ocr_line_count: int):

    if not raw_lines:
        return []

    try:
        qwen_llm = load_qwen()

        raw_text = "\n".join(raw_lines)

        prompt = f"""You are an expert at reconstructing handwritten answer sheets.

You MUST output EXACTLY {ocr_line_count} lines.

Rules:
- Merge spaced letters into words
- Keep ALL LaTeX exactly as-is
- Do NOT change meaning
- Do NOT explain anything
- Output ONLY clean lines

Raw OCR:
{raw_text}

Final Answer ({ocr_line_count} lines):"""

        sampling = SamplingParams(temperature=0, max_tokens=1024)
        outputs = qwen_llm.generate([prompt], sampling)

        result = outputs[0].outputs[0].text.strip()

        print("\n🧠 ================= QWEN STRUCTURED OUTPUT =================")
        print(result)
        print("===========================================================\n")

        lines = [l.strip() for l in result.split("\n") if l.strip()]

        if len(lines) > ocr_line_count:
            lines = lines[:ocr_line_count]
        elif len(lines) < ocr_line_count:
            while len(lines) < ocr_line_count:
                lines.append(lines[-1] if lines else "")

        return lines

    except Exception as e:
        print(f"⚠️ Qwen error: {e}")
        return [clean_unimumer_output(l) for l in raw_lines]

    finally:
        # 🔥 IMPORTANT: free Qwen after use
        del qwen_llm
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
        sheets = answer_sheets

        if not sheets:
            return {"error": "No answer sheets uploaded"}

        results = []

        # =========================
        # 🔥 STEP 1: LOAD UNIMuMER
        # =========================
        unimer_llm, unimer_sampling = load_unimer()

        for sheet_idx, sheet in enumerate(sheets):

            content = await sheet.read()
            images = convert_from_bytes(content)

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(f"{sheet_dir}/texts", exist_ok=True)
            os.makedirs(f"{sheet_dir}/formulas", exist_ok=True)

            all_texts = []
            page_line_counts = []

            # =========================
            # 🔹 SEGMENTATION
            # =========================
            for img in images:
                arr = np.array(img.convert("RGB"))

                text_count, _ = segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer_llm,
                    sampling_params=unimer_sampling
                )

                page_line_counts.append(text_count)

        # 🔥 FREE Uni-MuMER BEFORE QWEN
        del unimer_llm
        clear_gpu()

        for sheet_idx, sheet in enumerate(sheets):

            sheet_dir = f"output/sheet_{sheet_idx}"

            # =========================
            # 🔹 OCR
            # =========================
            all_texts = []

            for f in os.listdir(f"{sheet_dir}/texts"):
                with open(f"{sheet_dir}/texts/{f}", "rb") as file:
                    txt = ocr_from_image(file.read())
                    if txt.strip():
                        all_texts.append(txt.strip())

            ocr_lines = max(1, len(all_texts))

            # =========================
            # 🔹 LOAD LATEX
            # =========================
            latex_file = f"{sheet_dir}/formulas_latex.json"

            if os.path.exists(latex_file):
                latex_data = json.load(open(latex_file))
                raw = [v for v in latex_data.values() if v]
            else:
                raw = all_texts

            # =========================
            # 🔥 STEP 2: QWEN STRUCTURE
            # =========================
            structured_lines = restructure_with_qwen(raw, ocr_lines)

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
                "structured_output": structured_lines   # 👈 RETURN THIS ALSO
            })

            shutil.rmtree(sheet_dir, ignore_errors=True)

        return results[0]

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}