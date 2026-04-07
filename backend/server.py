from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.ocr import ocr_from_image
from Utils.similarity import text_similarity
from Utils.image_similarity import image_similarity
import numpy as np
import os
import sys
import shutil
import json
import traceback
from typing import List
import time

app = FastAPI()

from evaluation import router as eval_router
app.include_router(eval_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔹 LOAD MODELS
# =========================
sys.path.insert(0, '/content/Uni-MuMER')

from vllm import LLM, SamplingParams

# ---- Uni-MuMER ----
UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
unimer_llm = None
unimer_sampling = SamplingParams(temperature=0, max_tokens=512)

try:
    print("🔄 Loading Uni-MuMER...")
    unimer_llm = LLM(
        model=UNIMER_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85
    )
    print("✅ Uni-MuMER loaded")
except Exception as e:
    print(f"❌ Uni-MuMER failed: {e}")

# ---- QWEN (LLM) ----
QWEN_MODEL_PATH = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"
qwen_llm = None

try:
    print("🔄 Loading Qwen...")
    qwen_llm = LLM(
        model=QWEN_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.75
    )
    print("✅ Qwen loaded")
except Exception as e:
    print(f"❌ Qwen failed: {e}")

# =========================
# 🔹 CLEANER (fallback)
# =========================
def clean_unimumer_output(text):
    if any(c in text for c in ['\\', '{', '}', '^', '_']):
        return text

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
# 🔹 QWEN RESTRUCTURE
# =========================
def restructure_with_qwen(raw_lines: list, ocr_line_count: int) -> list:

    if not qwen_llm:
        print("⚠️ Qwen not available — fallback")
        return [clean_unimumer_output(l) for l in raw_lines]

    raw_text = "\n".join(raw_lines)

    prompt = f"""You are an expert at reconstructing handwritten answer sheets.

You MUST output EXACTLY {ocr_line_count} lines.

Rules:
- Merge spaced letters into words
- Keep ALL LaTeX exactly as-is
- Do NOT change meaning
- Do NOT explain anything
- Do NOT add numbering
- Output ONLY clean lines

Raw OCR:
{raw_text}

Final Answer ({ocr_line_count} lines):"""

    try:
        sampling = SamplingParams(temperature=0, max_tokens=1024)
        outputs = qwen_llm.generate([prompt], sampling)
        result = outputs[0].outputs[0].text.strip()

        lines = [l.strip() for l in result.split("\n") if l.strip()]

        # 🔥 FORCE EXACT LINE COUNT
        if len(lines) > ocr_line_count:
            lines = lines[:ocr_line_count]
        elif len(lines) < ocr_line_count:
            while len(lines) < ocr_line_count:
                lines.append(lines[-1] if lines else "")

        return lines

    except Exception as e:
        print(f"⚠️ Qwen error: {e}")
        return [clean_unimumer_output(l) for l in raw_lines]

# =========================
# 🔹 API
# =========================
@app.post("/similarity")
async def similarity(
    request: Request,
    questions: str = Form(...),
    answer_sheets: List[UploadFile] = File(...)
):
    try:
        start_time = time.time()
        questions_data = json.loads(questions)

        form = await request.form()
        diagrams = {}
        sheets = []

        for key, value in form.items():
            if isinstance(value, UploadFile):
                if key.startswith("diagram_"):
                    idx = int(key.split("_")[1])
                    diagrams[idx] = Image.open(BytesIO(await value.read()))
                elif key == "answer_sheets":
                    sheets.append(value)

        results = []

        for sheet_idx, sheet in enumerate(sheets):
            content = await sheet.read()
            images = convert_from_bytes(content)

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(f"{sheet_dir}/texts", exist_ok=True)
            os.makedirs(f"{sheet_dir}/formulas", exist_ok=True)

            all_texts = []

            for img in images:
                arr = np.array(img.convert("RGB"))
                segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer_llm,
                    sampling_params=unimer_sampling
                )

            # OCR
            text_files = os.listdir(f"{sheet_dir}/texts")
            for f in text_files:
                with open(f"{sheet_dir}/texts/{f}", "rb") as file:
                    txt = ocr_from_image(file.read())
                    if txt.strip():
                        all_texts.append(txt.strip())

            # Uni-MuMER output
            latex_file = f"{sheet_dir}/formulas_latex.json"
            all_formulas = []

            if os.path.exists(latex_file):
                latex_data = json.load(open(latex_file))
                raw = [v for v in latex_data.values() if v]

                ocr_lines = len(all_texts) if all_texts else len(raw)

                print(f"🔄 Qwen restructuring ({ocr_lines} lines)")
                all_formulas = restructure_with_qwen(raw, ocr_lines)

                for i, l in enumerate(all_formulas):
                    print(f"{i+1}: {l}")

            # scoring
            total = 0
            max_total = 0

            for q in questions_data:
                sim = 0
                if q.get("keyAnswer"):
                    for line in all_formulas:
                        sim = max(sim, text_similarity(q["keyAnswer"], line))

                score = sim * q.get("marks", 0)
                total += score
                max_total += q.get("marks", 0)

            results.append({
                "score": round(total, 2),
                "total": max_total
            })

            shutil.rmtree(sheet_dir, ignore_errors=True)

        return results[0]

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}