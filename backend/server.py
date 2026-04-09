from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.similarity import text_similarity

import numpy as np
import os, sys, shutil, json, traceback, gc
from typing import List

import torch
from vllm import LLM, SamplingParams

# =========================
# 🔹 APP INIT
# =========================

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
    expose_headers=["*"],
    max_age=3600,
)

# =========================
# 🔹 PATH SETUP
# =========================

sys.path.insert(0, '/content/Uni-MuMER')

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/drive/MyDrive/models/Qwen2.5-3B-Instruct"

# =========================
# 🔹 GPU CLEAN
# =========================

def clear_gpu():
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

# =========================
# 🔹 CLEAN Uni-MuMER OUTPUT
# Merges spaced-out single chars: G r a d i e n t → Gradient
# Preserves LaTeX math syntax ($ ... $)
# =========================

def clean_unimumer_output(text: str) -> str:
    import re

    if not text:
        return text

    # Split on LaTeX math segments to protect them
    parts = re.split(r'(\$[^$]*\$)', text)
    cleaned_parts = []

    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            # LaTeX block — preserve as-is
            cleaned_parts.append(part)
        else:
            # Merge spaced single characters: "G r a d" → "Grad"
            # Only merge sequences of single chars separated by spaces
            merged = re.sub(
                r'(?<!\S)((?:[A-Za-z0-9] )+[A-Za-z0-9])(?!\S)',
                lambda m: m.group(0).replace(' ', ''),
                part
            )
            cleaned_parts.append(merged)

    return ''.join(cleaned_parts)

# =========================
# 🔹 LOAD Uni-MuMER
# =========================

def load_unimer():
    print("🔄 Loading Uni-MuMER...")
    llm = LLM(
        model=UNIMER_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.90,
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
        gpu_memory_utilization=0.40,
        max_model_len=2048,
    )
    print("✅ Qwen loaded")
    return llm

# =========================
# 🔹 QWEN RESTRUCTURE
# =========================

def restructure_with_qwen(raw_lines: list, expected_lines: int) -> list:
    if not raw_lines:
        return []

    qwen_llm = None
    try:
        qwen_llm = load_qwen()

        raw_text = "\n".join(raw_lines)

        prompt = f"""You are an expert system that reconstructs noisy OCR text from handwritten answers.

RULES:
* Fix broken words (G r a d i e n t → Gradient)
* Merge characters into words
* Correct obvious OCR mistakes only when clear
* DO NOT change meaning
* DO NOT explain
* DO NOT summarize
* KEEP technical correctness
* OUTPUT EXACTLY {expected_lines} lines

INPUT:
{raw_text}

OUTPUT:
"""

        sampling = SamplingParams(temperature=0, max_tokens=1024)
        outputs  = qwen_llm.generate([prompt], sampling)
        result   = outputs[0].outputs[0].text.strip()

        lines = [l.strip() for l in result.split("\n") if l.strip()]

        # Enforce exact line count
        if len(lines) > expected_lines:
            lines = lines[:expected_lines]
        elif len(lines) < expected_lines:
            while len(lines) < expected_lines:
                lines.append(lines[-1] if lines else "")

        # Log output
        print("\n==============================")
        print("🧠 QWEN STRUCTURED OUTPUT")
        print("==============================")
        for i, line in enumerate(lines):
            print(f"Line {i+1}: {line}")
        print("==============================\n")

        return lines

    except Exception as e:
        print("⚠️ Qwen error:", str(e))
        return raw_lines  # fallback

    finally:
        try:
            if qwen_llm is not None:
                del qwen_llm
        except Exception:
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
    answer_sheets: List[UploadFile] = File(...),
):
    try:
        print("\n🚀 New request received")

        questions_data = json.loads(questions)

        if not answer_sheets:
            return {"error": "No answer sheets uploaded"}

        results = []

        # =========================
        # 🔥 LOAD Uni-MuMER ONCE (outside sheet loop)
        # =========================
        unimer_llm, unimer_sampling = load_unimer()

        for sheet_idx, sheet in enumerate(answer_sheets):

            content = await sheet.read()
            images  = convert_from_bytes(content)

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(sheet_dir, exist_ok=True)

            raw_lines   = []
            line_counts = []

            # 🔹 Segmentation — pass Uni-MuMER into each page
            for page_idx, img in enumerate(images):
                arr = np.array(img.convert("RGB"))

                count, _ = segment_lines_and_find_diagrams(
                    arr,
                    output_folder=sheet_dir,
                    llm=unimer_llm,
                    sampling_params=unimer_sampling,
                )

                line_counts.append(count)
                print(f"📄 Sheet {sheet_idx+1}, Page {page_idx+1}: {count} lines detected")

            # 🔹 Read Uni-MuMER output and clean it
            latex_file = os.path.join(sheet_dir, "formulas_latex.json")

            if os.path.exists(latex_file):
                with open(latex_file, "r") as f:
                    data = json.load(f)

                raw_lines = []
                print("\n==============================")
                print("📝 Uni-MuMER RAW OUTPUT")
                print("==============================")
                for i, (k, v) in enumerate(data.items()):
                    if v:
                        cleaned = clean_unimumer_output(v)
                        raw_lines.append(cleaned)
                        print(f"Line {i+1}: {cleaned}")
                print("==============================\n")
            else:
                print(f"⚠️ No latex file found at {latex_file}")

            expected_lines = max(line_counts) if line_counts else len(raw_lines)
            expected_lines = max(1, expected_lines)
            print(f"📊 Expected lines: {expected_lines}")

            # 🔥 QWEN restructure
            structured_lines = restructure_with_qwen(raw_lines, expected_lines)

            # =========================
            # 🔹 SCORING
            # =========================
            total     = 0
            max_total = 0

            for q in questions_data:
                sim = 0.0

                if q.get("keyAnswer"):
                    for line in structured_lines:
                        sim = max(sim, text_similarity(q["keyAnswer"], line))

                marks      = q.get("marks", 0)
                total     += sim * marks
                max_total += marks

            results.append({
                "score": round(total, 2),
                "total": max_total,
                "lines": structured_lines,
            })

            shutil.rmtree(sheet_dir, ignore_errors=True)

        # 🔥 FREE Uni-MuMER after all sheets are done
        try:
            del unimer_llm
        except Exception:
            pass
        clear_gpu()

        return results[0] if results else {"score": 0, "total": 0}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}