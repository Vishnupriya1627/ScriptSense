from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
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
import re
import shutil
import json
import traceback
from typing import List
import time

app = FastAPI()

from evaluation import router as eval_router
app.include_router(eval_router)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ── Load Uni-MuMER at startup ──────────────────────────────────
UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
sys.path.insert(0, '/content/Uni-MuMER')

unimer_llm = None
unimer_sampling = None

try:
    from vllm import LLM, SamplingParams
    print("🔄 Loading Uni-MuMER model at startup...")
    unimer_llm = LLM(
        model=UNIMER_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.95
    )
    unimer_sampling = SamplingParams(temperature=0, max_tokens=512)
    print("✅ Uni-MuMER loaded and ready!")
except Exception as e:
    print(f"⚠️ Uni-MuMER failed to load: {e}")
    unimer_llm = None
# ───────────────────────────────────────────────────────────────


# ── Uni-MuMER output cleaner ───────────────────────────────────
def clean_unimumer_output(text):
    """
    Fixes spaced-out characters from Uni-MuMER text output.
    'G r a d i e n t   d e s c e n t' → 'Gradient descent'
    Math lines like 'P \\times \\frac { 81 } { 512 }' are left untouched.
    """
    # If it contains LaTeX math symbols, keep as-is
    if any(c in text for c in ['\\', '{', '}', '^', '_']):
        return text

    # Pure text — merge single spaced-out characters into words
    tokens = text.split(' ')
    result = []
    i = 0
    while i < len(tokens):
        # If current token is a single letter, start merging
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            word = tokens[i]
            while i + 1 < len(tokens) and len(tokens[i + 1]) == 1 and tokens[i + 1].isalpha():
                word += tokens[i + 1]
                i += 1
            result.append(word)
        else:
            result.append(tokens[i])
        i += 1
    return ' '.join(result)
# ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "unimer_loaded": unimer_llm is not None
    }

@app.post("/similarity")
async def similarity(
    request: Request,
    questions: str = Form(...),
    answer_sheets: List[UploadFile] = File(...),
):
    try:
        start_time = time.time()
        print("\n" + "="*60)
        print(f"📨 REQUEST RECEIVED at {time.strftime('%H:%M:%S')}")
        print("="*60)
        
        try:
            questions_data = json.loads(questions)
            print(f"📝 Questions: {len(questions_data)} question(s)")
            for i, q in enumerate(questions_data):
                print(f"   Q{i+1}: marks={q.get('marks', 0)}, textWeight={q.get('textWeight', 1.0)}, diagramWeight={q.get('diagramWeight', 0.0)}")
        except Exception as e:
            print(f"❌ Failed to parse questions: {e}")
            return JSONResponse(status_code=400, content={"error": f"Invalid questions format: {str(e)}"})

        form = await request.form()
        diagrams = {}
        answer_sheets_list = []
        
        for key, value in form.items():
            if isinstance(value, UploadFile):
                if key.startswith('diagram_'):
                    try:
                        idx = int(key.split('_')[1])
                        content = await value.read()
                        if content and len(content) > 100:
                            diagrams[idx] = Image.open(BytesIO(content))
                            print(f"   🖼️ Loaded diagram for Q{idx+1}: {len(content)} bytes")
                    except Exception as e:
                        print(f"   ⚠️ Error loading diagram {key}: {e}")
                elif key == 'answer_sheets':
                    answer_sheets_list.append(value)
        
        if not answer_sheets_list and answer_sheets:
            answer_sheets_list = answer_sheets
        
        print(f"\n📄 Answer sheets to process: {len(answer_sheets_list)}")
        
        if not answer_sheets_list:
            return JSONResponse(status_code=400, content={"error": "No answer sheets provided"})

        all_results = []
        
        for sheet_idx, sheet in enumerate(answer_sheets_list):
            try:
                print(f"\n[{time.time()-start_time:.1f}s] 📄 Processing sheet {sheet_idx+1}: {sheet.filename}")
                
                sheet_content = await sheet.read()
                print(f"   📦 Read {len(sheet_content)} bytes")
                
                try:
                    images = convert_from_bytes(sheet_content)
                    print(f"   ✅ Converted to {len(images)} page(s)")
                except Exception as e:
                    print(f"   ❌ PDF conversion error: {e}")
                    continue

                sheet_output_dir = f"output/sheet_{sheet_idx}"
                os.makedirs(f"{sheet_output_dir}/texts", exist_ok=True)
                os.makedirs(f"{sheet_output_dir}/diagrams", exist_ok=True)
                os.makedirs(f"{sheet_output_dir}/formulas", exist_ok=True)
                
                all_texts = []
                all_diagrams = []
                
                for page_idx, image in enumerate(images):
                    try:
                        image = image.convert("RGB")
                        image_array = np.array(image)
                        text_count, diagram_count = segment_lines_and_find_diagrams(
                            image_array,
                            output_folder=sheet_output_dir,
                            min_height_threshold=15,
                            padding=5,
                            min_contour_width=1000,
                            llm=unimer_llm,
                            sampling_params=unimer_sampling
                        )
                        print(f"   📄 Page {page_idx+1}: {text_count} lines, {diagram_count} diagrams")
                    except Exception as e:
                        print(f"   ⚠️ Segmentation error on page {page_idx+1}: {e}")
                
                # ===== OCR (kept for fallback, not used in scoring) =====
                if os.path.exists(f"{sheet_output_dir}/texts"):
                    text_files = sorted([f for f in os.listdir(f"{sheet_output_dir}/texts") if f.endswith('.png')])
                    for text_file in text_files:
                        try:
                            text_path = os.path.join(f"{sheet_output_dir}/texts", text_file)
                            with open(text_path, "rb") as f:
                                ocr_result = ocr_from_image(f.read())
                                if ocr_result and ocr_result.strip():
                                    all_texts.append(ocr_result.strip())
                        except Exception as e:
                            print(f"   ⚠️ OCR error: {e}")

                # ===== READ UNI-MUMER RESULTS =====
                all_formulas = []
                latex_file = f"{sheet_output_dir}/formulas_latex.json"
                if os.path.exists(latex_file):
                    try:
                        latex_data = json.load(open(latex_file))
                        # Clean spaced-out characters from Uni-MuMER output
                        all_formulas = [clean_unimumer_output(v) for v in latex_data.values() if v]
                        print(f"   ➕ UniMuMER lines: {len(all_formulas)}")
                        for i, line in enumerate(all_formulas):
                            print(f"      Line {i+1}: {line}")
                    except Exception as e:
                        print(f"   ⚠️ Formula read error: {e}")
                else:
                    print(f"   ℹ️ No UniMuMER results")

                # ===== GET DIAGRAMS =====
                if os.path.exists(f"{sheet_output_dir}/diagrams"):
                    for f in os.listdir(f"{sheet_output_dir}/diagrams"):
                        if f.endswith(".png"):
                            try:
                                all_diagrams.append(Image.open(os.path.join(f"{sheet_output_dir}/diagrams", f)))
                            except Exception as e:
                                print(f"   ⚠️ Diagram load error: {e}")

                print(f"   📊 UniMuMER: {len(all_formulas)} line(s), Diagrams: {len(all_diagrams)}")

                total_obtained = 0
                total_possible = 0
                breakdown = []

                for q_idx, q_data in enumerate(questions_data):
                    print(f"\n   [{time.time()-start_time:.1f}s] 🔍 Question {q_idx + 1}...")
                    
                    # ===== UNI-MUMER ONLY FOR SCORING =====
                    text_sim = 0.0
                    if q_data.get('keyAnswer'):
                        try:
                            all_candidates = []

                            # Uni-MuMER individual lines
                            for f in all_formulas:
                                all_candidates.append(('UniMuMER', f))

                            # Uni-MuMER all lines joined as one block
                            if all_formulas:
                                joined = " ".join(all_formulas)
                                all_candidates.append(('UniMuMER-joined', joined))

                            # Score all candidates
                            scored = []
                            for source, text in all_candidates:
                                sim = text_similarity(q_data['keyAnswer'], text)
                                scored.append((sim, source, text))
                                print(f"      [{source}] {text[:60]} → {sim:.3f}")

                            if scored:
                                scored.sort(reverse=True)
                                text_sim = scored[0][0]
                                best_source = scored[0][1]
                                best_text = scored[0][2]
                                print(f"      🏆 Best: [{best_source}] {best_text[:60]} → {text_sim:.3f}")

                        except Exception as e:
                            print(f"      ⚠️ Similarity error: {e}")
                    
                    # ===== DIAGRAM SIMILARITY =====
                    diagram_sim = 0.0
                    if all_diagrams and q_idx in diagrams:
                        try:
                            key_diagram = diagrams[q_idx]
                            sims = [image_similarity(key_diagram, [d])[0] for d in all_diagrams]
                            diagram_sim = max(sims) if sims else 0.0
                            print(f"      🖼️ Best diagram similarity: {diagram_sim:.3f}")
                        except Exception as e:
                            print(f"      ⚠️ Diagram similarity error: {e}")
                    
                    marks = q_data.get('marks', 0)
                    text_weight = q_data.get('textWeight', 1.0)
                    diagram_weight = q_data.get('diagramWeight', 0.0)
                    
                    if q_idx not in diagrams:
                        diagram_weight = 0.0
                        text_weight = 1.0
                        print(f"      📝 No diagram key - text only")
                    elif len(all_diagrams) == 0:
                        diagram_weight = 0.0
                        text_weight = 1.0
                        print(f"      📝 No student diagram - text only")
                    else:
                        total_weight = text_weight + diagram_weight
                        if abs(total_weight - 1.0) > 0.01 and total_weight > 0:
                            text_weight = text_weight / total_weight
                            diagram_weight = diagram_weight / total_weight

                    weighted_similarity = (text_sim * text_weight) + (diagram_sim * diagram_weight)
                    obtained = weighted_similarity * marks
                    
                    print(f"      📊 ({text_sim:.3f} × {text_weight:.2f}) + ({diagram_sim:.3f} × {diagram_weight:.2f}) = {weighted_similarity:.3f}")
                    print(f"      ✅ Score: {obtained:.2f}/{marks}")
                    
                    breakdown.append({
                        "questionId": q_idx + 1,
                        "obtained": round(obtained, 2),
                        "possible": marks,
                        "textSimilarity": round(text_sim, 3),
                        "diagramSimilarity": round(diagram_sim, 3),
                        "formulasDetected": all_formulas
                    })
                    
                    total_obtained += obtained
                    total_possible += marks
                
                try:
                    if os.path.exists(sheet_output_dir):
                        shutil.rmtree(sheet_output_dir)
                        print(f"   🧹 Cleaned up {sheet_output_dir}")
                except:
                    pass
                
                percentage = round((total_obtained / total_possible * 100) if total_possible > 0 else 0, 1)
                sheet_result = {
                    "totalObtained": round(total_obtained, 2),
                    "totalPossible": total_possible,
                    "percentage": percentage,
                    "breakdown": breakdown
                }
                all_results.append(sheet_result)
                print(f"\n   [{time.time()-start_time:.1f}s] ✅ Sheet {sheet_idx+1}: {sheet_result['totalObtained']}/{sheet_result['totalPossible']} ({sheet_result['percentage']}%)")
                
            except Exception as e:
                print(f"❌ Error processing sheet {sheet_idx+1}: {e}")
                traceback.print_exc()
                continue

        try:
            if os.path.exists("output"):
                shutil.rmtree("output")
                print(f"\n🧹 Cleaned up main output folder")
        except:
            pass

        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ COMPLETED in {total_time:.1f}s")
        print(f"📊 Results: {len(all_results)} sheet(s) processed")
        print("="*60 + "\n")
        
        if all_results:
            return JSONResponse(content=all_results[0])
        else:
            return JSONResponse(
                status_code=500,
                content={"totalObtained": 0, "totalPossible": 0, "percentage": 0, "breakdown": []}
            )
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"totalObtained": 0, "totalPossible": 0, "percentage": 0, "breakdown": [], "error": f"Server error: {str(e)}"}
        )