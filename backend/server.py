from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes

import numpy as np
import os, sys, json, shutil, traceback, re, gc, time
from typing import List
from PIL import Image
from statistics import median

import torch
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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

@app.middleware("http")
async def add_ngrok_skip_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# =========================
# 🔹 PATHS
# =========================
sys.path.insert(0, '/content/Uni-MuMER')

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/models/Qwen2.5-3B-Instruct"

EMBEDDING_THRESHOLD = 0.60
ALPHA               = 0.6
N_CLUSTERS          = 4
ENSEMBLE_PASSES     = 3

# =========================
# 🔹 GLOBAL MODEL STATE
# =========================
_qwen_llm        = None
_qwen_sampling   = None
_unimer_llm      = None
_unimer_sampling = None
_embedder        = None


def unload_unimer():
    global _unimer_llm, _unimer_sampling
    if _unimer_llm is not None:
        print("━" * 50)
        print("🗑️  [MODEL] Unloading Uni-MuMER from GPU...")
        del _unimer_llm
        _unimer_llm      = None
        _unimer_sampling = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ [MODEL] Uni-MuMER unloaded. GPU memory freed.")
        print("━" * 50)


def unload_qwen():
    global _qwen_llm, _qwen_sampling
    if _qwen_llm is not None:
        print("━" * 50)
        print("🗑️  [MODEL] Unloading Qwen from GPU...")
        del _qwen_llm
        _qwen_llm      = None
        _qwen_sampling = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ [MODEL] Qwen unloaded. GPU memory freed.")
        print("━" * 50)


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("━" * 50)
        print("🔄 [MODEL] Loading sentence embedder (all-MiniLM-L6-v2)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ [MODEL] Sentence embedder loaded (CPU).")
        print("━" * 50)
    return _embedder


def get_unimer():
    global _unimer_llm, _unimer_sampling
    unload_qwen()
    if _unimer_llm is None:
        print("━" * 50)
        print("🔄 [MODEL] Loading Uni-MuMER-3B...")
        t0 = time.time()
        _unimer_llm = LLM(
            model=UNIMER_MODEL_PATH,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
        )
        _unimer_sampling = SamplingParams(temperature=0, max_tokens=4096)
        print(f"✅ [MODEL] Uni-MuMER loaded in {time.time()-t0:.1f}s.")
        print("━" * 50)
    else:
        print("⚡ [MODEL] Uni-MuMER already in memory — reusing.")
    return _unimer_llm, _unimer_sampling


def get_qwen():
    global _qwen_llm, _qwen_sampling
    unload_unimer()
    if _qwen_llm is None:
        print("━" * 50)
        print("🔄 [MODEL] Loading Qwen2.5-3B-Instruct...")
        t0 = time.time()
        _qwen_llm = LLM(
            model=QWEN_MODEL_PATH,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
        )
        _qwen_sampling = SamplingParams(temperature=0, max_tokens=1024)
        print(f"✅ [MODEL] Qwen loaded in {time.time()-t0:.1f}s.")
        print("━" * 50)
    else:
        print("⚡ [MODEL] Qwen already in memory — reusing.")
    return _qwen_llm, _qwen_sampling


# =========================
# 🔹 JSON EXTRACTION HELPER
# =========================

def extract_json(text: str, expected_type: type):
    text = re.sub(r"```[a-z]*|```", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, expected_type):
            return parsed
    except Exception:
        pass

    start_char = '{' if expected_type == dict else '['
    end_char   = '}' if expected_type == dict else ']'
    start = text.find(start_char)
    end   = text.rfind(end_char)
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, expected_type):
                return parsed
        except Exception:
            pass

    return None


# =========================
# 🔹 UNI-MuMER — OCR ALL SHEETS IN ONE PASS
# =========================

def clean_unimer_output(text: str) -> str:
    latex_blocks = {}

    def protect(m):
        key = f"__LATEX{len(latex_blocks)}__"
        latex_blocks[key] = m.group(0)
        return key

    text = re.sub(r'\$[^$]+\$', protect, text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', protect, text)
    text = re.sub(
        r'\b([A-Za-z])(( [A-Za-z])+)\b',
        lambda m: (m.group(1) + m.group(2)).replace(" ", ""),
        text
    )
    for key, val in latex_blocks.items():
        text = text.replace(key, val)
    return text


def ocr_all_sheets(sheets_images: list) -> list:
    """
    Load Uni-MuMER ONCE, OCR every sheet, then unload.
    sheets_images : list of lists — [ [page, page, ...], [page, ...], ... ]
    Returns       : list of raw_text strings, one per sheet.
    """
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    import tempfile

    print("━" * 50)
    print(f"🔍 [OCR] Uni-MuMER loading once for {len(sheets_images)} sheet(s)...")
    t0 = time.time()

    llm, sampling = get_unimer()
    processor     = AutoProcessor.from_pretrained(UNIMER_MODEL_PATH, trust_remote_code=True)

    all_raw_texts = []

    for sheet_idx, images in enumerate(sheets_images):
        print(f"\n  📄 [OCR] Sheet {sheet_idx + 1}/{len(sheets_images)} — {len(images)} page(s)")
        sheet_lines = []

        for page_idx, img in enumerate(images):
            print(f"       Page {page_idx + 1}/{len(images)}...")
            w, h = img.size
            if max(w, h) > 800:
                scale = 800 / max(w, h)
                img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                print(f"       Resized to {img.size[0]}x{img.size[1]}")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{tmp_path}"},
                    {"type": "text",  "text": "Read all the text and math formulas in this handwritten image. Output everything you see."}
                ]
            }]

            prompt_text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs          = {"prompt": prompt_text, "multi_modal_data": {"image": image_inputs}}
            output          = llm.generate([inputs], sampling)[0].outputs[0].text
            output          = clean_unimer_output(output.strip())
            os.unlink(tmp_path)

            lines = [l.strip() for l in output.split('\n') if l.strip()]
            for line in lines:
                sheet_lines.append(f"Line {len(sheet_lines) + 1}: {line}")

        raw_text = "\n".join(sheet_lines)
        all_raw_texts.append(raw_text)
        print(f"  ✅ Sheet {sheet_idx + 1}: {len(sheet_lines)} lines extracted")
        print(f"     Preview: {raw_text[:150].replace(chr(10), ' ')}{'...' if len(raw_text) > 150 else ''}")

    # 🔥 Unload Uni-MuMER after ALL sheets are done
    unload_unimer()

    elapsed = time.time() - t0
    print(f"\n✅ [OCR] All {len(sheets_images)} sheet(s) done in {elapsed:.1f}s. Uni-MuMER unloaded.")
    print("━" * 50)

    return all_raw_texts


# =========================
# 🔹 HELPER
# =========================

def wrap_into_lines(text: str, words_per_line: int = 8) -> list:
    words = text.split()
    return [" ".join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]


# =========================
# 🔹 QWEN PASS 1 — CLEAN TEXT
# =========================

def qwen_clean_text(raw_text: str) -> str:
    if not raw_text.strip():
        print("⚠️  [CLEAN] Empty raw text — skipping.")
        return ""

    print("🧹 [CLEAN] Qwen Pass 1: Fixing OCR noise and spacing...")
    t0 = time.time()

    qwen, sampling = get_qwen()

    prompt = (
        "<|im_start|>system\n"
        "You are an OCR post-processor for handwritten exam answer sheets. "
        "Fix all spacing, word-merging, and obvious OCR errors. "
        "Preserve the original meaning exactly — do NOT add, remove, or rephrase content. "
        "Output ONLY the corrected text as clean prose. No labels, no explanation."
        "<|im_end|>\n"
        f"<|im_start|>user\nFix the OCR errors in this text:\n\n{raw_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    result = qwen.generate([prompt], sampling)[0].outputs[0].text
    result = result.replace("<|im_end|>", "").strip()

    print(f"✅ [CLEAN] Done in {time.time()-t0:.1f}s. Output: {len(result)} chars.")
    print(f"   Preview: {result[:150].replace(chr(10), ' ')}{'...' if len(result) > 150 else ''}")
    return result


# =========================
# 🔹 QWEN PASS 2 — EXTRACT KEY POINTS
# =========================

def qwen_extract_key_points(cleaned_text: str) -> list:
    if not cleaned_text.strip():
        print("⚠️  [KP] Empty cleaned text — returning empty key points.")
        return []

    print("🔑 [KP] Qwen Pass 2: Extracting key points...")
    t0 = time.time()

    qwen, sampling = get_qwen()

    prompt = (
        "<|im_start|>system\n"
        "You are an exam answer analyser. "
        "Extract the distinct key points/facts/steps from the student's answer. "
        "Express each key point as a SHORT phrase of 5-8 words maximum. "
        "Return ONLY a raw JSON array of strings — no prose, no markdown fences.\n"
        'Example: ["Gradient descent minimizes the loss function", "Weights updated via backpropagation"]'
        "<|im_end|>\n"
        f"<|im_start|>user\nExtract key points:\n\n{cleaned_text}<|im_end|>\n"
        "<|im_start|>assistant\n["
    )

    raw    = "[" + qwen.generate([prompt], sampling)[0].outputs[0].text
    raw    = raw.replace("<|im_end|>", "").strip()
    points = extract_json(raw, list)

    if not isinstance(points, list):
        print("⚠️  [KP] JSON parse failed — falling back to sentence split.")
        sentences = re.split(r'[.;]\s*', cleaned_text)
        points    = [s.strip() for s in sentences if len(s.strip()) > 4]
    else:
        points = [str(p).strip() for p in points if str(p).strip()]

    print(f"✅ [KP] Done in {time.time()-t0:.1f}s. Extracted {len(points)} key points:")
    for i, p in enumerate(points):
        print(f"     {i+1}. {p}")
    return points


# =========================
# 🔹 EMBEDDING SCORE
# =========================

def score_embedding(student_points: list, teacher_points: list,
                    marks: int, threshold: float = EMBEDDING_THRESHOLD) -> float:
    if not student_points or not teacher_points:
        print("⚠️  [EMB] Missing points — returning 0.")
        return 0.0

    print(f"📐 [EMB] Computing embedding coverage...")
    embedder = get_embedder()
    t_embs   = embedder.encode(teacher_points)
    s_embs   = embedder.encode(student_points)
    sim_mat  = cosine_similarity(s_embs, t_embs)
    best     = sim_mat.max(axis=0)

    covered_count = int((best >= threshold).sum())
    coverage      = covered_count / len(teacher_points)
    emb_score     = round(coverage * marks, 2)

    print(f"   Covered: {covered_count}/{len(teacher_points)} (≥{threshold}) | "
          f"Coverage: {coverage:.2%} | Score: {emb_score}/{marks}")

    for ti, tp in enumerate(teacher_points):
        best_student = student_points[int(sim_mat[:, ti].argmax())]
        print(f"     TP{ti+1} [{best[ti]:.2f}] '{tp[:60]}' ← '{best_student[:60]}'")

    return emb_score


# =========================
# 🔹 LLM ENSEMBLE SCORE
# =========================

def score_llm_ensemble(student_points: list, key_answer: str,
                       marks: int, n_passes: int = ENSEMBLE_PASSES):
    print(f"🤖 [LLM] Self-consistency scoring: {n_passes} passes...")
    t0 = time.time()

    qwen, _ = get_qwen()
    sampling = SamplingParams(temperature=0.3, max_tokens=256)

    student_str = "\n".join(f"- {p}" for p in student_points) if student_points else "(no answer detected)"

    prompt = (
        "<|im_start|>system\n"
        "You are a strict but fair exam evaluator. "
        "Given a key answer and student key points, award a score. "
        "Respond ONLY with valid raw JSON — no markdown, no prose: "
        "{\"score\": <number>, \"remarks\": \"<one sentence>\"}"
        "<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Key Answer: {key_answer}\n\n"
        f"Student Key Points:\n{student_str}\n\n"
        f"Maximum marks: {marks}. Award a score between 0 and {marks}."
        "<|im_end|>\n"
        "<|im_start|>assistant\n{"
    )

    scores  = []
    remarks = ""
    for i in range(n_passes):
        raw = "{" + qwen.generate([prompt], sampling)[0].outputs[0].text
        raw = raw.replace("<|im_end|>", "").strip()
        parsed = extract_json(raw, dict)
        if parsed:
            s = max(0.0, min(float(marks), float(parsed.get("score", 0))))
            scores.append(s)
            if not remarks:
                remarks = parsed.get("remarks", "")
            print(f"     Pass {i+1}: score={s}/{marks}")
        else:
            print(f"     Pass {i+1}: ⚠️  parse failed — raw: {raw[:80]}")

    if not scores:
        print("⚠️  [LLM] All passes failed — returning 0.")
        return 0.0, ""

    final  = float(median(scores))
    spread = max(scores) - min(scores)
    print(f"   Scores: {scores} | Spread: {spread:.2f} | Median: {final}/{marks} | "
          f"Done in {time.time()-t0:.1f}s.")
    return final, remarks


# =========================
# 🔹 HYBRID SCORE
# =========================

def hybrid_score(student_points: list, teacher_points: list,
                 key_answer: str, marks: int,
                 alpha: float = ALPHA) -> dict:
    print("━" * 50)
    print(f"⚖️  [HYBRID] α={alpha}...")

    emb_score          = score_embedding(student_points, teacher_points, marks)
    llm_score, remarks = score_llm_ensemble(student_points, key_answer, marks)
    final              = round(alpha * emb_score + (1 - alpha) * llm_score, 2)

    print(f"   📐 Embedding: {emb_score}/{marks}  🤖 LLM: {llm_score}/{marks}  ✅ Hybrid: {final}/{marks}")
    print("━" * 50)

    return {"score": final, "emb_score": emb_score, "llm_score": llm_score, "remarks": remarks}


# =========================
# 🔹 EVALUATE ONE SHEET AGAINST ONE QUESTION
# =========================

def evaluate_one(key_points: list, question_data: dict) -> dict:
    """Score a single student's key points against a single question."""
    key_answer = question_data.get("keyAnswer", "")
    marks      = question_data.get("marks", 1)

    print(f"\n  📌 Scoring against question (max={marks})")
    teacher_points = qwen_extract_key_points(key_answer)

    if not teacher_points:
        print(f"  ⚠️  No teacher key points — scoring as 0.")
        return {"score": 0, "marks": marks, "remarks": "Could not extract teacher key points.",
                "emb_score": 0, "llm_score": 0}

    result = hybrid_score(key_points, teacher_points, key_answer, marks)
    return {
        "score":     result["score"],
        "marks":     marks,
        "remarks":   result["remarks"],
        "emb_score": result["emb_score"],
        "llm_score": result["llm_score"],
    }


# =========================
# 🔹 CLUSTER FEEDBACK
# =========================

def generate_cluster_feedback(all_students_data: list, key_answer: str) -> list:
    n = len(all_students_data)
    if n == 0:
        return all_students_data

    k = min(N_CLUSTERS, n)
    print("━" * 50)
    print(f"🔵 [CLUSTER] {n} student(s) → k={k} cluster(s)...")

    embedder = get_embedder()
    texts    = [s.get("cleaned_text", " ".join(s.get("key_points", []))) for s in all_students_data]
    embs     = embedder.encode(texts)

    if n == 1:
        labels    = np.array([0])
        centroids = embs
    else:
        kmeans    = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs)
        labels    = kmeans.labels_
        centroids = kmeans.cluster_centers_

    print(f"   Distribution: { {int(c): int((labels==c).sum()) for c in range(k)} }")

    cluster_feedbacks = {}
    qwen, sampling    = get_qwen()
    bloom_levels      = ["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]

    for cid in range(k):
        members = [i for i, l in enumerate(labels) if l == cid]
        if not members:
            continue

        dists   = [np.linalg.norm(embs[i] - centroids[cid]) for i in members]
        rep_idx = members[int(np.argmin(dists))]
        rep     = all_students_data[rep_idx]

        print(f"\n  🔵 Cluster {cid} — {len(members)} member(s), rep: index {rep_idx}")
        student_str = "\n".join(f"- {p}" for p in rep.get("key_points", [])) or "(none)"

        prompt_a = (
            "<|im_start|>system\n"
            "You are an expert exam coach. Given a key answer and student key points, "
            "produce a JSON object with exactly these keys: "
            "{\"strengths\": [\"...\"], \"improvements\": [\"...\"], \"suggestions\": [\"...\"]}\n"
            "strengths: 2-3 things the student did well.\n"
            "improvements: 2-3 specific gaps or missing concepts.\n"
            "suggestions: 2-3 concrete actionable tips.\n"
            "Output ONLY the raw JSON object. No prose, no markdown."
            "<|im_end|>\n"
            f"<|im_start|>user\nKey Answer:\n{key_answer}\n\n"
            f"Student Key Points:\n{student_str}<|im_end|>\n"
            "<|im_start|>assistant\n{"
        )

        raw_a    = "{" + qwen.generate([prompt_a], sampling)[0].outputs[0].text
        raw_a    = raw_a.replace("<|im_end|>", "").strip()
        feedback = extract_json(raw_a, dict)

        if not feedback:
            print(f"  ⚠️  Feedback JSON parse failed for cluster {cid}.")
            feedback = {"strengths": [], "improvements": [], "suggestions": []}
        else:
            for key in ["strengths", "improvements", "suggestions"]:
                if not isinstance(feedback.get(key), list):
                    feedback[key] = []

        print(f"     S:{len(feedback['strengths'])} I:{len(feedback['improvements'])} "
              f"Sg:{len(feedback['suggestions'])}")

        prompt_b = (
            "<|im_start|>system\n"
            "You are an expert in Bloom's Taxonomy for exam evaluation.\n"
            "For each of the 6 Bloom's levels, do the following:\n"
            "1. 'required': Does the KEY ANSWER demand this level? Score 0-100.\n"
            "   - 0   = not needed at all\n"
            "   - 50  = partially needed\n"
            "   - 100 = central to the answer\n"
            "2. 'demonstrated': Did the STUDENT KEY POINTS show this level? Score 0 to 'required' only.\n"
            "   - demonstrated CANNOT exceed required.\n"
            "   - If required=0, demonstrated must also be 0.\n"
            "Be strict. Most answers only strongly require 2-3 levels.\n"
            "Return ONLY a raw JSON array of exactly 6 objects. No prose, no markdown.\n"
            'Format: [{"level":"Remember","required":80,"demonstrated":60}]'
            "<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Key Answer:\n{key_answer}\n\n"
            f"Student Key Points:\n{student_str}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n["
        )

        raw_b  = "[" + qwen.generate([prompt_b], sampling)[0].outputs[0].text
        raw_b  = raw_b.replace("<|im_end|>", "").strip()
        blooms = extract_json(raw_b, list)

        if not blooms:
            print(f"  ⚠️  Bloom's JSON parse failed for cluster {cid} — using zeros.")
            blooms = [{"level": l, "required": 0, "demonstrated": 0} for l in bloom_levels]
        else:
            blooms_map = {b.get("level"): b for b in blooms if isinstance(b, dict)}
            blooms = []
            for level in bloom_levels:
                entry = blooms_map.get(level, {})
                req   = max(0, min(100, int(entry.get("required", 0))))
                dem   = max(0, min(req, int(entry.get("demonstrated", 0))))
                blooms.append({"level": level, "required": req, "demonstrated": dem})
            print(f"     Bloom's: {[(b['level'], b['required'], b['demonstrated']) for b in blooms]}")

        cluster_feedbacks[cid] = {
            "strengths":    feedback.get("strengths", []),
            "improvements": feedback.get("improvements", []),
            "suggestions":  feedback.get("suggestions", []),
            "blooms":       blooms,
        }

    for i, student in enumerate(all_students_data):
        student["analysis"]   = cluster_feedbacks.get(int(labels[i]), {
            "strengths": [], "improvements": [], "suggestions": [], "blooms": []
        })
        student["cluster_id"] = int(labels[i])

    print(f"\n✅ [CLUSTER] Done. {k} cluster(s), {n} student(s).")
    print("━" * 50)
    return all_students_data


# =========================
# 🔹 HEALTH
# =========================

@app.get("/health")
async def health():
    return {"status": "running"}


# =========================
# 🔹 SIMILARITY ENDPOINT
# =========================

@app.post("/similarity")
async def similarity(
    request: Request,
    questions: str  = Form(...),
    answer_sheets: List[UploadFile] = File(...),
    sheet_meta: str = Form(...),   # JSON: [{question_index, student_name}, ...]
):
    request_start = time.time()

    try:
        print("\n" + "═" * 60)
        print("🚀 [REQUEST] /similarity — all-at-once mode")
        print("═" * 60)

        questions_data = json.loads(questions)
        meta_list      = json.loads(sheet_meta)  # [{question_index, student_name}, ...]

        print(f"   Questions: {len(questions_data)}  |  Sheets: {len(answer_sheets)}")
        for i, m in enumerate(meta_list):
            print(f"   Sheet {i+1}: Q{m['question_index']+1} | {m['student_name']}")

        if not answer_sheets:
            return {"error": "No answer sheets uploaded"}

        if len(answer_sheets) != len(meta_list):
            return {"error": f"Mismatch: {len(answer_sheets)} sheets but {len(meta_list)} meta entries"}

        # ══════════════════════════════════════════════════════════
        # STAGE 1 — Read ALL PDFs into memory
        # ══════════════════════════════════════════════════════════
        print("\n📂 [STAGE 1] Reading all PDFs into memory...")
        sheets_images   = []
        sheet_filenames = []
        for sheet in answer_sheets:
            content = await sheet.read()
            images  = convert_from_bytes(content)
            sheets_images.append(images)
            sheet_filenames.append(sheet.filename)
            print(f"   {sheet.filename}: {len(images)} page(s)")

        # ══════════════════════════════════════════════════════════
        # STAGE 2 — Uni-MuMER: load ONCE → OCR ALL sheets → unload
        # ══════════════════════════════════════════════════════════
        print(f"\n🔍 [STAGE 2] Uni-MuMER: OCR all {len(sheets_images)} sheet(s) in one pass...")
        all_raw_texts = ocr_all_sheets(sheets_images)
        # ✅ Uni-MuMER is already unloaded inside ocr_all_sheets()

        # ══════════════════════════════════════════════════════════
        # STAGE 3 — Qwen: load ONCE → process ALL sheets → unload
        # We score each sheet only against its assigned question.
        # Then we group by student name and sum up per_question scores.
        # ══════════════════════════════════════════════════════════
        print(f"\n🤖 [STAGE 3] Qwen: evaluating all {len(all_raw_texts)} sheet(s) in one pass...")
        get_qwen()  # load once up-front

        # Pre-extract teacher key points for each question ONCE
        # so we don't repeat for every student
        print("\n📚 [STAGE 3a] Pre-extracting teacher key points for all questions...")
        teacher_kp_cache = {}
        for qi, q in enumerate(questions_data):
            print(f"   Q{qi+1}...")
            teacher_kp_cache[qi] = qwen_extract_key_points(q.get("keyAnswer", ""))

        # per_sheet_results[i] = {student_name, question_index, key_points,
        #                         student_answer, pq_result}
        per_sheet_results = []

        for sheet_idx, raw_text in enumerate(all_raw_texts):
            sheet_start  = time.time()
            meta         = meta_list[sheet_idx]
            student_name = meta["student_name"]
            qi           = int(meta["question_index"])
            qi           = max(0, min(qi, len(questions_data) - 1))  # clamp
            fname        = sheet_filenames[sheet_idx]

            print(f"\n{'─'*50}")
            print(f"📄 [{sheet_idx+1}/{len(all_raw_texts)}] {fname} → Q{qi+1} | {student_name}")
            print(f"{'─'*50}")

            cleaned_text         = qwen_clean_text(raw_text)
            student_answer_lines = wrap_into_lines(cleaned_text, words_per_line=8)
            key_points           = qwen_extract_key_points(cleaned_text)

            # Score only against the assigned question
            q_data         = questions_data[qi]
            teacher_points = teacher_kp_cache[qi]
            marks          = q_data.get("marks", 1)

            if not teacher_points:
                pq_result = {"score": 0, "marks": marks,
                             "remarks": "Could not extract teacher key points.",
                             "emb_score": 0, "llm_score": 0}
            else:
                h = hybrid_score(key_points, teacher_points, q_data.get("keyAnswer", ""), marks)
                pq_result = {
                    "score":     h["score"],
                    "marks":     marks,
                    "remarks":   h["remarks"],
                    "emb_score": h["emb_score"],
                    "llm_score": h["llm_score"],
                }

            elapsed = time.time() - sheet_start
            print(f"✅ Sheet done in {elapsed:.1f}s — {pq_result['score']}/{marks}")

            per_sheet_results.append({
                "student_name":   student_name,
                "question_index": qi,
                "key_points":     key_points,
                "student_answer": student_answer_lines,
                "pq_result":      pq_result,
            })

        # ✅ Unload Qwen ONCE after ALL sheets evaluated
        unload_qwen()

        # ══════════════════════════════════════════════════════════
        # STAGE 4 — Merge by student name
        # Each student gets one entry with per_question[] sized to
        # all questions. Slots they didn't upload stay as 0.
        # ══════════════════════════════════════════════════════════
        print("\n🔀 [STAGE 4] Merging results by student name...")

        total_marks = sum(q.get("marks", 0) for q in questions_data)

        # Use ordered dict to preserve first-seen order
        student_map: dict = {}

        for res in per_sheet_results:
            key  = res["student_name"].strip().lower()
            name = res["student_name"]
            qi   = res["question_index"]

            if key not in student_map:
                # Initialise with zero per_question slots for every question
                student_map[key] = {
                    "student_name":   name,
                    "score":          0.0,
                    "total":          total_marks,
                    "student_answer": [],
                    "key_points":     [],
                    "per_question": [
                        {"score": 0, "marks": q.get("marks", 1),
                         "remarks": "", "emb_score": 0, "llm_score": 0}
                        for q in questions_data
                    ],
                }

            entry = student_map[key]
            # Slot in this question's result
            entry["per_question"][qi] = res["pq_result"]
            # Accumulate score
            entry["score"] = round(
                sum(pq["score"] for pq in entry["per_question"]), 2
            )
            # Append answer lines and key points
            entry["student_answer"] += res["student_answer"]
            entry["key_points"]     += res["key_points"]

        students = list(student_map.values())
        scores   = [s["score"] for s in students]

        response_data = students  # return as flat array — frontend normaliseBatchResult handles it

        total_elapsed = time.time() - request_start
        print(f"\n🏁 [REQUEST] /similarity done in {total_elapsed:.1f}s — "
              f"{len(students)} unique student(s)")
        print("═" * 60)

        return response_data

    except Exception as e:
        traceback.print_exc()
        print(f"❌ [ERROR] /similarity: {e}")
        return {"error": str(e)}


# =========================
# 🔹 ANALYSE ENDPOINT
# =========================

@app.post("/analyse")
async def analyse(request: Request):
    request_start = time.time()

    try:
        print("\n" + "═" * 60)
        print("🚀 [REQUEST] /analyse")
        print("═" * 60)

        body       = await request.json()
        key_answer = body.get("keyAnswer", "")
        key_points = body.get("keyPoints", [])
        score      = body.get("score", 0)
        total      = body.get("total", 1)

        print(f"   Key answer: {len(key_answer)} chars  |  Key points: {len(key_points)}  |  Score: {score}/{total}")

        student_data = [{
            "key_points":   key_points,
            "cleaned_text": " ".join(key_points),
        }]

        student_data = generate_cluster_feedback(student_data, key_answer)
        analysis     = student_data[0].get("analysis", {})

        result = {
            "strengths":    analysis.get("strengths", []),
            "improvements": analysis.get("improvements", []),
            "suggestions":  analysis.get("suggestions", []),
            "blooms":       analysis.get("blooms", []),
        }

        # ✅ Unload Qwen after analysis
        unload_qwen()

        elapsed = time.time() - request_start
        print(f"\n✅ [REQUEST] /analyse done in {elapsed:.1f}s")
        print(json.dumps(result, indent=2))
        print("═" * 60)

        return result

    except Exception as e:
        traceback.print_exc()
        print(f"❌ [ERROR] /analyse: {e}")
        return {"error": str(e)}