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

# =========================
# 🔹 PATHS
# =========================
sys.path.insert(0, '/content/Uni-MuMER')

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"
QWEN_MODEL_PATH   = "/content/models/Qwen2.5-3B-Instruct"

# Cosine similarity threshold for embedding coverage scoring
EMBEDDING_THRESHOLD = 0.60

# Hybrid scoring weight: final = ALPHA * embedding_score + (1-ALPHA) * llm_score
ALPHA = 0.6

# Number of clusters for batch feedback generation
N_CLUSTERS = 4

# Number of LLM passes for self-consistency ensemble scoring
ENSEMBLE_PASSES = 3

# =========================
# 🔹 GLOBAL MODEL STATE
# =========================
_qwen_llm        = None
_qwen_sampling   = None
_unimer_llm      = None
_unimer_sampling = None
_embedder        = None

# ─────────────────────────────────────────────
# Model loading / unloading
# ─────────────────────────────────────────────

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
            max_model_len=4096,   # increased from 2048 to avoid mid-JSON truncation
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
    """
    Robustly extract a JSON object ({}) or array ([]) from messy LLM output.
    Strips markdown fences, leading prose, and trailing text.
    """
    text = re.sub(r"```[a-z]*|```", "", text).strip()

    # Direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, expected_type):
            return parsed
    except Exception:
        pass

    # Scan for outermost matching bracket pair
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
# 🔹 UNI-MuMER INFERENCE
# =========================

def clean_unimer_output(text: str) -> str:
    """Protect LaTeX blocks from the letter-merging regex."""
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


def run_unimer_on_images(images: list) -> dict:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    import tempfile

    print("━" * 50)
    print(f"🔍 [OCR] Starting Uni-MuMER inference on {len(images)} page(s)...")
    t0 = time.time()

    llm, sampling = get_unimer()
    processor = AutoProcessor.from_pretrained(UNIMER_MODEL_PATH, trust_remote_code=True)

    all_lines = []
    for page_idx, img in enumerate(images):
        print(f"  📄 [OCR] Processing page {page_idx + 1}/{len(images)}...")
        w, h = img.size
        if max(w, h) > 800:
            scale = 800 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
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

        prompt_text         = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _     = process_vision_info(messages)
        inputs              = {"prompt": prompt_text, "multi_modal_data": {"image": image_inputs}}
        output              = llm.generate([inputs], sampling)[0].outputs[0].text
        output              = clean_unimer_output(output.strip())
        os.unlink(tmp_path)

        lines = [l.strip() for l in output.split('\n') if l.strip()]
        for line in lines:
            all_lines.append(f"Line {len(all_lines) + 1}: {line}")

    raw_text = "\n".join(all_lines)
    elapsed  = time.time() - t0

    print(f"✅ [OCR] Done. {len(all_lines)} lines extracted in {elapsed:.1f}s.")
    print(f"   Raw preview: {raw_text[:200].replace(chr(10), ' ')}{'...' if len(raw_text) > 200 else ''}")
    print("━" * 50)

    return {"raw_text": raw_text, "lines": all_lines}


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

    print(f"✅ [CLEAN] Done in {time.time()-t0:.1f}s. Output length: {len(result)} chars.")
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

    # Prefix forcing: start assistant turn with '[' to guide JSON output
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
    """
    For each teacher key point, find the best cosine-similar student key point.
    Coverage = fraction of teacher points with a match >= threshold.
    Score    = coverage * marks.
    """
    if not student_points or not teacher_points:
        print("⚠️  [EMB] Missing student or teacher points — returning 0.")
        return 0.0

    print(f"📐 [EMB] Computing embedding coverage score...")
    print(f"   Teacher points : {len(teacher_points)}")
    print(f"   Student points : {len(student_points)}")
    print(f"   Threshold      : {threshold}")

    embedder  = get_embedder()
    t_embs    = embedder.encode(teacher_points)             # (T, 384)
    s_embs    = embedder.encode(student_points)             # (S, 384)
    sim_mat   = cosine_similarity(s_embs, t_embs)           # (S, T)
    best      = sim_mat.max(axis=0)                         # (T,) best student match per teacher point

    covered_count = int((best >= threshold).sum())
    coverage      = covered_count / len(teacher_points)
    emb_score     = round(coverage * marks, 2)

    print(f"   Covered        : {covered_count}/{len(teacher_points)} teacher points (≥{threshold})")
    print(f"   Coverage       : {coverage:.2%}")
    print(f"   Embedding score: {emb_score}/{marks}")

    # Print per-teacher-point detail
    for ti, tp in enumerate(teacher_points):
        best_student = student_points[int(sim_mat[:, ti].argmax())]
        print(f"     TP{ti+1} [{best[ti]:.2f}] '{tp[:60]}' ← '{best_student[:60]}'")

    return emb_score


# =========================
# 🔹 LLM ENSEMBLE SCORE (self-consistency)
# =========================

def score_llm_ensemble(student_points: list, key_answer: str,
                        marks: int, n_passes: int = ENSEMBLE_PASSES) -> float:
    """
    Run Qwen scoring n_passes times with temperature=0.3.
    Return the median score (self-consistency).
    """
    print(f"🤖 [LLM] Self-consistency scoring: {n_passes} passes at temperature=0.3...")
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

    final = float(median(scores))
    spread = max(scores) - min(scores)
    print(f"   Scores across passes : {scores}")
    print(f"   Spread (max-min)     : {spread:.2f}  {'⚠️ high variance' if spread > marks * 0.3 else '✅ stable'}")
    print(f"   Median score         : {final}/{marks}")
    print(f"   Done in {time.time()-t0:.1f}s.")
    return final, remarks


# =========================
# 🔹 HYBRID SCORE
# =========================

def hybrid_score(student_points: list, teacher_points: list,
                 key_answer: str, marks: int,
                 alpha: float = ALPHA) -> dict:
    """
    final = alpha * embedding_score + (1 - alpha) * llm_ensemble_score
    Returns dict with all component scores for transparency.
    """
    print("━" * 50)
    print(f"⚖️  [HYBRID] Computing hybrid score (α={alpha})...")

    emb_score            = score_embedding(student_points, teacher_points, marks)
    llm_score, remarks   = score_llm_ensemble(student_points, key_answer, marks)
    final                = round(alpha * emb_score + (1 - alpha) * llm_score, 2)

    print(f"   📐 Embedding score : {emb_score}/{marks}")
    print(f"   🤖 LLM ensemble    : {llm_score}/{marks}")
    print(f"   ✅ Hybrid final    : {final}/{marks}  (α={alpha}×emb + {1-alpha}×llm)")
    print("━" * 50)

    return {
        "score":     final,
        "emb_score": emb_score,
        "llm_score": llm_score,
        "remarks":   remarks,
    }


# =========================
# 🔹 EVALUATE (replaces old qwen_evaluate)
# =========================

def evaluate_with_hybrid(key_points: list, questions_data: list) -> dict:
    """
    For each question, extract teacher key points then run hybrid scoring.
    """
    total_score  = 0.0
    max_total    = 0
    per_question = []

    print("━" * 50)
    print(f"📝 [EVAL] Starting hybrid evaluation for {len(questions_data)} question(s)...")
    print("━" * 50)

    for i, q in enumerate(questions_data):
        key_answer = q.get("keyAnswer", "")
        marks      = q.get("marks", 1)
        max_total += marks

        print(f"\n  📌 Q{i+1} (max={marks})")

        # Extract teacher key points from the key answer
        print(f"  🔑 [EVAL] Extracting teacher key points for Q{i+1}...")
        teacher_points = qwen_extract_key_points(key_answer)

        if not teacher_points:
            print(f"  ⚠️  [EVAL] No teacher key points extracted for Q{i+1} — scoring as 0.")
            per_question.append({
                "score": 0, "marks": marks,
                "remarks": "Could not extract teacher key points.",
                "emb_score": 0, "llm_score": 0,
            })
            continue

        result = hybrid_score(key_points, teacher_points, key_answer, marks)
        total_score += result["score"]

        per_question.append({
            "score":     result["score"],
            "marks":     marks,
            "remarks":   result["remarks"],
            "emb_score": result["emb_score"],
            "llm_score": result["llm_score"],
        })

    print("\n" + "━" * 50)
    print(f"🏁 [EVAL] Final Score: {round(total_score, 2)}/{max_total}")
    print("━" * 50)

    return {
        "score":        round(total_score, 2),
        "total":        max_total,
        "per_question": per_question,
    }


# =========================
# 🔹 CLUSTER FEEDBACK HELPER
# =========================

def generate_cluster_feedback(all_students_data: list, key_answer: str) -> list:
    """
    Groups students by answer embedding similarity, generates feedback once
    per cluster centroid, assigns to all members.

    all_students_data: list of dicts:
      {key_points: list, cleaned_text: str}

    Returns same list with added 'feedback' key per student.
    """
    n = len(all_students_data)
    if n == 0:
        return all_students_data

    k = min(N_CLUSTERS, n)
    print("━" * 50)
    print(f"🔵 [CLUSTER] Clustering {n} student answer(s) into k={k} groups...")

    embedder = get_embedder()
    texts    = [s.get("cleaned_text", " ".join(s.get("key_points", []))) for s in all_students_data]
    embs     = embedder.encode(texts)   # (N, 384)

    if n == 1:
        labels   = np.array([0])
        centroids = embs
    else:
        kmeans    = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs)
        labels    = kmeans.labels_
        centroids = kmeans.cluster_centers_

    print(f"   Cluster distribution: { {int(c): int((labels==c).sum()) for c in range(k)} }")

    cluster_feedbacks = {}
    qwen, sampling    = get_qwen()

    for cid in range(k):
        members = [i for i, l in enumerate(labels) if l == cid]
        if not members:
            continue

        # Pick representative closest to centroid
        dists   = [np.linalg.norm(embs[i] - centroids[cid]) for i in members]
        rep_idx = members[int(np.argmin(dists))]
        rep     = all_students_data[rep_idx]

        print(f"\n  🔵 Cluster {cid} — {len(members)} member(s), representative: index {rep_idx}")

        student_str = (
            "\n".join(f"- {p}" for p in rep.get("key_points", []))
            or "(none)"
        )

        # ── Pass A: Strengths / Improvements / Suggestions ────────────────
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
            print(f"  ⚠️  [CLUSTER] Feedback JSON parse failed for cluster {cid}.")
            feedback = {"strengths": [], "improvements": [], "suggestions": []}
        else:
            for key in ["strengths", "improvements", "suggestions"]:
                if not isinstance(feedback.get(key), list):
                    feedback[key] = []

        print(f"     Strengths    : {len(feedback['strengths'])} items")
        print(f"     Improvements : {len(feedback['improvements'])} items")
        print(f"     Suggestions  : {len(feedback['suggestions'])} items")

        # ── Pass B: Bloom's Taxonomy ───────────────────────────────────────
        bloom_levels = ["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]

        prompt_b = (
            "<|im_start|>system\n"
            "You are an expert in Bloom's Taxonomy. "
            "Rate each of the 6 Bloom's levels (Remember, Understand, Apply, Analyse, Evaluate, Create). "
            "For each level rate: required (0-100) and demonstrated (0-100). "
            "Return ONLY a raw JSON array of exactly 6 objects. No prose, no markdown.\n"
            'Example: [{"level":"Remember","required":80,"demonstrated":70}]'
            "<|im_end|>\n"
            f"<|im_start|>user\nKey Answer:\n{key_answer}\n\n"
            f"Student Key Points:\n{student_str}<|im_end|>\n"
            "<|im_start|>assistant\n["
        )

        raw_b  = "[" + qwen.generate([prompt_b], sampling)[0].outputs[0].text
        raw_b  = raw_b.replace("<|im_end|>", "").strip()
        blooms = extract_json(raw_b, list)

        if not blooms:
            print(f"  ⚠️  [CLUSTER] Bloom's JSON parse failed for cluster {cid} — using zeros.")
            blooms = [{"level": l, "required": 0, "demonstrated": 0} for l in bloom_levels]
        else:
            # Normalise: ensure all 6 levels present, values clamped 0-100
            blooms_map = {b.get("level"): b for b in blooms if isinstance(b, dict)}
            blooms = []
            for level in bloom_levels:
                entry = blooms_map.get(level, {})
                blooms.append({
                    "level":        level,
                    "required":     max(0, min(100, int(entry.get("required", 0)))),
                    "demonstrated": max(0, min(100, int(entry.get("demonstrated", 0)))),
                })
            print(f"     Bloom's: {[(b['level'], b['demonstrated']) for b in blooms]}")

        cluster_feedbacks[cid] = {
            "strengths":    feedback.get("strengths", []),
            "improvements": feedback.get("improvements", []),
            "suggestions":  feedback.get("suggestions", []),
            "blooms":       blooms,
        }

    # Assign cluster feedback to every student
    for i, student in enumerate(all_students_data):
        student["analysis"]   = cluster_feedbacks.get(int(labels[i]), {
            "strengths": [], "improvements": [], "suggestions": [], "blooms": []
        })
        student["cluster_id"] = int(labels[i])

    print(f"\n✅ [CLUSTER] Feedback generated for {k} clusters, assigned to {n} student(s).")
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
    answer_sheets: List[UploadFile] = File(...)
):
    request_start = time.time()

    try:
        print("\n" + "═" * 60)
        print("🚀 [REQUEST] /similarity — new request received")
        print("═" * 60)

        questions_data = json.loads(questions)
        print(f"   Questions     : {len(questions_data)}")
        print(f"   Answer sheets : {len(answer_sheets)}")

        if not answer_sheets:
            return {"error": "No answer sheets uploaded"}

        results = []

        for sheet_idx, sheet in enumerate(answer_sheets):
            sheet_start = time.time()
            print(f"\n{'─'*50}")
            print(f"📄 [SHEET {sheet_idx+1}/{len(answer_sheets)}] Processing: {sheet.filename}")
            print(f"{'─'*50}")

            sheet_dir = f"output/sheet_{sheet_idx}"
            os.makedirs(sheet_dir, exist_ok=True)

            content = await sheet.read()
            images  = convert_from_bytes(content)
            print(f"   Pages in PDF  : {len(images)}")

            # ── STAGE 1: Uni-MuMER OCR ────────────────────────────────────
            unimer_result = run_unimer_on_images(images)
            raw_text      = unimer_result.get("raw_text", "")

            # ── STAGE 2: Clean text ───────────────────────────────────────
            cleaned_text         = qwen_clean_text(raw_text)
            student_answer_lines = wrap_into_lines(cleaned_text, words_per_line=8)

            # ── STAGE 3: Extract student key points ───────────────────────
            key_points = qwen_extract_key_points(cleaned_text)

            # ── STAGE 4: Hybrid scoring ───────────────────────────────────
            eval_result = evaluate_with_hybrid(key_points, questions_data)

            final_result = {
                "score":          eval_result["score"],
                "total":          eval_result["total"],
                "student_answer": student_answer_lines,
                "key_points":     key_points,
                "per_question":   eval_result["per_question"],
            }

            elapsed = time.time() - sheet_start
            print(f"\n{'═'*60}")
            print(f"📦 [RESULT] Sheet {sheet_idx+1} complete in {elapsed:.1f}s")
            print(f"   Score      : {final_result['score']}/{final_result['total']}")
            print(f"   Key points : {len(key_points)}")
            print(json.dumps(final_result, indent=2))
            print("═" * 60)

            results.append(final_result)
            shutil.rmtree(sheet_dir, ignore_errors=True)

        total_elapsed = time.time() - request_start
        print(f"\n🏁 [REQUEST] /similarity complete in {total_elapsed:.1f}s")

        return results[0] if results else {"score": 0, "total": 0}

    except Exception as e:
        traceback.print_exc()
        print(f"❌ [ERROR] /similarity failed: {e}")
        return {"error": str(e)}


# =========================
# 🔹 ANALYSE ENDPOINT
# =========================

@app.post("/analyse")
async def analyse(request: Request):
    request_start = time.time()

    try:
        print("\n" + "═" * 60)
        print("🚀 [REQUEST] /analyse — new request received")
        print("═" * 60)

        body       = await request.json()
        key_answer = body.get("keyAnswer", "")
        key_points = body.get("keyPoints", [])
        score      = body.get("score", 0)
        total      = body.get("total", 1)
        remarks    = body.get("remarks", "")

        print(f"   Key answer length : {len(key_answer)} chars")
        print(f"   Key points        : {len(key_points)}")
        print(f"   Score             : {score}/{total}")

        # Wrap into single-student list and run cluster feedback
        # (with 1 student, clustering trivially assigns to 1 cluster)
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

        elapsed = time.time() - request_start
        print(f"\n✅ [REQUEST] /analyse complete in {elapsed:.1f}s")
        print(json.dumps(result, indent=2))
        print("═" * 60)

        return result

    except Exception as e:
        traceback.print_exc()
        print(f"❌ [ERROR] /analyse failed: {e}")
        return {"error": str(e)}