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

# Directory to persist Uni-MuMER OCR outputs between model loads
OCR_CACHE_DIR = "/content/ocr_cache"
os.makedirs(OCR_CACHE_DIR, exist_ok=True)

# =========================
# 🔹 HYPERPARAMETERS
# =========================
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
    """Robustly extract JSON object or array from messy LLM output."""
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
# 🔹 UNI-MuMER INFERENCE
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


def run_unimer_on_images(images: list, student_label: str) -> str:
    """
    Run Uni-MuMER on a list of PIL images.
    Returns raw_text string.
    student_label used only for logging.
    """
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    import tempfile

    print(f"  🔍 [OCR] Running Uni-MuMER on '{student_label}' ({len(images)} page(s))...")
    t0 = time.time()

    llm, sampling = get_unimer()
    processor = AutoProcessor.from_pretrained(UNIMER_MODEL_PATH, trust_remote_code=True)

    all_lines = []
    for page_idx, img in enumerate(images):
        w, h = img.size
        if max(w, h) > 800:
            scale = 800 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

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
            all_lines.append(f"Line {len(all_lines)+1}: {line}")

    raw_text = "\n".join(all_lines)
    print(f"  ✅ [OCR] '{student_label}' — {len(all_lines)} lines in {time.time()-t0:.1f}s.")
    print(f"     Preview: {raw_text[:120].replace(chr(10),' ')}{'...' if len(raw_text)>120 else ''}")
    return raw_text


# ── BATCH OCR: all sheets through Uni-MuMER, save to OCR_CACHE_DIR ──────────

def batch_ocr(sheets_bytes: list, student_names: list) -> list:
    """
    Runs Uni-MuMER on every sheet in one model load.
    Saves each result as a .json file in OCR_CACHE_DIR.

    Returns list of cache file paths in same order as input.
    """
    print("\n" + "═" * 60)
    print(f"🔍 [BATCH OCR] Starting — {len(sheets_bytes)} sheet(s)")
    print("═" * 60)

    # Clear old cache
    shutil.rmtree(OCR_CACHE_DIR, ignore_errors=True)
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)

    cache_paths = []

    for idx, (content, name) in enumerate(zip(sheets_bytes, student_names)):
        print(f"\n  📄 Sheet {idx+1}/{len(sheets_bytes)}: {name}")
        images    = convert_from_bytes(content)
        raw_text  = run_unimer_on_images(images, name)

        cache_path = os.path.join(OCR_CACHE_DIR, f"sheet_{idx:04d}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "student_name": name,
                "raw_text":     raw_text,
                "sheet_index":  idx,
            }, f, ensure_ascii=False, indent=2)

        cache_paths.append(cache_path)
        print(f"  💾 [OCR] Saved to {cache_path}")

    print(f"\n✅ [BATCH OCR] All {len(sheets_bytes)} sheets processed and cached.")
    print("═" * 60)
    return cache_paths


# =========================
# 🔹 HELPERS
# =========================

def wrap_into_lines(text: str, words_per_line: int = 8) -> list:
    words = text.split()
    return [" ".join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]


# =========================
# 🔹 QWEN PASS 1 — CLEAN TEXT
# =========================

def qwen_clean_text(raw_text: str, label: str = "") -> str:
    if not raw_text.strip():
        print(f"  ⚠️  [CLEAN] '{label}' — empty raw text, skipping.")
        return ""

    print(f"  🧹 [CLEAN] Cleaning OCR output for '{label}'...")
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
    print(f"  ✅ [CLEAN] '{label}' done in {time.time()-t0:.1f}s — {len(result)} chars.")
    return result


# =========================
# 🔹 QWEN PASS 2 — EXTRACT KEY POINTS
# =========================

def qwen_extract_key_points(cleaned_text: str, label: str = "") -> list:
    if not cleaned_text.strip():
        print(f"  ⚠️  [KP] '{label}' — empty text, returning [].")
        return []

    print(f"  🔑 [KP] Extracting key points for '{label}'...")
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
        print(f"  ⚠️  [KP] JSON parse failed for '{label}' — sentence split fallback.")
        sentences = re.split(r'[.;]\s*', cleaned_text)
        points    = [s.strip() for s in sentences if len(s.strip()) > 4]
    else:
        points = [str(p).strip() for p in points if str(p).strip()]

    print(f"  ✅ [KP] '{label}' — {len(points)} key points in {time.time()-t0:.1f}s.")
    for i, p in enumerate(points):
        print(f"       {i+1}. {p}")
    return points


# =========================
# 🔹 EMBEDDING SCORE
# =========================

def score_embedding(student_points: list, teacher_points: list,
                    marks: int, threshold: float = EMBEDDING_THRESHOLD) -> float:
    if not student_points or not teacher_points:
        print("  ⚠️  [EMB] Missing points — returning 0.")
        return 0.0

    embedder = get_embedder()
    t_embs   = embedder.encode(teacher_points)
    s_embs   = embedder.encode(student_points)
    sim_mat  = cosine_similarity(s_embs, t_embs)
    best     = sim_mat.max(axis=0)

    covered_count = int((best >= threshold).sum())
    coverage      = covered_count / len(teacher_points)
    emb_score     = round(coverage * marks, 2)

    print(f"  📐 [EMB] Coverage: {covered_count}/{len(teacher_points)} points ≥{threshold} → {emb_score}/{marks}")
    for ti, tp in enumerate(teacher_points):
        best_s = student_points[int(sim_mat[:, ti].argmax())]
        print(f"       TP{ti+1} [{best[ti]:.2f}] '{tp[:55]}' ← '{best_s[:55]}'")

    return emb_score


# =========================
# 🔹 LLM ENSEMBLE SCORE
# =========================

def score_llm_ensemble(student_points: list, key_answer: str,
                        marks: int, n_passes: int = ENSEMBLE_PASSES) -> tuple:
    print(f"  🤖 [LLM] Self-consistency: {n_passes} passes at temp=0.3...")
    t0 = time.time()

    qwen, _  = get_qwen()
    sampling = SamplingParams(temperature=0.3, max_tokens=256)
    student_str = "\n".join(f"- {p}" for p in student_points) if student_points else "(no answer detected)"

    prompt = (
        "<|im_start|>system\n"
        "You are a strict but fair exam evaluator. "
        "Respond ONLY with raw JSON — no markdown, no prose: "
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
        raw    = "{" + qwen.generate([prompt], sampling)[0].outputs[0].text
        raw    = raw.replace("<|im_end|>", "").strip()
        parsed = extract_json(raw, dict)
        if parsed:
            s = max(0.0, min(float(marks), float(parsed.get("score", 0))))
            scores.append(s)
            if not remarks:
                remarks = parsed.get("remarks", "")
            print(f"       Pass {i+1}: {s}/{marks}")
        else:
            print(f"       Pass {i+1}: ⚠️  parse failed — {raw[:60]}")

    if not scores:
        print("  ⚠️  [LLM] All passes failed — returning 0.")
        return 0.0, ""

    final  = float(median(scores))
    spread = max(scores) - min(scores)
    print(f"  ✅ [LLM] Passes={scores}  spread={spread:.2f}  median={final}/{marks}  ({time.time()-t0:.1f}s)")
    return final, remarks


# =========================
# 🔹 HYBRID SCORE
# =========================

def hybrid_score(student_points: list, teacher_points: list,
                 key_answer: str, marks: int, alpha: float = ALPHA) -> dict:
    print(f"  ⚖️  [HYBRID] α={alpha}  marks={marks}")
    emb_score          = score_embedding(student_points, teacher_points, marks)
    llm_score, remarks = score_llm_ensemble(student_points, key_answer, marks)
    final              = round(alpha * emb_score + (1 - alpha) * llm_score, 2)
    print(f"  ✅ [HYBRID] emb={emb_score}  llm={llm_score}  final={final}/{marks}")
    return {"score": final, "emb_score": emb_score, "llm_score": llm_score, "remarks": remarks}


# =========================
# 🔹 CLUSTER SCORE BANDS
# (compare cluster centroids to teacher answer embedding)
# =========================

def assign_cluster_score_bands(
    all_embs: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    teacher_emb: np.ndarray,
    marks: int,
    k: int,
) -> dict:
    """
    For each cluster, compute cosine similarity between its centroid
    and the teacher answer embedding.  Map similarity → score band.

    Returns {cluster_id: {"band": str, "band_score": float}}
    """
    # teacher_emb shape: (1, D) or (D,)
    if teacher_emb.ndim == 1:
        teacher_emb = teacher_emb.reshape(1, -1)

    bands = {}
    print("  🎯 [BAND] Computing cluster score bands vs teacher embedding...")
    for cid in range(k):
        centroid = centroids[cid].reshape(1, -1)
        sim      = float(cosine_similarity(centroid, teacher_emb)[0][0])

        # Map cosine similarity to a grade band
        if sim >= 0.80:
            band       = "Excellent"
            band_score = round(marks * 1.0, 2)
        elif sim >= 0.65:
            band       = "Good"
            band_score = round(marks * 0.75, 2)
        elif sim >= 0.50:
            band       = "Average"
            band_score = round(marks * 0.55, 2)
        elif sim >= 0.35:
            band       = "Below Average"
            band_score = round(marks * 0.35, 2)
        else:
            band       = "Poor"
            band_score = round(marks * 0.15, 2)

        bands[cid] = {"band": band, "band_score": band_score, "centroid_sim": round(sim, 4)}
        print(f"     Cluster {cid}: sim={sim:.4f} → {band} ({band_score}/{marks})")

    return bands


# =========================
# 🔹 EVALUATE (batch, Qwen already loaded)
# =========================

def evaluate_student(student_key_points: list, questions_data: list,
                     teacher_key_points_cache: dict) -> dict:
    """
    Hybrid scoring for one student.
    teacher_key_points_cache: {question_index: list} — pre-extracted, reused across students.
    """
    total_score  = 0.0
    max_total    = 0
    per_question = []

    for i, q in enumerate(questions_data):
        key_answer     = q.get("keyAnswer", "")
        marks          = q.get("marks", 1)
        max_total     += marks
        teacher_points = teacher_key_points_cache.get(i, [])

        if not teacher_points:
            per_question.append({
                "score": 0, "marks": marks,
                "remarks": "No teacher key points extracted.",
                "emb_score": 0, "llm_score": 0,
                "band": "N/A", "band_score": 0,
            })
            continue

        result       = hybrid_score(student_key_points, teacher_points, key_answer, marks)
        total_score += result["score"]
        per_question.append({
            "score":     result["score"],
            "marks":     marks,
            "remarks":   result["remarks"],
            "emb_score": result["emb_score"],
            "llm_score": result["llm_score"],
        })

    return {
        "score":        round(total_score, 2),
        "total":        max_total,
        "per_question": per_question,
    }


# =========================
# 🔹 CLUSTER FEEDBACK
# =========================

def generate_cluster_feedback(all_students_data: list, key_answer: str,
                               teacher_key_points: list, marks: int) -> list:
    """
    K-means cluster student answers.
    1. Assign score bands based on centroid-to-teacher similarity.
    2. Generate Qwen feedback once per cluster centroid.
    3. Assign both to every student in the cluster.
    """
    n = len(all_students_data)
    if n == 0:
        return all_students_data

    k = min(N_CLUSTERS, n)
    print("━" * 50)
    print(f"🔵 [CLUSTER] {n} student(s) → k={k} clusters")

    embedder = get_embedder()
    texts    = [" ".join(s.get("key_points", [])) for s in all_students_data]
    embs     = embedder.encode(texts)                  # (N, D)

    # Also embed the teacher answer for band comparison
    teacher_text = " ".join(teacher_key_points) if teacher_key_points else key_answer
    teacher_emb  = embedder.encode([teacher_text])     # (1, D)

    if n == 1:
        labels    = np.array([0])
        centroids = embs.reshape(1, -1)
    else:
        km        = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs)
        labels    = km.labels_
        centroids = km.cluster_centers_

    dist_counts = {int(c): int((labels == c).sum()) for c in range(k)}
    print(f"   Distribution: {dist_counts}")

    # ── Score bands from centroid similarity ─────────────────────────────
    score_bands = assign_cluster_score_bands(embs, labels, centroids, teacher_emb, marks, k)

    # ── Qwen feedback per cluster ─────────────────────────────────────────
    qwen, sampling = get_qwen()
    bloom_levels   = ["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]
    cluster_data   = {}

    for cid in range(k):
        members = [i for i, l in enumerate(labels) if l == cid]
        if not members:
            continue

        dists   = [np.linalg.norm(embs[i] - centroids[cid]) for i in members]
        rep_idx = members[int(np.argmin(dists))]
        rep     = all_students_data[rep_idx]

        print(f"\n  🔵 Cluster {cid} ({len(members)} student(s))  rep=index {rep_idx}  band={score_bands[cid]['band']}")

        student_str = "\n".join(f"- {p}" for p in rep.get("key_points", [])) or "(none)"

        # Pass A: Strengths / Improvements / Suggestions
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
            print(f"  ⚠️  [CLUSTER] Feedback parse failed for cluster {cid}.")
            feedback = {"strengths": [], "improvements": [], "suggestions": []}
        else:
            for key in ["strengths", "improvements", "suggestions"]:
                if not isinstance(feedback.get(key), list):
                    feedback[key] = []

        # Pass B: Bloom's Taxonomy
        prompt_b = (
            "<|im_start|>system\n"
            "You are an expert in Bloom's Taxonomy for exam evaluation.\n"
            "For each of the 6 Bloom's levels, rate:\n"
            "1. 'required': how much the KEY ANSWER demands this level (0-100).\n"
            "2. 'demonstrated': how much the STUDENT showed this level (0 to required max).\n"
            "demonstrated CANNOT exceed required. Be strict — most answers only require 2-3 levels strongly.\n"
            "Return ONLY a raw JSON array of exactly 6 objects. No prose, no markdown.\n"
            'Format: [{"level":"Remember","required":80,"demonstrated":60}]'
            "<|im_end|>\n"
            f"<|im_start|>user\nKey Answer:\n{key_answer}\n\n"
            f"Student Key Points:\n{student_str}<|im_end|>\n"
            "<|im_start|>assistant\n["
        )
        raw_b  = "[" + qwen.generate([prompt_b], sampling)[0].outputs[0].text
        raw_b  = raw_b.replace("<|im_end|>", "").strip()
        blooms = extract_json(raw_b, list)

        if not blooms:
            print(f"  ⚠️  [CLUSTER] Bloom's parse failed for cluster {cid} — zeros.")
            blooms = [{"level": l, "required": 0, "demonstrated": 0} for l in bloom_levels]
        else:
            bmap   = {b.get("level"): b for b in blooms if isinstance(b, dict)}
            blooms = []
            for level in bloom_levels:
                entry = bmap.get(level, {})
                req   = max(0, min(100, int(entry.get("required", 0))))
                dem   = max(0, min(req,  int(entry.get("demonstrated", 0))))
                blooms.append({"level": level, "required": req, "demonstrated": dem})

        print(f"     Strengths={len(feedback['strengths'])}  Improvements={len(feedback['improvements'])}  Suggestions={len(feedback['suggestions'])}")
        print(f"     Bloom's: {[(b['level'][:3], b['demonstrated']) for b in blooms]}")

        cluster_data[cid] = {
            "analysis": {
                "strengths":    feedback.get("strengths", []),
                "improvements": feedback.get("improvements", []),
                "suggestions":  feedback.get("suggestions", []),
                "blooms":       blooms,
            },
            "band":      score_bands[cid]["band"],
            "band_score": score_bands[cid]["band_score"],
        }

    # Assign to every student
    for i, student in enumerate(all_students_data):
        cid               = int(labels[i])
        cdata             = cluster_data.get(cid, {})
        student["analysis"]   = cdata.get("analysis", {"strengths": [], "improvements": [], "suggestions": [], "blooms": []})
        student["cluster_id"] = cid
        student["band"]       = cdata.get("band", "N/A")
        student["band_score"] = cdata.get("band_score", 0)

    print(f"\n✅ [CLUSTER] Done — {k} clusters, {n} students assigned.")
    print("━" * 50)
    return all_students_data


# =========================
# 🔹 HEALTH
# =========================

@app.get("/health")
async def health():
    return {"status": "running"}


# =========================
# 🔹 BATCH SIMILARITY ENDPOINT
# Accepts multiple answer sheets for the same question set.
# Pipeline:
#   Phase 1 — Uni-MuMER: OCR all sheets → cache to disk → unload
#   Phase 2 — Qwen: clean + extract + score all from cache → unload
#   Phase 3 — Clustering: assign score bands + generate batch feedback
# =========================

@app.post("/similarity")
async def similarity(
    request: Request,
    questions: str = Form(...),
    answer_sheets: List[UploadFile] = File(...),
    student_names: str = Form(default="[]"),   # JSON array of names, optional
):
    request_start = time.time()

    try:
        print("\n" + "═" * 60)
        print("🚀 [REQUEST] /similarity — batch pipeline start")
        print("═" * 60)

        questions_data = json.loads(questions)
        names_list     = json.loads(student_names)

        # Pad / generate student names
        sheet_names = []
        for idx in range(len(answer_sheets)):
            if idx < len(names_list) and names_list[idx].strip():
                sheet_names.append(names_list[idx].strip())
            else:
                sheet_names.append(answer_sheets[idx].filename or f"Student_{idx+1}")

        print(f"   Questions     : {len(questions_data)}")
        print(f"   Answer sheets : {len(answer_sheets)}")
        print(f"   Student names : {sheet_names}")

        if not answer_sheets:
            return {"error": "No answer sheets uploaded"}

        # ── Read all file bytes upfront (async, before any model load) ────
        all_bytes = []
        for sheet in answer_sheets:
            content = await sheet.read()
            all_bytes.append(content)

        # ═══════════════════════════════════════════════════
        # PHASE 1 — UNI-MuMER: OCR all sheets
        # ═══════════════════════════════════════════════════
        print("\n" + "─" * 60)
        print("🔍 PHASE 1 — UNI-MuMER OCR (all sheets)")
        print("─" * 60)
        phase1_start = time.time()

        cache_paths = batch_ocr(all_bytes, sheet_names)

        unload_unimer()  # explicitly unload after all OCR is done
        print(f"⏱️  Phase 1 complete in {time.time()-phase1_start:.1f}s")

        # ═══════════════════════════════════════════════════
        # PHASE 2 — QWEN: clean + extract + score all sheets
        # ═══════════════════════════════════════════════════
        print("\n" + "─" * 60)
        print("🧠 PHASE 2 — QWEN processing (all sheets)")
        print("─" * 60)
        phase2_start = time.time()

        # Pre-extract teacher key points ONCE per question (reused for all students)
        print("\n  📚 [EVAL] Pre-extracting teacher key points...")
        teacher_kp_cache = {}
        for i, q in enumerate(questions_data):
            print(f"    Q{i+1}:")
            teacher_kp_cache[i] = qwen_extract_key_points(q.get("keyAnswer", ""), label=f"Teacher Q{i+1}")

        # Process each student
        student_results = []
        for idx, cache_path in enumerate(cache_paths):
            name = sheet_names[idx]
            print(f"\n  {'─'*45}")
            print(f"  👤 [{idx+1}/{len(cache_paths)}] Processing: {name}")
            print(f"  {'─'*45}")

            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            raw_text = cached["raw_text"]

            # Pass 1: clean
            cleaned_text = qwen_clean_text(raw_text, label=name)

            # Pass 2: extract student key points
            key_points = qwen_extract_key_points(cleaned_text, label=name)

            # Pass 3: hybrid score
            print(f"\n  📝 [EVAL] Scoring {name}...")
            eval_result = evaluate_student(key_points, questions_data, teacher_kp_cache)

            student_results.append({
                "student_name":   name,
                "student_index":  idx,
                "key_points":     key_points,
                "cleaned_text":   cleaned_text,
                "student_answer": wrap_into_lines(cleaned_text, words_per_line=8),
                "score":          eval_result["score"],
                "total":          eval_result["total"],
                "per_question":   eval_result["per_question"],
            })

            print(f"  ✅ {name}: {eval_result['score']}/{eval_result['total']}")

        print(f"\n⏱️  Phase 2 complete in {time.time()-phase2_start:.1f}s")

        # ═══════════════════════════════════════════════════
        # PHASE 3 — CLUSTERING: score bands + batch feedback
        # ═══════════════════════════════════════════════════
        print("\n" + "─" * 60)
        print("🔵 PHASE 3 — CLUSTERING & FEEDBACK")
        print("─" * 60)
        phase3_start = time.time()

        # Use first question's key answer and teacher points for clustering
        # (for multi-question papers, clustering on Q1 is standard practice)
        primary_key_answer    = questions_data[0].get("keyAnswer", "") if questions_data else ""
        primary_teacher_kp    = teacher_kp_cache.get(0, [])
        primary_marks         = questions_data[0].get("marks", 1) if questions_data else 1

        student_results = generate_cluster_feedback(
            student_results,
            primary_key_answer,
            primary_teacher_kp,
            primary_marks,
        )

        print(f"⏱️  Phase 3 complete in {time.time()-phase3_start:.1f}s")

        # ── Build final response ──────────────────────────────────────────
        total_elapsed = time.time() - request_start

        # Summary stats
        all_scores      = [s["score"] for s in student_results]
        score_total     = student_results[0]["total"] if student_results else 0
        avg_score       = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
        highest         = max(all_scores) if all_scores else 0
        lowest          = min(all_scores) if all_scores else 0

        final_response = {
            "batch_size":    len(student_results),
            "average_score": avg_score,
            "highest_score": highest,
            "lowest_score":  lowest,
            "total_marks":   score_total,
            "students":      student_results,
        }

        print("\n" + "═" * 60)
        print(f"🏁 [DONE] Batch complete in {total_elapsed:.1f}s")
        print(f"   Students  : {len(student_results)}")
        print(f"   Avg score : {avg_score}/{score_total}")
        print(f"   Highest   : {highest}   Lowest: {lowest}")
        for s in student_results:
            print(f"   {s['student_name']:30s} {s['score']}/{s['total']}  [{s.get('band','N/A')}]")
        print("═" * 60)

        # If only one student, also return flat keys so existing frontend still works
        if len(student_results) == 1:
            sr = student_results[0]
            final_response.update({
                "score":          sr["score"],
                "total":          sr["total"],
                "student_answer": sr["student_answer"],
                "key_points":     sr["key_points"],
                "per_question":   sr["per_question"],
            })

        return final_response

    except Exception as e:
        traceback.print_exc()
        print(f"❌ [ERROR] /similarity failed: {e}")
        return {"error": str(e)}


# =========================
# 🔹 ANALYSE ENDPOINT
# (single student, called from frontend after evaluation)
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

        print(f"   Key answer : {len(key_answer)} chars")
        print(f"   Key points : {len(key_points)}")
        print(f"   Score      : {score}/{total}")

        # Extract teacher key points for clustering
        teacher_points = qwen_extract_key_points(key_answer, label="Teacher (analyse)")

        student_data = [{
            "key_points":   key_points,
            "cleaned_text": " ".join(key_points),
        }]

        student_data = generate_cluster_feedback(
            student_data,
            key_answer,
            teacher_points,
            total,
        )

        analysis = student_data[0].get("analysis", {})
        result   = {
            "strengths":    analysis.get("strengths", []),
            "improvements": analysis.get("improvements", []),
            "suggestions":  analysis.get("suggestions", []),
            "blooms":       analysis.get("blooms", []),
            "band":         student_data[0].get("band", "N/A"),
            "band_score":   student_data[0].get("band_score", 0),
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