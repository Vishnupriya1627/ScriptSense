"""
evaluation.py  —  ScriptSense answer evaluation pipeline
Place this file in: backend/evaluation.py

Pipeline per question:
  1. BERTScore zero-filter  → drop clearly wrong answers (score < threshold)
  2. SBERT clustering       → group similar answers; judge once per cluster centroid
  3. Qwen 1.5B LLM judge   → score cluster centroid vs answer key
  4. Confidence escalation  → low-confidence centroids re-judged by Qwen 3B/7B
  5. Score propagation      → all students in a cluster inherit centroid score

Dependencies (add to requirements.txt if missing):
    sentence-transformers
    bert-score
    transformers
    torch
    scikit-learn
    numpy
    fastapi
    pydantic

Models are loaded lazily on first use.
On Colab: SBERT + BERTScore run fine on T4.
Qwen models must already be on Google Drive (as you have them).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("scriptsense.evaluation")

# ─────────────────────────────────────────────
# CONFIG  (tweak these without touching logic)
# ─────────────────────────────────────────────
BERT_SCORE_THRESHOLD = 0.55      # below this → zero score immediately
SBERT_MODEL_NAME     = "all-MiniLM-L6-v2"   # fast, good quality
CLUSTER_EPS          = 0.30      # DBSCAN eps (cosine distance); lower = tighter clusters
CLUSTER_MIN_SAMPLES  = 1         # allow singleton clusters
QWEN_SMALL_MODEL     = os.getenv("QWEN_SMALL_MODEL", "/content/drive/MyDrive/models/Uni-MuMER-3B")
QWEN_LARGE_MODEL     = os.getenv("QWEN_LARGE_MODEL", "/content/drive/MyDrive/models/Uni-MuMER-3B")
CONFIDENCE_THRESHOLD = 0.75      # qwen small score confidence below this → escalate
QWEN_MAX_NEW_TOKENS  = 256


# ─────────────────────────────────────────────
# LAZY MODEL REGISTRY
# ─────────────────────────────────────────────
_models: dict[str, Any] = {}


def _get_sbert():
    if "sbert" not in _models:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SBERT model …")
        _models["sbert"] = SentenceTransformer(SBERT_MODEL_NAME)
    return _models["sbert"]


def _get_qwen(size: str = "small"):
    """size: 'small' | 'large'"""
    key = f"qwen_{size}"
    if key not in _models:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = QWEN_SMALL_MODEL if size == "small" else QWEN_LARGE_MODEL
        logger.info(f"Loading Qwen {size} from {path} …")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _models[key] = (tokenizer, model)
    return _models[key]


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────
class QuestionAnswer(BaseModel):
    question_id: str
    ocr_text: str          # OCR output for this question
    max_marks: float = 1.0


class StudentSheet(BaseModel):
    student_id: str
    answers: list[QuestionAnswer]


class AnswerKeyItem(BaseModel):
    question_id: str
    expected_answer: str
    max_marks: float = 1.0


class EvaluationRequest(BaseModel):
    answer_key: list[AnswerKeyItem]
    student_sheets: list[StudentSheet]


@dataclass
class QuestionResult:
    question_id: str
    score: float
    max_marks: float
    feedback: str
    method: str = "llm"          # "zero_filter" | "cluster_cache" | "llm" | "llm_escalated"


@dataclass
class StudentResult:
    student_id: str
    total_score: float
    max_total: float
    question_results: list[QuestionResult] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        return round(self.total_score / self.max_total * 100, 1) if self.max_total else 0.0


# ─────────────────────────────────────────────
# STEP 1 — BERTScore zero-filter
# ─────────────────────────────────────────────
def bertscore_filter(
    student_texts: list[str],
    reference: str,
    threshold: float = BERT_SCORE_THRESHOLD,
) -> tuple[list[bool], list[float]]:
    """
    Returns:
        passes  : list[bool]   — True if answer passes filter
        f1s     : list[float]  — BERTScore F1 per answer
    """
    from bert_score import score as bscore  # import here to keep startup fast

    if not student_texts:
        return [], []

    references = [reference] * len(student_texts)
    _, _, F1 = bscore(
        student_texts,
        references,
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    f1s = F1.tolist()
    passes = [f >= threshold for f in f1s]
    return passes, f1s


# ─────────────────────────────────────────────
# STEP 2 — SBERT clustering (DBSCAN cosine)
# ─────────────────────────────────────────────
def cluster_answers(texts: list[str]) -> tuple[list[int], np.ndarray]:
    """
    Returns:
        labels      : cluster label per text (-1 = noise → treated as singleton)
        embeddings  : (N, D) array
    """
    from sklearn.cluster import DBSCAN

    sbert = _get_sbert()
    embeddings = sbert.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    if len(texts) == 1:
        return [0], embeddings

    db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embeddings).tolist()

    # Remap -1 (noise) to unique labels so every answer is in a cluster
    next_label = max(labels) + 1
    remapped = []
    for lbl in labels:
        if lbl == -1:
            remapped.append(next_label)
            next_label += 1
        else:
            remapped.append(lbl)

    return remapped, embeddings


def find_centroid_index(indices: list[int], embeddings: np.ndarray) -> int:
    """Return index of the answer closest to the cluster mean embedding."""
    cluster_embs = embeddings[indices]
    mean = cluster_embs.mean(axis=0)
    dists = np.linalg.norm(cluster_embs - mean, axis=1)
    return indices[int(np.argmin(dists))]


# ─────────────────────────────────────────────
# STEP 3 & 4 — Qwen LLM judge with escalation
# ─────────────────────────────────────────────
_JUDGE_PROMPT = """You are an exam answer evaluator. Given the expected answer and a student's answer, score the student.

Expected answer: {reference}
Student's answer: {student}
Maximum marks: {max_marks}

Reply with ONLY a JSON object like:
{{"score": <number>, "confidence": <0.0-1.0>, "feedback": "<one sentence>"}}
Do not include any other text."""


def _call_qwen(prompt: str, size: str = "small") -> dict:
    """Call Qwen and parse JSON response. Returns dict with score/confidence/feedback."""
    import json
    import re

    tokenizer, model = _get_qwen(size)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Extract JSON block
    match = re.search(r"\{.*?\}", generated, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract score with regex
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', generated)
    score = float(score_match.group(1)) if score_match else 0.0
    return {"score": score, "confidence": 0.5, "feedback": generated[:200]}


def judge_answer(
    student_text: str,
    reference: str,
    max_marks: float,
) -> tuple[float, float, str, str]:
    """
    Returns: (score, confidence, feedback, method_used)
    """
    prompt = _JUDGE_PROMPT.format(
        reference=reference,
        student=student_text,
        max_marks=max_marks,
    )

    result = _call_qwen(prompt, size="small")
    score      = min(float(result.get("score", 0.0)), max_marks)
    confidence = float(result.get("confidence", 0.5))
    feedback   = result.get("feedback", "")
    method     = "llm"

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"Low confidence ({confidence:.2f}) — escalating to large Qwen …")
        try:
            result2 = _call_qwen(prompt, size="large")
            score    = min(float(result2.get("score", score)), max_marks)
            feedback = result2.get("feedback", feedback)
            method   = "llm_escalated"
        except Exception as e:
            logger.warning(f"Escalation failed: {e} — using small model score")

    return score, confidence, feedback, method


# ─────────────────────────────────────────────
# MAIN EVALUATION FUNCTION  (per question)
# ─────────────────────────────────────────────
def evaluate_question(
    question_id: str,
    student_texts: list[str],          # one per student, same order as student_ids
    student_ids: list[str],
    reference: str,
    max_marks: float,
) -> dict[str, QuestionResult]:
    """
    Runs the full pipeline for one question across all students.
    Returns: {student_id: QuestionResult}
    """
    results: dict[str, QuestionResult] = {}

    # ── 1. BERTScore zero-filter ──
    passes, f1s = bertscore_filter(student_texts, reference)

    survivor_indices = []
    for i, (sid, text, passed, f1) in enumerate(
        zip(student_ids, student_texts, passes, f1s)
    ):
        if not passed or not text.strip():
            results[sid] = QuestionResult(
                question_id=question_id,
                score=0.0,
                max_marks=max_marks,
                feedback=f"Answer too dissimilar to expected (BERTScore F1={f1:.2f}).",
                method="zero_filter",
            )
            logger.info(f"[{question_id}] {sid} → zero-filtered (F1={f1:.2f})")
        else:
            survivor_indices.append(i)

    if not survivor_indices:
        return results

    # ── 2. SBERT clustering on survivors ──
    survivor_texts = [student_texts[i] for i in survivor_indices]
    survivor_ids   = [student_ids[i]   for i in survivor_indices]

    labels, embeddings = cluster_answers(survivor_texts)

    # Group by cluster
    clusters: dict[int, list[int]] = {}
    for local_idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(local_idx)

    # ── 3 & 4. Judge one centroid per cluster; propagate to members ──
    for label, local_indices in clusters.items():
        centroid_local = find_centroid_index(local_indices, embeddings)
        centroid_text  = survivor_texts[centroid_local]

        score, confidence, feedback, method = judge_answer(
            centroid_text, reference, max_marks
        )

        cluster_size = len(local_indices)
        logger.info(
            f"[{question_id}] cluster {label} "
            f"(size={cluster_size}) → score={score}/{max_marks} "
            f"conf={confidence:.2f} method={method}"
        )

        # Propagate score to all cluster members
        for local_idx in local_indices:
            sid = survivor_ids[local_idx]
            member_method = method if local_idx == centroid_local else "cluster_cache"
            results[sid] = QuestionResult(
                question_id=question_id,
                score=score,
                max_marks=max_marks,
                feedback=feedback,
                method=member_method,
            )

    return results


# ─────────────────────────────────────────────
# BATCH EVALUATION  (all students × all questions)
# ─────────────────────────────────────────────
def run_batch_evaluation(request: EvaluationRequest) -> list[StudentResult]:
    """
    Synchronous batch evaluation.
    Called from the FastAPI endpoint via asyncio.to_thread.
    """
    # Build lookup: question_id → AnswerKeyItem
    key_map = {item.question_id: item for item in request.answer_key}

    # Build lookup: question_id → {student_id: ocr_text}
    qid_to_student_answers: dict[str, dict[str, str]] = {
        qid: {} for qid in key_map
    }
    for sheet in request.student_sheets:
        for qa in sheet.answers:
            if qa.question_id in qid_to_student_answers:
                qid_to_student_answers[qa.question_id][sheet.student_id] = qa.ocr_text

    # Collect per-student per-question results
    all_results: dict[str, dict[str, QuestionResult]] = {
        sheet.student_id: {} for sheet in request.student_sheets
    }

    for qid, key_item in key_map.items():
        student_map = qid_to_student_answers[qid]
        if not student_map:
            continue

        student_ids   = list(student_map.keys())
        student_texts = [student_map[sid] for sid in student_ids]

        q_results = evaluate_question(
            question_id=qid,
            student_texts=student_texts,
            student_ids=student_ids,
            reference=key_item.expected_answer,
            max_marks=key_item.max_marks,
        )

        for sid, qr in q_results.items():
            all_results[sid][qid] = qr

    # Build final StudentResult list
    final: list[StudentResult] = []
    for sheet in request.student_sheets:
        sid = sheet.student_id
        q_results_list = list(all_results[sid].values())
        total    = sum(r.score     for r in q_results_list)
        max_total = sum(r.max_marks for r in q_results_list)
        final.append(StudentResult(
            student_id=sid,
            total_score=round(total, 2),
            max_total=round(max_total, 2),
            question_results=q_results_list,
        ))

    return final


# ─────────────────────────────────────────────
# FASTAPI ROUTER  — mount this in main.py
# ─────────────────────────────────────────────
router = APIRouter(prefix="/evaluate", tags=["evaluation"])


class EvaluationResponse(BaseModel):
    results: list[dict]


@router.post("/batch", response_model=EvaluationResponse)
async def batch_evaluate(request: EvaluationRequest):
    """
    POST /evaluate/batch
    Accepts multiple student sheets + answer key.
    Returns scores and feedback per student per question.
    """
    student_results = await asyncio.to_thread(run_batch_evaluation, request)

    output = []
    for sr in student_results:
        output.append({
            "student_id":   sr.student_id,
            "total_score":  sr.total_score,
            "max_total":    sr.max_total,
            "percentage":   sr.percentage,
            "questions": [
                {
                    "question_id": qr.question_id,
                    "score":       qr.score,
                    "max_marks":   qr.max_marks,
                    "feedback":    qr.feedback,
                    "method":      qr.method,
                }
                for qr in sr.question_results
            ],
        })

    return {"results": output}


@router.post("/single", response_model=EvaluationResponse)
async def single_evaluate(sheet: StudentSheet, answer_key: list[AnswerKeyItem]):
    """
    POST /evaluate/single
    Convenience endpoint for one student at a time.
    """
    req = EvaluationRequest(answer_key=answer_key, student_sheets=[sheet])
    return await batch_evaluate(req)