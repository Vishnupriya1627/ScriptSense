from sentence_transformers import SentenceTransformer, util
import torch
import re

def latex_to_readable(text):
    """Convert LaTeX symbols to readable text for better similarity matching."""
    text = text.replace('\\frac', 'fraction')
    text = text.replace('\\sqrt', 'square root')
    text = text.replace('\\sum', 'sum')
    text = text.replace('\\int', 'integral')
    text = text.replace('\\alpha', 'alpha')
    text = text.replace('\\beta', 'beta')
    text = text.replace('\\theta', 'theta')
    text = text.replace('^ { 2 }', 'squared')
    text = text.replace('^ { 3 }', 'cubed')
    text = re.sub(r'\^.*?\}', '', text)
    text = re.sub(r'[{}_\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_similarity(key_answer, test_answer, model_name='all-MiniLM-L6-v2') -> float:
    """
    Calculate semantic similarity between two texts.
    Returns: Float between 0.0 (different) and 1.0 (same meaning)
    """
    try:
        # Clean both inputs in case either is LaTeX
        key_clean = latex_to_readable(key_answer)
        test_clean = latex_to_readable(test_answer)

        model = SentenceTransformer(model_name)
        embeddings = model.encode([key_clean, test_clean], convert_to_tensor=True)
        cos_sim = util.cos_sim(embeddings[0], embeddings[1])
        similarity = cos_sim.item()
        normalized = (similarity + 1) / 2
        normalized = max(0.0, min(1.0, normalized))
        return float(normalized)
        
    except Exception as e:
        print(f"[text_similarity error] {e}")
        return 0.0