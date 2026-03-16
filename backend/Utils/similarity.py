from sentence_transformers import SentenceTransformer, util
import torch

def text_similarity(key_answer, test_answer, model_name='all-MiniLM-L6-v2') -> float:
    """
    Calculate semantic similarity between two texts.
    Returns: Float between 0.0 (different) and 1.0 (same meaning)
    """
    try:
        # Use CPU (faster for small models, no GPU needed)
        model = SentenceTransformer(model_name)
        
        # Encode both sentences
        embeddings = model.encode([key_answer, test_answer], 
                                 convert_to_tensor=True)
        
        # Compute cosine similarity
        cos_sim = util.cos_sim(embeddings[0], embeddings[1])
        similarity = cos_sim.item()
        
        # Normalize from [-1, 1] to [0, 1]
        normalized = (similarity + 1) / 2
        
        # Ensure it's between 0 and 1
        normalized = max(0.0, min(1.0, normalized))
        
        return float(normalized)
        
    except Exception as e:
        print(f"[text_similarity error] {e}")
        return 0.0
