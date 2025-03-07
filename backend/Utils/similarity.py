from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from math import ceil

def text_similarity(key_answer, test_answer, model_name = 'stsb-roberta-large') -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name).to(device) 
    key_embeddings = model.encode(key_answer)
    test_embeddings = model.encode(test_answer)

    #averaging the embeddings
    # avg_key_embedding = np.mean(key_embeddings, axis=0) 
    # avg_test_embedding = np.mean(test_embeddings, axis=0)
    
    # similarity = cosine_similarity(avg_key_embedding, avg_test_embedding)
    similarity = cosine_similarity([key_embeddings], [test_embeddings])[0][0]    
    print(f"Similarity: {similarity}") 
    return ceil(similarity.item() * 10) / 10

