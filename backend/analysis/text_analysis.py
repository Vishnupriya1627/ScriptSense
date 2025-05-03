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

def jaccard_similarity(text1, text2):
    """Measures similarity based on overlapping tokens"""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0
    
def levenshtein_distance_normalized(text1, text2):
    """Calculates edit distance normalized by length"""
    from Levenshtein import distance
    
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0  # Both strings are empty
        
    # Convert distance to similarity (1 - normalized distance)
    return 1 - (distance(text1, text2) / max_len)

def compare_embedding_models(key_answer, test_answer, model_names):
    """Compare similarity scores using different embedding models"""
    results = {}
    
    for model_name in model_names:
        sim = text_similarity(key_answer, test_answer, model_name)
        results[model_name] = sim
        
    return results

def length_adjusted_similarity(key_answer, test_answer, model_name='stsb-roberta-large'):
    """Adjusts similarity score based on length ratio"""
    base_similarity = text_similarity(key_answer, test_answer, model_name)
    
    # Calculate length ratio (smaller/larger)
    key_length = len(key_answer.split())
    test_length = len(test_answer.split())
    
    length_ratio = min(key_length, test_length) / max(key_length, test_length)
    
    # Apply length penalty (can be tuned)
    adjusted_similarity = base_similarity * (0.5 + 0.5 * length_ratio)
    
    return adjusted_similarity

def paragraph_similarity(key_answer, test_answer, model_name='stsb-roberta-large'):
    """Computes similarity at paragraph level and aggregates"""
    key_paragraphs = [p.strip() for p in key_answer.split('\n\n') if p.strip()]
    test_paragraphs = [p.strip() for p in test_answer.split('\n\n') if p.strip()]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name).to(device)
    
    # Encode all paragraphs
    key_embeddings = model.encode(key_paragraphs)
    test_embeddings = model.encode(test_paragraphs)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(key_embeddings, test_embeddings)
    
    # For each key paragraph, find the most similar test paragraph
    key_to_test_scores = [max(row) if len(row) > 0 else 0 for row in similarity_matrix]
    
    # For each test paragraph, find the most similar key paragraph
    test_to_key_scores = [max(col) if len(col) > 0 else 0 for col in similarity_matrix.T]
    
    # Combined score (bidirectional)
    if key_to_test_scores and test_to_key_scores:
        avg_key_to_test = sum(key_to_test_scores) / len(key_to_test_scores)
        avg_test_to_key = sum(test_to_key_scores) / len(test_to_key_scores)
        return (avg_key_to_test + avg_test_to_key) / 2
    else:
        return 0.0

def keyword_enhanced_similarity(key_answer, test_answer, model_name='stsb-roberta-large',
                               keyword_weight=0.3):
    """Enhances similarity with keyword matching"""
    import nltk
    from nltk.corpus import stopwords
    
    # Download required NLTK data (run once)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # Get base semantic similarity
    semantic_sim = text_similarity(key_answer, test_answer, model_name)
    
    # Extract keywords from key_answer (removing stopwords)
    stop_words = set(stopwords.words('english'))
    key_tokens = [w.lower() for w in nltk.word_tokenize(key_answer) 
                  if w.isalnum() and w.lower() not in stop_words]
    
    # Count keywords present in test_answer
    test_tokens = [w.lower() for w in nltk.word_tokenize(test_answer)]
    test_tokens_set = set(test_tokens)
    
    matched_keywords = sum(1 for word in key_tokens if word in test_tokens_set)
    keyword_coverage = matched_keywords / len(key_tokens) if key_tokens else 0
    
    # Combine semantic similarity with keyword coverage
    combined_sim = (1 - keyword_weight) * semantic_sim + keyword_weight * keyword_coverage
    
    return combined_sim

def domain_terminology_similarity(key_answer, test_answer, domain_terms, 
                                 model_name='stsb-roberta-large', term_weight=0.4):
    """Enhances similarity with domain-specific terminology matching"""
    # Get base semantic similarity
    semantic_sim = text_similarity(key_answer, test_answer, model_name)
    
    # Extract domain terms from key_answer
    key_terms = [term for term in domain_terms if term.lower() in key_answer.lower()]
    if not key_terms:
        return semantic_sim  # No domain terms in key answer
    
    # Check how many key terms appear in test_answer
    matched_terms = sum(1 for term in key_terms if term.lower() in test_answer.lower())
    term_coverage = matched_terms / len(key_terms) if key_terms else 0
    
    # Combine semantic similarity with terminology coverage
    combined_sim = (1 - term_weight) * semantic_sim + term_weight * term_coverage
    
    return combined_sim

# Example usage
domain_terms = ["neural network", "backpropagation", "gradient descent", "activation function"]


# Example models to compare
model_names = [
    'stsb-roberta-large',
    'bert-base-nli-mean-tokens' 
]

import pandas as pd
import numpy as np
from tabulate import tabulate

def generate_similarity_table(key_answers, test_answers, domain_terms=None, models=None):
    """Generates a comprehensive similarity analysis table"""
    if models is None:
        models = ['stsb-roberta-large']
    
    results = []
    
    for i, (key, test) in enumerate(zip(key_answers, test_answers)):
        row = {
            "Example": i+1,
            "Key Length": len(key.split()),
            "Test Length": len(test.split()),
            "Length Ratio": min(len(key.split()), len(test.split())) / 
                         max(len(key.split()), len(test.split()))
        }
        
        # Add semantic similarity for each model
        for model in models:
            row[f"Semantic ({model})"] = text_similarity(key, test, model)
        
        # Add lexical similarity
        row["Jaccard"] = jaccard_similarity(key, test)
        
        # Add term coverage if domain terms provided
        if domain_terms:
            key_terms = [term for term in domain_terms if term.lower() in key.lower()]
            if key_terms:
                matched_terms = sum(1 for term in key_terms if term.lower() in test.lower())
                row["Term Coverage"] = matched_terms / len(key_terms)
            else:
                row["Term Coverage"] = np.nan
        
        # Calculate combined score (can be customized)
        row["Combined Score"] = 0.6 * row[f"Semantic ({models[0]})"] + \
                              0.2 * row["Jaccard"] + \
                              0.2 * row["Length Ratio"]
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print table
    print(tabulate(df, headers='keys', tablefmt='grid'))
    
    return df

# Example usage
key_answers = [
    "Machine learning is a branch of artificial intelligence that enables computers to learn the data , identify patterns , and make decisions with minimal human intervention . Instead of being explicitly programmed for specific tasks , Machine learning algorithms analyze and interpret large datasets to Correlations that can be used for predictions or decision making . These algorithms Continuously improve their accuracy over time as they process more datasets to unconver trends and correlations that can be used for predictions or decision making. These algorithms continuously improve their accuracy over time as they process more data."
] * 3

test_answers = [
    "Machine learning is a branch of artifical intelligence that enables computers to learn from the data, identify patterns and make decisions with minimal human intervention. Instead of being explicitly programmed for specific tasks, machine learning algorithms analyze and interpret large datasets to uncover trends and correlations that can be used for predictions or decision making. These algorithms continuously improve their accuracy over time as they process more data.", 
    "Machine learning is a field of artificial intelligence that allows computers to learn from data, recognize patterns and make decisions with minimal human involvement. Rather than being explicity programmed for specific tasks, machine learning algorithms examine and interpret large datasets to identify meaningful relationships that aid in predictions or decision making. These algorithms continuously refine their accuracy over time by processing increase amounts of data, uncovering trends and correlations that enhance their predictive capabilities over time gradually.",
    "Cyber bullying is one of the significant problems that need to be eradicated. Due to cyber bullying, youngsters face many issues related to their health like depression, low self esteem, suicidal thoughts and it even leads to low academic performances. Cyber bullying is a form of bullying that takes place over digital devices like computers, tablets and mobile phones. It can take many forms such as sending mean messages, spreading rumors, sharing embarrassing photos or videos and impersonating someone online. Cyber bullying can happen 24/7 and it can be difficult to escape from it. It is important to raise awareness about cyber bullying and its effects on mental health. Schools, parents and communities should work together to educate youngsters about the dangers of cyber bullying and how to prevent it.",
]

domain_terms = [
    "algorithms", "artificial intelligence", "data", "patterns", "predictions",
    "decisions", "machine learning", "correlations", "accuracy", "datasets",
    "uncover", "trends", "correlations", "tasks", "programmed", "interpret",
    "branch", "learn", "identify", "human intervention", "explicitly",
    "analyze", "interpret", "large datasets", "correlations", "decision making",
    "improve", "accuracy", "process", "data", "uncover", "trends",
]

similarity_df = generate_similarity_table(key_answers, test_answers, domain_terms, models=model_names)

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_similarity_comparison(similarity_df):
    """Creates visualizations for similarity metrics comparison"""
    # Prepare data for visualization
    metric_cols = [col for col in similarity_df.columns 
                  if col not in ['Example', 'Key Length', 'Test Length']]
    
    # Melt the dataframe for easier plotting
    plot_df = similarity_df.melt(id_vars=['Example'], 
                                value_vars=metric_cols,
                                var_name='Metric', value_name='Score')
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Example', y='Score', hue='Metric', data=plot_df)
    plt.title('Comparison of Similarity Metrics Across Examples')
    plt.ylim(0, 1)
    plt.ylabel('Similarity Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('similarity_comparison.png', dpi=300)
    plt.show()
    
    # Create heatmap for correlation between metrics
    metric_data = similarity_df[metric_cols]
    plt.figure(figsize=(10, 8))
    sns.heatmap(metric_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Similarity Metrics')
    plt.tight_layout()
    plt.savefig('metric_correlation_heatmap.png', dpi=300)
    plt.show()

# Visualize the results
visualize_similarity_comparison(similarity_df)
