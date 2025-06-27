import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from collections import defaultdict

# --------------------------
# NLP PREPROCESSING
# --------------------------
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Advanced text normalization with spaCy"""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop 
        and not token.is_punct
        and token.is_alpha
    ]
    return " ".join(tokens)

# --------------------------
# DOMAIN PROFILES (Expanded with LLM-generated keywords)
# --------------------------
domain_keywords = {
    "Full Stack": "html css javascript react nodejs api django flask frontend backend webdev restapi redux express mongodb",
    "AI": "machinelearning ml nlp computer vision cv neuralnetworks tensorflow pytorch supervised unsupervised reinforcementlearning",
    "Data Science": "pandas numpy matplotlib seaborn sklearn statistics sql tableau powerbi spark hadoop datamining etl",
    "Gen AI": "llm chatgpt gpt4 generativeai diffusion stable-diffusion midjourney promptengineering langchain embeddings transformers",
    "Python": "python scripting automation flask django fastapi pandas numpy pytest celery asyncio multiprocessing"
}

# --------------------------
# LEVEL SCORING (Weighted keywords)
# --------------------------
level_keywords = {
    "Beginner": {
        "terms": ["basics", "intro", "learning", "tutorial", "no experience"],
        "weight": 1
    },
    "Intermediate": {
        "terms": ["project", "api", "react", "nodejs", "pandas", "flask", "teamwork"],
        "weight": 2
    },
    "Advanced": {
        "terms": ["kubernetes", "docker", "llm", "distributed", "scalable", "ci/cd", "architecture"],
        "weight": 3
    }
}

# --------------------------
# LOAD & PREPROCESS USER DATA
# --------------------------
with open("data/user_answers.json") as f:
    user_data = json.load(f)

user_text = " ".join(
    preprocess(answer) 
    for item in user_data 
    for answer in item["answers"]
)

# --------------------------
# HYBRID DOMAIN PREDICTION
# --------------------------
def hybrid_domain_prediction(user_text, domains):
    """Combine BERT and TF-IDF with ensemble weighting"""
    
    # BERT Embeddings
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    domain_texts = list(domains.values()) + [user_text]
    bert_embeddings = bert_model.encode(domain_texts)
    bert_sim = cosine_similarity([bert_embeddings[-1]], bert_embeddings[:-1])[0]
    
    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(domain_texts)
    tfidf_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Ensemble (60% BERT, 40% TF-IDF)
    combined_scores = 0.6 * bert_sim + 0.4 * tfidf_sim
    best_idx = np.argmax(combined_scores)
    
    return {
        "domain": list(domains.keys())[best_idx],
        "confidence": float(combined_scores[best_idx]),
        "all_scores": dict(zip(domains.keys(), combined_scores))
    }

domain_result = hybrid_domain_prediction(user_text, domain_keywords)

# --------------------------
# ADVANCED LEVEL ESTIMATION
# --------------------------
def calculate_level_score(text):
    """Weighted scoring with term frequency analysis"""
    doc = nlp(text)
    score = 0
    term_counts = defaultdict(int)
    
    for level_name, level_info in level_keywords.items():
        for term in level_info["terms"]:
            if term in text:
                score += level_info["weight"] * text.count(term)
                term_counts[level_name] += text.count(term)  # Use level_name as key instead of the dict
    
    # Normalize by answer length
    normalized_score = score / (len(doc) + 1e-6) * 100
    
    if normalized_score < 30:
        return "Beginner", normalized_score, term_counts
    elif normalized_score < 70:
        return "Intermediate", normalized_score, term_counts
    else:
        return "Advanced", normalized_score, term_counts
level, level_score, detected_terms = calculate_level_score(user_text)

# --------------------------
# EVALUATION METRICS
# --------------------------
def print_metrics(domain_result, level_result):
    print(f"\n🔍 Recommended Domain: {domain_result['domain']} (Confidence: {domain_result['confidence']:.1%})")
    print(f"📊 Level: {level_result[0]} (Score: {level_result[1]:.1f}/100)")
    
    print("\n🔧 Detected Keywords:")
    for level, terms in level_result[2].items():
        if terms:
            print(f"  - {level}: {terms} matched terms")
    
    print("\n🌐 Domain Similarities:")
    for domain, score in domain_result['all_scores'].items():
        print(f"  - {domain.ljust(12)}: {score:.3f}")

print_metrics(domain_result, (level, level_score, detected_terms))

# --------------------------
# OPTIONAL: SAVE FOR ANALYSIS
# --------------------------
output = {
    "domain": domain_result["domain"],
    "confidence": domain_result["confidence"],
    "level": level,
    "level_score": level_score,
    "detected_terms": detected_terms
}

with open("assessment_results.json", "w") as f:
    json.dump(output, f, indent=2)