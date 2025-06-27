import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from collections import defaultdict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2
    ]
    return " ".join(tokens)

domain_keywords = {
    "Full Stack": "html css javascript react nodejs api django flask frontend backend webdev restapi redux express mongodb sql nosql websockets responsive",
    "AI": "machinelearning ml nlp computer vision cv neuralnetworks tensorflow pytorch supervised unsupervised reinforcementlearning classification regression clustering",
    "Data Science": "pandas numpy matplotlib seaborn sklearn statistics sql tableau powerbi spark hadoop datamining etl datawarehouse featureengineering",
    "Gen AI": "llm chatgpt gpt4 generativeai diffusion stable-diffusion midjourney promptengineering langchain embeddings transformers huggingface fine-tuning",
    "Python": "python scripting automation flask django fastapi pandas numpy pytest celery asyncio multiprocessing decorators generators"
}

level_keywords = {
    "Beginner": {
        "terms": [
            "basic", "introduction", "learn", "tutorial", "starting", "foundation", 
            "overview", "getting started", "hello world", "step-by-step", "beginner-friendly",
            "easy", "hands-on", "first project", "simple", "entry-level"
        ],
        "weight": 1
    },
    "Intermediate": {
        "terms": [
            "project", "implementation", "optimize", "debug", "analysis", "integration", 
            "api", "framework", "deployment", "testing", "version control", "git", 
            "docker", "refactor", "performance", "error handling", "modules", "libraries"
        ],
        "weight": 2
    },
    "Advanced": {
        "terms": [
            "architecture", "scalability", "distributed", "microservices", "high-availability",
            "deployment", "production-grade", "kubernetes", "llm", "fine-tuning", "optimization",
            "low-latency", "real-time", "asynchronous", "pipeline", "devops", "ci/cd",
            "auto-scaling", "containers", "observability", "monitoring", "orchestration",
            "zero-downtime", "security", "encryption"
        ],
        "weight": 3
    }
}

def load_user_answers(file_path):
    with open(file_path) as f:
        user_data = json.load(f)

    processed_texts = []
    for item in user_data:
        question = item.get("question", "")
        answers = " ".join(item.get("answers", []))
        combined = f"{question} {answers}"
        processed = preprocess(combined)
        processed_texts.append(processed)

    return " ".join(processed_texts), user_data

def hybrid_domain_prediction(user_text, domains):
    model = SentenceTransformer('all-mpnet-base-v2')

    domain_texts = [f"{name} {desc}" for name, desc in domains.items()]
    all_texts = domain_texts + [user_text]

    embeddings = model.encode(all_texts, convert_to_tensor=True)
    user_embedding = embeddings[-1].unsqueeze(0).cpu().numpy()
    domain_embeddings = embeddings[:-1].cpu().numpy()

    similarities = cosine_similarity(user_embedding, domain_embeddings)[0]

    # Soft normalize: clamp max sim to ~0.85 and scale to 50–75%
    sim_clipped = np.clip(similarities, 0.4, 0.85)
    sim_norm = (sim_clipped - 0.4) / (0.85 - 0.4)  # 0 → low match, 1 → high
    confidence_scores = sim_norm * 25 + 50  # Range ~50–75

    best_idx = int(np.argmax(confidence_scores))

    return {
        "domain": list(domains.keys())[best_idx],
        "confidence": round(float(confidence_scores[best_idx]), 2),
        "all_scores": dict(zip(domains.keys(), confidence_scores))
    }

def calculate_level_score(text, raw_data):
    doc = nlp(text)
    score = 0
    term_counts = defaultdict(int)
    answer_lengths = [len(answer) for item in raw_data for answer in item["answers"]]
    avg_answer_length = np.mean(answer_lengths) if answer_lengths else 0

    for level_name, level_info in level_keywords.items():
        for term in level_info["terms"]:
            if term in text:
                occurrences = text.count(term)
                score += level_info["weight"] * occurrences
                term_counts[level_name] += occurrences

    length_score = min(1, avg_answer_length / 500) * 30
    score += length_score

    complexity_features = {
        "avg_sentence_length": sum(len(sent) for sent in doc.sents) / (len(list(doc.sents)) + 1e-6),
        "unique_terms": len(set(token.text for token in doc)),
        "technical_terms": sum(1 for token in doc if token.text in domain_keywords.get(domain_result["domain"], ""))
    }
    complexity_score = min(30,
        complexity_features["avg_sentence_length"] * 0.2 +
        complexity_features["unique_terms"] * 0.05 +
        complexity_features["technical_terms"] * 0.5
    )
    score += complexity_score

    normalized_score = min(100, score * 100 / (len(doc) + 1e-6))

    if normalized_score < 33:
        return "Beginner", normalized_score, term_counts
    elif normalized_score < 66:
        return "Intermediate", normalized_score, term_counts
    else:
        return "Advanced", normalized_score, term_counts

def print_metrics(domain_result, level_result, raw_data):
    print(f"\nRecommended Domain: {domain_result['domain']} (Confidence: {domain_result['confidence']:.1f}%)")
    print(f"Level: {level_result[0]} (Score: {level_result[1]:.1f}/100)")

# Load user input
user_text, raw_user_data = load_user_answers("data/user_answers.json")

# Predict domain
domain_result = hybrid_domain_prediction(user_text, domain_keywords)

# Estimate level
level, level_score, detected_terms = calculate_level_score(user_text, raw_user_data)

# Print output
print_metrics(domain_result, (level, level_score, detected_terms), raw_user_data)

# Save results
output = {
    "domain": domain_result["domain"],
    "confidence": domain_result["confidence"],
    "level": level,
    "level_score": level_score,
    "detected_terms": detected_terms,
    "answer_stats": {
        "total_answers": len(raw_user_data),
        "avg_answer_length": np.mean([len(answer) for item in raw_user_data for answer in item["answers"]]),
        "total_words": len(user_text.split())
    }
}

with open("assessment_results.json", "w") as f:
    json.dump(output, f, indent=2)
