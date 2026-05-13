import os
import pickle
import numpy as np
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Chargement des variables d'environnement
load_dotenv()

# =========================================
# ⚙️ CONFIGURATION
# =========================================
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
DATA_PATH            = os.getenv("DATA_PATH", "rag_data.pkl")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
TOP_K                = int(os.getenv("TOP_K", 4))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))
DEVICE               = os.getenv("DEVICE", "cpu")
DEBUG                = os.getenv("DEBUG", "False").lower() == "true"

SYSTEM_PROMPT = (
    "Tu es un assistant expert pour analyser les réunions d'entreprise. "
    "Réponds UNIQUEMENT en te basant sur le contexte fourni. "
    "Sois précis, utile et garde un ton professionnel. Si la réponse ne figure "
    "absolument pas dans le contexte, signale-le."
)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)

# =========================================
# 🚀 INITIALISATION DES MODÈLES & DONNÉES
# =========================================
print("⏳ Chargement des modèles...")

try:
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    doc_embeddings = normalize(data["embeddings"])
    print(f"✅ {len(chunks)} segments chargés depuis {DATA_PATH}.")
except FileNotFoundError:
    print(f"❌ ERREUR : Le fichier {DATA_PATH} est introuvable. RAG impossible.")
    chunks = []
    doc_embeddings = []

model_emb = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print(f"✅ Gemini '{GEMINI_MODEL}' prêt.")


# =========================================
# 🧮 LOGIQUE MÉTIER (MMR RE-RANKING)
# =========================================
def mmr_rerank(query_emb, doc_embs, indices, top_k, lambda_param=0.6):
    selected = []
    candidate_indices = list(indices)

    while len(selected) < top_k and candidate_indices:
        best_idx, best_score = None, -np.inf
        
        for idx in candidate_indices:
            relevance = cosine_similarity(query_emb, [doc_embs[idx]])[0][0]
            
            if not selected:
                redundancy = 0
            else:
                redundancy = max(
                    cosine_similarity([doc_embs[idx]], [doc_embs[s]])[0][0]
                    for s in selected
                )
                
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx

        if best_idx is not None:
            selected.append(best_idx)
            candidate_indices.remove(best_idx)

    return selected


# =========================================
# 🌐 ROUTES API
# =========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "chunks": len(chunks), 
        "model": GEMINI_MODEL
    })

@app.route("/chat", methods=["POST"])
def query():
    body = request.get_json(silent=True)
    
    if not body or "question" not in body:
        return jsonify({"error": "Champ 'question' manquant."}), 400

    question = body["question"].strip()
    filter_speaker = body.get("filter_speaker", "all") 

    if not question:
        return jsonify({"error": "Question vide."}), 400

    # --- A. Retrieval ---
    query_emb = model_emb.encode([question], normalize_embeddings=True)
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]

    ranked_indices = np.argsort(similarities)[::-1]
    filtered_indices = [i for i in ranked_indices if similarities[i] >= SIMILARITY_THRESHOLD]

    # Force fallback if similarity threshold is too strict
    if not filtered_indices:
        filtered_indices = ranked_indices[:2] 

    # --- B. MMR Re-ranking ---
    top_indices = mmr_rerank(query_emb, doc_embeddings, filtered_indices[:TOP_K * 3], TOP_K)
    
    # --- C. Construction du contexte ---
    context_parts = []
    sources = []
    
    for i in top_indices:
        chunk_text = chunks[i]
        
        # Filtre sur l'intervenant
        if filter_speaker != "all" and filter_speaker.lower() not in chunk_text.lower():
            continue 

        context_parts.append(chunk_text)
        sources.append({
            "content": chunk_text[:300] + "...",
            "speaker": filter_speaker if filter_speaker != "all" else "Extrait",
            "department": "Réunion",
            "score": round(float(similarities[i]), 3)
        })

    if not context_parts:
        return jsonify({
            "answer": f"Je ne trouve pas d'intervention spécifique de {filter_speaker} à ce sujet.",
            "sources": []
        })

    context = "\n\n".join(context_parts)

    # --- D. Génération de la réponse (Gemini) ---
    try:
        prompt = f"CONTEXTE DE LA RÉUNION :\n{context}\n\nQUESTION DE L'UTILISATEUR : {question}"
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2 
            )
        )
        answer = response.text
        
    except Exception as e:
        return jsonify({"error": f"Erreur Gemini : {str(e)}"}), 500

    return jsonify({
        "answer": answer, 
        "sources": sources
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=DEBUG)