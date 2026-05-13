import pickle
import sys
import os
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from RAG.config import *

class MeetingRAGAnalyzer:
    """
    Système RAG (Retrieval-Augmented Generation) de niveau production 
    pour l'analyse sémantique des transcriptions de réunions.
    """
    
    def __init__(self):
        print("⏳ Initialisation du système RAG d'analyse de réunions...")
        self.chunks, self.doc_embeddings = self._load_data(DATA_PATH)
        self.model_emb = self._init_embedding_model(EMBEDDING_MODEL, DEVICE)
        self.client = self._init_gemini_client(GEMINI_API_KEY)
        
        self.system_prompt = (
            "Tu es un expert en analyse de réunions d'entreprise. "
            "Réponds UNIQUEMENT en utilisant le contexte fourni. "
            "Sois précis, analytique et concis. Si l'information est absente, "
            "indique-le clairement sans inventer."
        )
        print(f"✅ Système prêt. ({len(self.chunks)} segments chargés)")

    def _load_data(self, path: str) -> Tuple[List[str], np.ndarray]:
        """Charge et normalise la base de connaissances vectorielle."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data["chunks"], normalize(data["embeddings"])
        except FileNotFoundError:
            print(f"❌ Erreur critique : Fichier de données {path} introuvable.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            sys.exit(1)

    def _init_embedding_model(self, model_name: str, device: str) -> SentenceTransformer:
        """Initialise le modèle d'encodage local."""
        return SentenceTransformer(model_name, device=device)

    def _init_gemini_client(self, api_key: str) -> genai.Client:
        """Configure le client LLM Google Gemini."""
        if not api_key:
            print("❌ Erreur critique : GEMINI_API_KEY manquante dans l'environnement.")
            sys.exit(1)
        return genai.Client(api_key=api_key)

    def mmr_rerank(self, query_emb: np.ndarray, candidate_indices: List[int], top_k: int, lambda_param: float = 0.6) -> List[int]:
        """
        Applique l'algorithme Maximal Marginal Relevance (MMR) pour garantir 
        la diversité sémantique des documents récupérés.
        """
        selected = []
        candidates = list(candidate_indices)

        while len(selected) < top_k and candidates:
            best_idx, best_score = None, -np.inf
            for idx in candidates:
                relevance = cosine_similarity(query_emb, [self.doc_embeddings[idx]])[0][0]
                
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        cosine_similarity([self.doc_embeddings[idx]], [self.doc_embeddings[s]])[0][0]
                        for s in selected
                    )
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, idx

            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)

        return selected

    def retrieve_context(self, query: str) -> str:
        """Récupère et filtre les segments de réunion les plus pertinents."""
        query_emb = self.model_emb.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_emb, self.doc_embeddings)[0]

        ranked_indices = np.argsort(similarities)[::-1]
        filtered_indices = [
            idx for idx in ranked_indices if similarities[idx] >= SIMILARITY_THRESHOLD
        ]

        # 🚀 FALLBACK INTELLIGENT (Le fix pour le bug des questions courtes)
        if not filtered_indices:
            filtered_indices = ranked_indices[:3].tolist() # On force les 3 meilleurs

        if DEBUG:
            print(f"\n[DEBUG] Top similarités : {[round(similarities[i], 3) for i in ranked_indices[:5]]}")

        top_indices = self.mmr_rerank(
            query_emb, 
            filtered_indices[:TOP_K * 3], 
            TOP_K
        )
        
        return "\n\n".join([self.chunks[i] for i in top_indices])

    def generate_answer(self, query: str, context: str) -> Optional[str]:
        """Génère la réponse finale via Gemini en utilisant le contexte strict."""
        try:
            prompt = f"CONTEXTE DE LA RÉUNION :\n{context}\n\nQUESTION : {query}"
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.1
                )
            )
            return response.text
        except Exception as e:
            print(f"❌ Erreur lors de la génération Gemini : {e}")
            return None

    def run_cli(self):
        """Lance l'interface en ligne de commande interactive."""
        print("-" * 60)
        print("🎙️ Assistant RAG de Réunion prêt ! (Tapez 'exit' pour quitter)")
        print("-" * 60)

        try:
            while True:
                query = input("\n❓ Posez votre question : ").strip()
                if query.lower() in ['q', 'quit', 'exit']:
                    print("👋 Fermeture de l'assistant. À bientôt !")
                    break
                if not query:
                    continue

                context = self.retrieve_context(query)
                
                if DEBUG:
                    print(f"\n[DEBUG] Contexte extrait :\n{context[:400]}...\n")

                answer = self.generate_answer(query, context)
                if answer:
                    print(f"\n🤖 Réponse :\n{answer}")

        except KeyboardInterrupt:
            # Gestion propre de l'interruption par l'utilisateur (Ctrl+C)
            print("\n\n⚠️ Interruption détectée. Fermeture propre du système.")
            sys.exit(0)


if __name__ == "__main__":
    # Instanciation et exécution
    analyzer = MeetingRAGAnalyzer()
    analyzer.run_cli()