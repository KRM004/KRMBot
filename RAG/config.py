import os
import sys
from dotenv import load_dotenv

# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

# =========================================
# 🧠 Modèles & Chemins
# =========================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
DATA_PATH       = os.getenv("DATA_PATH", "rag_data.pkl")

# =========================================
# 🌟 Gemini API
# =========================================
# On récupère la clé sans valeur par défaut en dur pour des raisons de sécurité
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

# Sécurité : On bloque l'exécution si la clé n'est pas trouvée
if not GEMINI_API_KEY:
    print("❌ ERREUR CRITIQUE : GEMINI_API_KEY est introuvable.")
    print("👉 Assure-toi d'avoir un fichier .env à la racine avec ta clé API.")
    sys.exit(1)

GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# =========================================
# ✂️ Paramètres de Découpage (Chunking)
# =========================================
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 200))

# =========================================
# 🔍 Paramètres de Recherche & Génération
# =========================================
TOP_K                = int(os.getenv("TOP_K", 4))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))

# =========================================
# ⚙️ Ressources Système
# =========================================
DEVICE      = os.getenv("DEVICE", "cpu")
LLM_THREADS = int(os.getenv("LLM_THREADS", 6))
DEBUG       = os.getenv("DEBUG", "False").lower() == "true"