from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle
import os

from pretraitement.cleaning import load_file, preprocess_text, split_by_speaker
from RAG.config import *

# =========================================
# 📁 Chemin robuste
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "reunion_2026.txt")

# =========================================
# 📥 Load + Clean
# =========================================
text = load_file(file_path)

# Nettoyage qui préserve les noms de locuteurs et les majuscules
clean_text = preprocess_text(text)

# =========================================
# ✂️ Chunking amélioré (ZÉRO PHRASE COUPÉE)
# =========================================

# ✅ CORRECTION : is_separator_regex=True et lookbehinds (?<=...)
# Cela garantit que la ponctuation est CONSERVÉE à la fin de la phrase
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        r"\n\n",          # Tour de parole (double saut)
        r"\n",            # Tour de parole simple
        r"(?<=\. )",      # Coupe APRÈS un point
        r"(?<=\? )",      # Coupe APRÈS un point d'interrogation
        r"(?<=! )",       # Coupe APRÈS un point d'exclamation
        r"(?<=, )",       # Coupe après une virgule si chunk trop long
        " ",
        ""
    ],
    length_function=len,
    is_separator_regex=True, # Indispensable !
)

chunks = splitter.split_text(clean_text)

# =========================================
# 🗣️ Enrichissement avec les locuteurs
# =========================================
enriched_chunks = []
speaker_segments = split_by_speaker(text)

for chunk in chunks:
    best_speaker = None
    best_count = 0
    chunk_lower = chunk.lower()
    
    for seg in speaker_segments:
        count = chunk_lower.count(seg["speaker"].lower())
        if count > best_count:
            best_count = count
            best_speaker = seg["speaker"]

    if best_speaker and best_count > 0:
        enriched_chunks.append(f"[Locuteur principal : {best_speaker}]\n{chunk.strip()}")
    else:
        enriched_chunks.append(chunk.strip())

print(f"✅ Nombre de chunks créés : {len(enriched_chunks)}")

if DEBUG:
    for i, c in enumerate(enriched_chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{c[:200]}")

# =========================================
# 🧠 Embedding
# =========================================

print("\n⏳ Génération des embeddings en cours...")

# On utilise le modèle MPNet défini dans config.py
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

embeddings = model.encode(
    enriched_chunks,
    show_progress_bar=True,
    batch_size=16,                # CPU-safe pour votre i7
    normalize_embeddings=True,    # Optimisation vitale pour le Cosine Similarity
)

# =========================================
# 💾 Save
# =========================================
data_path = os.path.join(BASE_DIR, DATA_PATH)

with open(data_path, "wb") as f:
    pickle.dump({
        "chunks": enriched_chunks,
        "embeddings": embeddings
    }, f)

print(f"✅ Index sauvegardé avec succès dans : {data_path}")