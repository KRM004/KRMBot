import re

def load_file(path: str) -> str:
    """Charge un fichier texte"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def preprocess_text(text: str) -> str:
    """
    Nettoyage optimisé pour RAG sur transcription de réunion.
    """
    # 1. Normaliser les espaces autour des deux-points, SAUF si c'est une heure (ex: 14:30)
    # On utilise un "lookbehind" et "lookahead" pour éviter les chiffres
    text = re.sub(r"(?<!\d)\s*:\s*(?!\d)", " : ", text)

    # 2. Supprimer les caractères bizarres, mais on GARDE les sauts de ligne (\n)
    # \w inclut déjà les lettres accentuées en Python 3. On garde la ponctuation standard.
    text = re.sub(r"[^\w\s.,!?:\-'\"]", " ", text)

    # 3. Normaliser les espaces multiples, SANS toucher aux sauts de ligne
    # On cible uniquement les espaces " " et les tabulations "\t"
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Supprimer les sauts de ligne multiples (plus de 2 deviennent 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_by_speaker(text: str) -> list[dict]:
    """
    Découpe la transcription par locuteur.
    Gère les prénoms composés, les numéros (Intervenant 1), etc.
    """
    # NOUVELLE REGEX : 
    # ^\s* : Début de ligne (optionnel avec des espaces)
    # ([\w\s.-]+): Capture le nom du locuteur (lettres, espaces, points, tirets)
    # \s*:       : Suivi par des deux-points
    # re.MULTILINE permet au ^ de matcher le début de chaque ligne, pas juste le début du fichier
    pattern = r"^\s*([\w\s.-]+)\s*:"
    
    # re.split va alterner : [texte_avant, locuteur1, texte1, locuteur2, texte2...]
    parts = re.split(pattern, text, flags=re.MULTILINE)

    segments = []
    
    # Si le texte ne commence pas directement par un locuteur (ex: introduction)
    if parts[0].strip():
        segments.append({"speaker": "Inconnu/Contexte", "text": parts[0].strip()})

    # Parcourir les locuteurs et leurs textes
    for i in range(1, len(parts) - 1, 2):
        speaker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        
        if content:
            segments.append({
                "speaker": speaker, 
                "text": f"{speaker} : {content}"
            })

    return segments