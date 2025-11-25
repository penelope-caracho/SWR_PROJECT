
from wordfreq import zipf_frequency
from gensim.models import KeyedVectors
from config import FASTTEXT_PATH


ft = KeyedVectors.load(str(FASTTEXT_PATH), mmap="r")

UMLAUT_MAP = str.maketrans("äöü", "aou")


def deumlaut(s: str) -> str:
    """Wandelt ä/ö/ü in a/o/u um (ärzt -> arzt)."""
    return s.translate(UMLAUT_MAP)


def choose_masc_base(text_lower: str, lemma: str) -> str:
    """
    Versucht für eine feminine -in-Form (z.B. metzgerin, ärztin, kollegin)
    eine sinnvolle maskuline Basisform zu finden.
    """
    candidates = []

    # spaCy-Lemma, falls es sich unterscheidet
    if lemma and lemma != text_lower:
        candidates.append(lemma)

    # "-in" abschneiden
    base = text_lower[:-2]
    candidates.append(base)

    # Variante mit zusätzlichem 'e' (kollegin -> kollege)
    candidates.append(base + "e")

    # de-umlaut Varianten
    candidates.append(deumlaut(base))
    candidates.append(deumlaut(base + "e"))

    # Duplikate entfernen, sehr kurze Formen verwerfen
    unique = []
    for c in candidates:
        c = c.lower()
        if len(c) > 2 and c not in unique:
            unique.append(c)

    best = None
    best_score = -1.0

    for cand in unique:
        freq = zipf_frequency(cand, "de")
        in_ft = cand in ft
        score = (1 if in_ft else 0) + freq

        if score > best_score:
            best_score = score
            best = cand

    # nur zur Überprüfung:
    print(f"\n[Candidates-Test] Wort: {text_lower}")
    print(f"  Kandidaten: {unique}")
    print(f"  Bester Kandidat: {best} (Score={best_score:.2f})")

    # Wenn kein Kandidat sinnvoll erscheint, zur Originalform zurück
    return best if best_score > 0 else text_lower


