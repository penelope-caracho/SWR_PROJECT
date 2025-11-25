from gensim.models import KeyedVectors
from wordfreq import zipf_frequency
import numpy as np
from gender_utils import choose_masc_base
from config import (
    FASTTEXT_PATH,
    SIMILARITY_THRESHOLD,
    FREQ_MIN,
    FREQ_MIN_OOV_ADJ,
    FREQ_MIN_OOV_NOUN,
    FREQ_HARD_MIN,
)


# FastText Modell laden (KeyedVectors)
ft = KeyedVectors.load(str(FASTTEXT_PATH), mmap="r")


def get_noun_core(noun_token):
    """
    „Kern-“ Nomen extrahieren (für Bindestrichwörter gedacht), 
    damit die semantische Ähnlichkeit mit dem Adjektiv 
    besser gemessen werden kann. Bsp.: Java-Klausur → Klausur
    """
    text = noun_token.text
    if "-" in text:
        return text.split("-")[-1].lower()
    return text.lower()


def check_adjective_list(doc, replacements, adj_token, threshold=SIMILARITY_THRESHOLD):
    """
    Prüft eine Liste vorgeschlagener Ersatzadjektive hinsichtlich ihrer
    semantischen Plausibilität im gegebenen Satzkontext.

    Die Funktion bewertet jedes Ersatzadjektiv anhand folgender Kriterien:
      • Ähnlichkeit zum Kopfnomen (Similarity über FastText)
      • Wahrscheinlichkeit des Adjektiv-Nomen-Bigrams (Zipf-Frequenz)
      • Behandlung von Eigennamen mittels Platzhalter ("person")
      • Unterscheidung, ob Adjektiv oder Nomen im FastText-Vokabular vorkommen
      • Kombination aller Signale zu einer finalen Plausibilitätsentscheidung

    """
    noun_token = adj_token.head

    # Nomen(-Kern)
    orig_noun_core = get_noun_core(noun_token)

    # Variablen zur Abgeckung von Sonderfällen:
    noun_for_sim = orig_noun_core    # Ersatz-Variable für Platzhalter bei Eigennamen-Eingabe
    used_person_placeholder = False
    head_for_freq = orig_noun_core   # Variable für Freq. für Gender Equality bei Berufen (hinzugef. nach Oral Pres.)

    # Bei Eigennamen-Eingabe Platzhalter: "Person"
    if noun_token.pos_ == "PROPN":
        noun_for_sim = "person"   
        used_person_placeholder = True

    # für Gender Equality bei Berufen (hinzugef. nach Oral Pres.):
    # aktuell noch das Problem, dass wenn weiblicher Vorname, 
    # der auf -in endet und nicht als Eigenname erkannt wird, wird Endung abgeschnitten
    # Bsp: Shirin (-> Shire) (könnte durch Namens-Lexikon abgefangen werden)
    else:
        # Gender-Normalisierung nur für feminine -in-Formen
        gender = noun_token.morph.get("Gender")
        text_lower = orig_noun_core
        lemma = noun_token.lemma_.lower()

        if gender == ["Fem"] and text_lower.endswith("in") and len(text_lower) > 4:
            base = choose_masc_base(text_lower, lemma)
            noun_for_sim = base
            head_for_freq = base


    # speichert Plausibiltäts-Resultate
    results = []

    for new_adj in replacements:
        new_adj_lower = new_adj.lower()

        # Vokabular-Checks
        adj_in_vocab = new_adj_lower in ft
        noun_in_vocab = head_for_freq in ft

        # Similarity 
        similarity = None
        try:
            similarity = ft.similarity(new_adj_lower, noun_for_sim)
        except KeyError:
            similarity = None

        # Bigramm: Adj|Noun(core)
        bigram = f"{new_adj_lower} {head_for_freq}"
        freq_score = zipf_frequency(bigram, "de")

        # Plausibiltätsbedingungen
        plausible = (
            (similarity is not None and similarity > threshold and freq_score >= FREQ_MIN)
            or (freq_score > FREQ_MIN_OOV_ADJ and (not adj_in_vocab))
            or (freq_score > FREQ_MIN_OOV_NOUN and (not noun_in_vocab))
            or (freq_score > FREQ_HARD_MIN)
        )

        # Ergebnisse der Plausibilitätsprüfung speichern
        results.append({
            "replacement": new_adj,
            "(proper-)noun": noun_token.text,       
            "noun core": orig_noun_core,  
            "sim noun core": noun_for_sim,  
            "head_for_freq": head_for_freq, # (hinzugefügt nach Oral Pers.)         
            "similarity": similarity,
            "freq score": freq_score,
            "plausible": plausible,
            "adj in vocab": adj_in_vocab,
            "noun in vocab": noun_in_vocab,
            "person placeholder": used_person_placeholder,
            "unkown word": (similarity is None and freq_score == 0),
        })

    return results


ADJ_PREFIXES = [
    "un", "in", "ir", "im", "il", "be", "ab", "zu", "an", "ge", "an", "im", "um", "da", "dar",
    "zer", "ent", "all", "bei", "ab", "an", "be", "auf", "aus", "da", "de", "ge", "mit", "weg",
    "ein", "her", "hin", "los", "vor", "ver", "neo", "pro", "sub", "ur", "alt", "neu", "nah",
    "nicht", "kontra", "haupt", "nieder", "ober", "erz", "hoch", "un", "ur", "alt", "neu", "alt", "jung",
    "anti", "mega", "neo", "pro", "sub", "ur", "un", "in", "im", "ir", "miss", "il", "ver", 
    "miss", "fehl", "all", "bei", "hinter", "haupt", "knall", "gegen",
    "infra", "inter", "intra", "extra", "trans", "super", "hyper", "ultra", 
    "pseudo", "multi", "brand", "über", "unter", "vor", "ver", "hinter", 
    "gleich", "mehr", "semi", "quasi", "zurück", "gross", "groß",
    "klein", "weit", "nah", "dunkel", "hell", "kalt", "warm",  "weich", "hart", 
    "scharf", "mild", "stark", "brand", "hauch", "bild", "mini", "maxi"
]


def find_prefix(wort, nlp):
    """
    Soll, ein "produktives" Präfix in einem Adjektiv zu erkennen.

    - Lemmatisiert das Wort und prüft bekannte Präfixe.
    - Validiert den Reststamm über FastText-Vokabular oder Wortfrequenz.
    - Nutzt „ge“-Varianten als Fallback (z.B. verliebt → geliebt).

    """
    doc = nlp(wort)
    lemma = doc[0].lemma_.lower()

    for pref in ADJ_PREFIXES:
        if not lemma.startswith(pref):
            continue

        rest = lemma[len(pref):]
        if len(rest) <= 2:
            continue

        # --- 1. lemmata prüfen ---
        base_candidates = set()

        # rohe Restform
        base_candidates.add(rest)

        # Lemma des "Restes"
        rest_lemma = nlp(rest)[0].lemma_.lower()
        base_candidates.add(rest_lemma)

        # fallback, falls spacy lemma failt: 
        # häufige Adjektiv-Endungen kappen
        for suf in ("e", "en", "er", "es"):
            if rest.endswith(suf) and len(rest) - len(suf) > 2:
                base_candidates.add(rest[:-len(suf)])

        # zuerst nur diese "Basen" gegen Vokabular/Frequenz prüfen
        for cand in base_candidates:
            try:
                if cand in ft or zipf_frequency(cand, "de") > 2.0:
                    return pref
            except Exception:
                continue

        # --- 2. fallback: "ge"-Vorsetzen (falls Überpr. nicht erfolgreich) ---

        ge_candidates = set()

        # "ge"-Vorsetzen 
        for cand in base_candidates:
            if not cand.startswith("ge"):
                ge_candidates.add("ge" + cand)

        # Sonderfall: ver-gnügt -> genügt
        if rest.startswith("gn"):
            ge_candidates.add("ge" + rest)

        for cand in ge_candidates:
            try:
                if cand in ft or zipf_frequency(cand, "de") > 2.0:
                    return pref
            except Exception:
                continue

    return None


# Soll verhindern dass Komposita-Anfänge mehrfach verwendet werden
def find_vorsilbe(wort, laenge=6): # die ersten 6 Buchstaben
    return wort[:laenge].lower()