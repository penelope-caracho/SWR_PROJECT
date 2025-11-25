import re
import spacy
from spacy_sentiws import spaCySentiWS
from config import SENTIWS_PATH

# NLP Setup / Pipeline mit SpaCy und SentiWS
nlp = spacy.load("de_core_news_md")
nlp.add_pipe("sentiws", config={"sentiws_path": str(SENTIWS_PATH)})

# Prüft, ob ein Satz "gültig" ist, bevor er zur Auswahl gestellt wird
def is_valid_sentence(satz):
    doc = nlp(satz)

    # Leerzeichenfehler vom Modell bei unbekannteren Wörtern/Namen aussortieren / BSP: "doofeJulio"
    if re.search(r"[a-zäöüß][A-ZÄÖÜ]", satz):
        return False

    # Adjektiv-Nomen Paar finden / prüfen
    adj_noun_pairs = [
        (t, t.head) for t in doc 
        if t.pos_ == "ADJ" and (t.head.pos_ == "NOUN" or t.head.pos_ == "PROPN")
    ]
    if not adj_noun_pairs:
        return False

    # Artikel-Nomen-Kongruenz prüfen
    for token in doc:
        if token.pos_ == "DET" and (token.head.pos_ == "NOUN" or token.head.pos_ == "PROPN"):
            if token.morph.get("Gender") != token.head.morph.get("Gender"):
                return False
            #print("Artikel:", token.text, token.morph.get("Gender"))
            #print("Nomen:", token.head.text, token.head.morph.get("Gender"))

    return True

