import warnings
# Unterdrückt eine macOS-/LibreSSL-spezifische Import-Warnung von urllib3
# ("urllib3 v2 only supports OpenSSL 1.1.1+, currently compiled with LibreSSL …"),
# die für dieses Projekt funktional irrelevant ist.
warnings.filterwarnings("ignore", module="urllib3")

import requests
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from adjective_checker import check_adjective_list, find_prefix, find_vorsilbe
from is_valid_sentence import is_valid_sentence, nlp
from config import API_URL, ESPEAK_LIB_PATH, SENTIMENT_NEG_THRESHOLD, CFG

sentiment_override = CFG["sentiment"].get("override", {})

"""
Programm für die Evaluation von Ersatzadjektiven.

Dieses Programm kommuniziert mit dem separat laufenden Text-Generator
(FastAPI-Endpunkt `/generate`), lädt zu einem benutzerdefinierten
Themenwort mehrere generierte Satzvarianten und ermöglicht dem Benutzer
die Auswahl eines Satzes.

Im zweiten Schritt bewertet das Skript vorgeschlagene Ersatzadjektive
anhand mehrerer linguistischer und semantischer Kriterien:

- phonetischer Anlautvergleich (phonemizer)
- Wortartprüfung und syntaktischer Kontext (spaCy)
- Präfix- und Kompositionsregeln (adjective_checker.py)
- semantische Plausibilität durch FastText-Ähnlichkeit und Bigram-Frequenzen (adjective_checker.py)
- Polarität / Sentiment (SentiWS)

Das Programm fasst alle Prüfungen zusammen, gibt Analyse-Resultate
für jedes Ersatzwort aus und ermittelt die Gesamtpunktzahl.
"""

# explizite Bindung der espeak-ng-Library (macOS + Homebrew)
EspeakWrapper.set_library(ESPEAK_LIB_PATH)


print(""" 
######################################################################
#                    SWEAR-WORD-REPLACER Vers. 1.0                   #
######################################################################
#      Dieses Programm lädt zu einem vorgeschlagenen Themen-Wort     #
#     generierte Sätze mit einem Fluchwort-Adjektiv-Nomen-Paar und   #
#        fordert den Benutzer (nach Satzvarianten-Wahl) auf,         #
#      Ersatzadjektive vorzuschlagen, die z.B. mit dem gleichen      #
#              phonetischen Laut (Anlaut) beginnen.                  #
######################################################################      
""")

prompt = input("Gib ein Thema (Substantiv/Nomen) oder einen Namen ein und drücke ENTER: ")


satz = None  # Hier wird später der gewählte Satz gespeichert

# Anfrage an den Text-Generator (in Schleife, bis ein gültiger Satz gewählt wird)
while satz is None:

    response = requests.post(API_URL, json={"prompt": prompt})
    # Fehlerbehandlung Webserver
    if response.status_code != 200:
        print("Fehler bei Anfrage:", response.text)
        exit(1)

    generated_sentences = response.json()["sentences"]

    # Validitäts-Filter 
    valid_sentences = [s for s in generated_sentences if is_valid_sentence(s)]

    if not valid_sentences:
        print("Keine gültigen Sätze gefunden / neue Generierung…")
        continue

    #  Ausgabe der gültigen Sätze zur Auswahl
    print("\nWähle einen der generierten Sätze:\n")
    for i, s in enumerate(valid_sentences, start=1):
        print(f"{i}: {s}")
    print("0: Neue Sätze generieren")

    try:
        wahl = int(input("Gib die Nummer des Satzes ein und drücke ENTER: "))
        if wahl == 0:
            continue
        elif 1 <= wahl <= len(valid_sentences):
            satz = valid_sentences[wahl-1]
        else:
            print("Ungültige Auswahl, bitte erneut versuchen.")
    except ValueError:
        print("Bitte eine Zahl eingeben.")

print(f"\nGewählter Satz: {satz}")


# NLP Setup
doc1 = nlp(satz)


# Tauschwort-Adjektiv (+ Kopfnomen) finden
adj_token = None
for token in doc1:
    if token.pos_ == "ADJ" and (token.head.pos_ == "NOUN" or token.head.pos_ == "PROPN"):
        adj_token = token
        break

# Fall: nichts gefunden
if adj_token is None:
    raise ValueError("Kein Adjektiv-Nomen Paar im Satz gefunden!")

tausch_wort = adj_token.text

# Wortindex des Tauschworts im Satz finden
wort_index = [i for i, token in enumerate(doc1) if token.text == tausch_wort][0]

# Phonetische Umschrift des Tauschworts, in Variable speichern
ipa_wort = phonemize(tausch_wort, language='de', strip=True, backend="espeak")


# Printausgabe User-Infos
print("Tauschwort: " + tausch_wort)
print("Index des Tauschworts im Satz: " + str(wort_index))
print("Tauschwort in IPA:", ipa_wort)

print(f"""
Ersetze das Adjektiv '{tausch_wort}' durch möglichst viele andere mögliche Adjektive,
die an die Stelle des Wortes im Satz treten könnten. 

Die vollständigen Bewertungskriterien für die Adjektive sind:

- gleicher phonetischer Anlaut
- plausibel im Zusammenhang mit dem Kopfnomen
- positive bis neutrale Polarität (Sentiment-Wert)
- Präfix- und Kompositaregel: zusammengesetzte Adjektive mit dem selben Präfix dürfen nur einmal vorkommen
- bei Fehlern wird kein Punkt vergeben
- jedes korrekte Ersatzwort gibt 1 Punkt
- am Ende wird die Gesamtpunktzahl der gefunden Ersatzwörter ausgegeben
""")

# User-Eingabe der Ersatz-Adjektive
adjectives = input("\nGib deine Vorschläge für Adjektive durch Komma getrennt ein und drücke am Ende ENTER: ").split(", ")


benutzte_prefixe = set()   # speichert verwendete Präfixe

benutzte_vorsilben = set() # speichert verwendete "Vorsilben"

score = 0 # final score Ersatz-Adjektive


def pretty_print_result(result_dict):
    """
    für formatierte Ausgabe der Plausibilitätsprüfungsergebnisse
    """
    print("\n--- Plausibilitätsprüfung ---")
    for key, value in result_dict.items():
        print(f"{key:20}: {value}")
    print("-----------------------------\n")


# Überprüfung der Ersatz-Adjektive auf die Kriterien
for a in adjectives:
    wort = a.strip()
    # neu überspringt leere oder fehlerhafte Einträge
    if not wort:
        continue  
    # Satz mit ausgetauschtem Wort 
    neuer_satz = satz.replace(tausch_wort, wort)
    print()
    print(neuer_satz)
    doc2 = nlp(neuer_satz)
    # Fehlerzähler
    minus = 0
    # phonetische Umschrift des Ersatzworts
    adj_phon = phonemize(a.strip(), language='de', strip=True, backend="espeak")

    # sicherheitshalber prüfen, ob nicht gleiches Wort wie Tauschwort eingegeben wird^^
    if a == tausch_wort:
        print(f"Fehler: '{a}' ist das gleiche Wort wie das Tauschwort.")
        minus += 1
        
    # prüfen, ob der phonetische Anlaut gleich ist
    if adj_phon[0] != ipa_wort[0]:  
        print(f"Fehler: '{a}' hat einen anderen phonetischen Anfang ({adj_phon}).")
        minus += 1
    else:
        print(f"Korrekt: '{a}' hat den gleichen phonetischen Anfang ({adj_phon}).")

    # prüfen, ob das Ersatzwort ein Adjektiv ist
    if doc2[wort_index].pos_ != "ADJ":
        print(f"Fehler: '{a}' ist kein Adjektiv (POS={doc2[wort_index].pos_}).")
        minus += 1

    # Präfixregel (über Funktion aus adjective_checker.py)
    prefix = find_prefix(wort, nlp)
    if prefix in benutzte_prefixe and prefix is not None:
        print(f"Fehler: Das Präfix (oder Kompositum) '{prefix}' wurde schon verwendet.")
        minus += 1

    # Vorsilbenregel (über Funktion aus adjective_checker.py)
    vorsilbe = find_vorsilbe(wort)
    if vorsilbe in benutzte_vorsilben and vorsilbe is not None:
        print(f"Fehler: Der erste Teil des Kompositums wurde schon verwendet.")
        minus += 1

    # Plausibilitätsprüfung (über Funktion aus adjective_checker.py)
    results = check_adjective_list(doc1, [wort], adj_token)
    if results[0]["unkown word"]:
        pretty_print_result(results[0])
        print(f"'{wort}' ist ein unbekanntes Wort oder ein Tippfehler liegt vor (es kann leider nicht gewertet werden).")
        minus += 1
    elif not results[0]["plausible"]:
        pretty_print_result(results[0])
        print(f"Fehler: '{wort}' ist im Satzzusammenhang nicht plausibel.")
        minus += 1
    else:
        pretty_print_result(results[0])
        print(f"Korrekt: '{wort}' ist im Satzzusammenhang plausibel.")

    # Sentiment-Prüfung mit SentiWS
    tok = doc2[wort_index]
    token_sent = doc2[wort_index]._.sentiws

    # Lemma-basiertes Override: exemplarisch, hier für "klein"
    lemma_lower = tok.lemma_.lower()
    if lemma_lower in sentiment_override:
        token_sent = sentiment_override[lemma_lower]

    if token_sent is None:
        print(f"Achtung: Kein Sentiment-Wert für '{wort}' gefunden. Wird als neutral gewertet.")
    else:
        if token_sent < SENTIMENT_NEG_THRESHOLD:
            print(f"Fehler: '{wort}' hat eine negative Polarität (Score={token_sent}).")
            minus += 1
        else:
            print(f"Korrekt: '{wort}' wird als positiv / neutral gewertet (Score={token_sent}).")
    
    # Wertung
    if minus == 0:

        benutzte_prefixe.add(prefix)
        benutzte_vorsilben.add(vorsilbe)

        print(f"'{a}' ist ein zulässiger Ersatz für '{tausch_wort}'. 1 Punkt!")
        score += 1
    else:
        print(f"'{a}' ist kein zulässiger Ersatz für '{tausch_wort}'. {minus} Fehler = 0 Punkte.")

print()

# Endwertung
if score == 1:
    print("Wow, du hast ", score, " zulässiges Ersatzwort gefunden!")
else:
    print("Wow, du hast ", score, " zulässige Ersatzwörter gefunden!")
