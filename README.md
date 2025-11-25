# Swear-Word-Replacer 1.0

## Projektbeschreibung

Der **Swear-Word-Replacer** ist ein experimentelles textbasiertes Spiel, das sich humorvoll mit der Ersetzung beleidigender Adjektive auseinandersetzt. 
Ein feingetuntes deutsches GPT-2-Modell generiert Beispielsätze mit Fluchwort- Adjektiv-Nomen-Paaren. Die SpielerInnen schlagen alternative, positive Adjektive vor, die anschließend automatisch bewertet werden – anhand von phonetischer Ähnlichkeit, semantischer Plausibilität, syntaktischem Kontext und Sentimentwerten. Das Projekt wurde im Rahmen einer universitären Arbeit entwickelt.


Dieses Projekt besteht aus zwei verbundenen Komponenten:

1. **Textgenerator (über FastAPI)**
   Erzeugt kurze Satzvarianten auf Basis eines feingetunten deutschen GPT-2-Modells.

2. **Evaluationsprogramm (swr-eval.py)**
   Bewertet Vorschläge für Ersatzadjektive anhand phonetischer, syntaktischer, semantischer und sentimentbasierter Kriterien.


---

Hinweis: Das Projekt wurde unter macOS entwickelt und getestet.  
Unter Windows kann die Nutzung von phonemizer/espeak-ng abweichen und erfordert ggf. zusätzliche Anpassungen.

## Voraussetzungen

- macOS oder Linux
- Python **3.10** oder höher (getestet mit 3.10)
- Installiertes `espeak-ng` (für phonemizer)

### macOS-Hinweis (wichtig)

Homebrew installiert die notwendige Bibliothek üblicherweise hier:

```
/opt/homebrew/lib/libespeak-ng.dylib
```

Dieser Pfad wird in der Datei **config.yaml** eingetragen.

---

## Installation

```
git clone <REPO>
cd swr_project

python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

### Externe Ressourcen (nicht enthalten)

Folgende Dateien dürfen aus Lizenz- oder Größen-Gründen **nicht im Repository** enthalten sein:

- **FastText-Vektoren (Meta)** – nicht redistributierbar  
- **SentiWS-Daten** – nicht redistributierbar (nur akademische Nutzung)  
- **Feingetuntes GPT-2-Modell** – sehr groß; muss lokal bereitgestellt werden

---

## Ordnerstruktur

```
swr_project/
│
├── swr_eval.py
│
├── model/
│   ├── text_gen.py
│   └── finetuned_model/               
│
├── data/
│   ├── fasttext/
│   └── SentiWS_v2.0/
│
├── adjective_checker.py
├── gender_utils.py                    
├── is_valid_sentence.py
│
├── config.yaml
├── config.py
├── requirements.txt
│
├── README.md
└── LICENSE

```

---

## Nutzung

### 1. FastAPI-Server starten

```
source env/bin/activate
uvicorn model.text_gen:app --host 127.0.0.1 --port 8001
```

### 2. Programm starten **in neuem Terminal**

```
source env/bin/activate
python swr_eval.py
```

---

## Konfiguration

Parameter werden in `config.yaml` verwaltet:

- Pfade zu FastText, SentiWS und Modell
- API-Einstellungen
- phonemizer-Library-Pfad
- Similarity- und Frequenz-Schwellen
- Sentiment-Overrides
- Generatorparameter

---

## Warnungen unter macOS

Die `urllib3` / LibreSSL-Warnung wird unterdrückt:

```python
import warnings
warnings.filterwarnings("ignore", module="urllib3")
```

---

## Lizenz

Dieses Projekt selbst hat **keine eigene Lizenz** (All Rights Reserved). Der Quellcode darf nicht ohne Erlaubnis wiederverwendet oder verteilt werden.

### Modelle

Das verwendete feingetunte GPT-2-Modell basiert auf:

- **kkirchheim/german-gpt2-medium** – MIT-Lizenz  
- **Atomic-Ai/AtomicGPT-3** – MIT-Lizenz  


## Evaluierung und Reproduzierbarkeit

Die Modellevaluation für die Text-Generierung befindet sich im Unterordner (nur auf Anfrage erhältlich). 
Dieser enthält: 

`data/evaluation_model/`:
- `evaluate_model.py` – generiert Sätze aus dem Modell und speichert sie als CSV  
- `deu_news_2024_30K` - Corpus aus dem Nomen als Prompts ausgewählt wurden
- `eval_ue_gs.py` – berechnet die Spiel- und Satzqualität  
- `negative_noun_extractor.py` – extrahiert Testnomen aus dem Korpus  
- `evaluation_results11ann.csv` – manuell annotierte Evaluationsdatei  

`data/filter_kv/`:
- `filterkv.py`

`data/finetuning_model/`:
- `train_model.py`
- `trainingsdaten.jsonl`


Diese Komponenten dienen ausschließlich der **Reproduzierbarkeit der Modellbewertung** und sind **nicht Teil des Spiels**.


### Externe Komponenten und Ressourcen

Das Projekt verwendet folgende Ressourcen:  

- **spaCy** und **spacy-sentiws** – linguistische Analyse und Sentiment-Annotation  
  [https://spacy.io/](https://spacy.io/)  
- **SentiWS** (Universität Leipzig) – deutsches Sentimentlexikon, eingebunden über *spacy-sentiws*  
  [https://wortschatz.uni-leipzig.de/en/download/sentiws](https://wortschatz.uni-leipzig.de/en/download/sentiws)  
- **FastText** (Meta) – vortrainierte Wortvektoren für semantische Ähnlichkeitsmessungen  
  [https://fasttext.cc/](https://fasttext.cc/)  
- **Transformers** (Hugging Face) – Fine-Tuning und Modellverwaltung  
  [https://huggingface.co/](https://huggingface.co/)  
- **FastAPI** – Bereitstellung des Textgenerators als lokaler Webserver  
  [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)  
- **wordfreq** – Berechnung lexikalischer Zipf-Frequenzen  
  [https://pypi.org/project/wordfreq/](https://pypi.org/project/wordfreq/)  
- **Gensim** – Zugriff auf FastText-Vektoren und Serialisierung  
  [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)  
- **phonemizer** – phonetische Umschrift (mit Backend **eSpeak NG**)  
  [https://github.com/bootphon/phonemizer](https://github.com/bootphon/phonemizer)
  [https://github.com/espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)  
- **NumPy** – numerische Berechnungen  
  [https://numpy.org/](https://numpy.org/)