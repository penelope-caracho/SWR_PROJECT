from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import (
    MODEL_DIR,
    GEN_MAX_LENGTH,
    GEN_NUM_RETURN_SEQUENCES,
    GEN_TOP_K,
    GEN_TOP_P,
    GEN_TEMPERATURE,
)

app = FastAPI()

# Modell und Tokenizer laden
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

# Sicherstellen, dass ein gültiges Padding-Token existiert
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

class PromptRequest(BaseModel):
    """Request-Body für /generate: enthält den Eingabe-Prompt."""
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    """Erzeugt kurze Textfortsetzungen zu einem gegebenen Prompt.

    1. Prompt tokenisieren
    2. Mehrere Varianten via Sampling generieren
    3. Generierte Texte bereinigen:
       - Entfernen des wiederholten Prompts (Echo-Filter)
       - Entfernen störender Präfixe wie führender '-'
       - Entfernen abgeschnittener Kleinschreib-Fragmente am Satzanfang
    4. Bereinigte Sätze als Liste zurückgeben

    Rückgabeformat:
        {"sentences": [...]} mit jeweils 4 generierten Varianten.
    """
    prompt = request.prompt

    # Prompt tokenisieren
    inputs = tokenizer(prompt, return_tensors="pt")

    # Text generieren (Sampling statt deterministisch)
    # outputs = model.generate(
    #     **inputs,
    #     max_length=18, # vorher 15
    #     num_return_sequences=4, 
    #     do_sample=True,
    #     top_k=50,
    #     top_p=0.9,
    #     temperature=0.6, # !! wenn 0.7 -> veränderte adj. (und ggf. kein swear-adj.) !!
    #     pad_token_id=tokenizer.pad_token_id
    # )
    outputs = model.generate(
        **inputs,
        max_length=GEN_MAX_LENGTH,
        num_return_sequences=GEN_NUM_RETURN_SEQUENCES,
        do_sample=True,
        top_k=GEN_TOP_K,
        top_p=GEN_TOP_P,
        temperature=GEN_TEMPERATURE,
        pad_token_id=tokenizer.pad_token_id
    )


    results = [] # Speicher für generierte Sätze

    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)

        # Prompt am Satzanfang abschneiden -> "Echoing" entfernen
        if text.startswith(prompt):
            text = text[len(prompt):].strip()   

        # Enfernt generierte Wortreste des Echoings nach Bindestrichwörtern
        if text.startswith("-"):
            parts = text.split()
            if parts[0].startswith("-"):
                parts = parts[1:]
            text = " ".join(parts)

        # Falls das erste „Wort“ mit '-' beginnt, entfernen
        if text and text[0].islower():
            parts = text.split()
            if parts:
                parts = parts[1:]
            text = " ".join(parts)

        # Rest bereinigen
        text = text.strip()

        results.append(text)

    return {"sentences": results}
