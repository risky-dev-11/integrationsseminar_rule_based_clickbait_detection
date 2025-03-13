from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import sys
import pandas as pd
import re
import nltk
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import importlib.util

# Pfad-Konfiguration
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
NEURAL_MODEL_PATH = BASE_DIR / "model" / "clickbait_detection_model.h5"
RULE_BASED_MODEL_PATH = BASE_DIR / "model" / "rule_based_classifier.pkl"

# NLTK-Ressourcen herunterladen
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# FastAPI-App erstellen
app = FastAPI(title="Clickbait Detection API")

# Statische Dateien (Frontend) einbinden
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# POS-Tags Dictionary definieren
pos_tags_dict = {
    'CC': 'coordinating conjunction',
    'CD': 'cardinal digit',
    'DT': 'determiner',
    'EX': 'existential there',
    'FW': 'foreign word',
    'IN': 'preposition/subordinating conjunction',
    'JJ': 'adjective',
    'JJR': 'adjective, comparative',
    'JJS': 'adjective, superlative',
    'LS': 'list marker',
    'MD': 'modal',
    'NN': 'noun, singular',
    'NNS': 'noun plural',
    'NNP': 'proper noun, singular',
    'NNPS': 'proper noun, plural',
    'PDT': 'predeterminer',
    'POS': 'possessive ending',
    'PRP': 'personal pronoun',
    'PRP$': 'possessive pronoun',
    'RB': 'adverb',
    'RBR': 'adverb, comparative',
    'RBS': 'adverb, superlative',
    'RP': 'particle',
    'TO': 'to',
    'UH': 'interjection',
    'VB': 'verb, base form',
    'VBD': 'verb, past tense',
    'VBG': 'verb, gerund/present participle',
    'VBN': 'verb, past participle',
    'VBP': 'verb, sing. present, non-3d',
    'VBZ': 'verb, 3rd person sing. present',
    'WDT': 'wh-determiner',
    'WP': 'wh-pronoun',
    'WP$': 'possessive wh-pronoun',
    'WRB': 'wh-adverb'
}

# Globale Variablen für die Modelle
neural_model = None
rule_based_model = None
tokenizer = None
max_words = 10000
scaler = None
padded_shape = None

# Regelbasierte Klassifikatorfunktion
def rule_based_clickbait_classifier(NumberStart, PRP, NoNumber, DT, NN):
    """Die regelbasierte Klassifikatorfunktion aus dem Jupyter Notebook."""
    if not PRP:  # personal pronoun == False
        if not NumberStart:  # == False
            if not DT:  # determiner == False
                return 0
            else:  # determiner == True
                if not NN:  # noun, singular == False
                    return 1
                else:  # noun, singular == True
                    return 0
        else:  # numberStart == True
            return 1
    else:  # personal pronoun == True
        return 1

# Funktion zum Laden oder Erstellen des neuronalen Modells
def load_or_create_neural_model():
    """Lädt das neuronale Modell oder erstellt es, falls es nicht existiert."""
    global tokenizer, scaler, padded_shape
    
    if os.path.exists(NEURAL_MODEL_PATH):
        try:
            print(f"Versuche neuronales Modell zu laden von: {NEURAL_MODEL_PATH}")
            model = load_model(NEURAL_MODEL_PATH)
            
            # Lade auch Tokenizer und Scaler
            tokenizer_path = NEURAL_MODEL_PATH.parent / 'tokenizer.pkl'
            scaler_path = NEURAL_MODEL_PATH.parent / 'scaler.pkl'
            
            if os.path.exists(tokenizer_path) and os.path.exists(scaler_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                    
                print("Neuronales Modell und Hilfskomponenten erfolgreich geladen")
                return model
            else:
                print("Warnung: Tokenizer oder Scaler nicht gefunden")
                
        except Exception as e:
            print(f"Fehler beim Laden des neuronalen Modells: {str(e)}")
    
    print("Neuronales Modell nicht gefunden. Erstelle neues Modell...")
    
    # Laden des Moduls
    spec = importlib.util.spec_from_file_location(
        "neural_classifier_clickbait_detection",
        BASE_DIR / "classifier" / "neural_classifier_clickbait_detection.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # Das Modul enthält bereits den Code zum Erstellen und Trainieren des Modells
    model = module.build_and_save_neural_model(BASE_DIR)
    
    # Die globalen Variablen für die Vorverarbeitung aktualisieren
    if hasattr(module, 'tokenizer'):
        tokenizer = module.tokenizer
    
    if hasattr(module, 'scaler'):
        scaler = module.scaler
    
    return model

# Funktion zum Laden oder Erstellen des regelbasierten Modells
def load_or_create_rule_based_model():
    """Lädt das regelbasierte Modell oder erstellt es, falls es nicht existiert."""
    if os.path.exists(RULE_BASED_MODEL_PATH):
        try:
            with open(RULE_BASED_MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Fehler beim Laden des regelbasierten Modells: {str(e)}")
            return None
    else:
        print("Regelbasiertes Modell nicht gefunden. Erstelle neues Modell...")
        # Laden des Moduls
        spec = importlib.util.spec_from_file_location(
            "rule_based_clickbait_classifier",
            BASE_DIR / "classifier" / "rule_based_clickbait_classifier.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Erstellen und Trainieren des Modells
        return module.build_and_save_rule_based_model()

# Laden der Modelle und der Vorverarbeitungskomponenten beim Start
@app.on_event("startup")
async def load_clickbait_models():
    global neural_model, rule_based_model, tokenizer, scaler, padded_shape
    
    print("Lade Clickbait-Erkennungsmodelle...")
    
    # Neuronales Modell laden
    neural_model = load_or_create_neural_model()
    if neural_model:
        print("Neurales Modell erfolgreich geladen")
    else:
        print("WARNUNG: Neuronales Modell konnte nicht geladen werden")
    
    # Regelbasiertes Modell laden oder erstellen
    rule_based_model = load_or_create_rule_based_model()
    if rule_based_model:
        print("Regelbasiertes Modell erfolgreich geladen oder erstellt")
    else:
        print("WARNUNG: Regelbasiertes Modell konnte nicht geladen oder erstellt werden")

    # Prüfen ob mindestens ein Modell geladen wurde
    if not neural_model and not rule_based_model:
        print("FEHLER: Keines der Modelle konnte geladen werden!")
        sys.exit(1)
    
    # Dataset laden für Tokenizer und Scaler Training (wird für neuronales Modell benötigt)
    if neural_model:
        dataset_path = BASE_DIR / "classifier" / "clickbait_data.csv"
        if not os.path.exists(dataset_path):
            print(f"FEHLER: Dataset nicht gefunden unter: {dataset_path}")
            sys.exit(1)
        
        dataset = pd.read_csv(dataset_path)
        
        # Tokenizer erstellen und auf Daten fitten
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(dataset['headline'])
        
        # TextSequenzen erstellen um die Form zu bekommen
        sequences = tokenizer.texts_to_sequences(dataset['headline'])
        padded = pad_sequences(sequences)
        padded_shape = padded.shape[1]
        
        # Feature-Spalten hinzufügen
        dataset[['NoNumber', 'NumberStart', 'NumberMiddle']] = dataset['headline'].apply(add_number_columns)
        
        # POS-Tags hinzufügen
        tokens = dataset['headline'].apply(nltk.word_tokenize)
        pos_tags = tokens.apply(lambda tokens: [tag for word, tag in nltk.pos_tag(tokens)])
        
        # POS-Tags Spalten erstellen
        pos_columns = pd.DataFrame(index=dataset.index)
        for tag, description in pos_tags_dict.items():
            pos_columns[description] = False
        
        for i, tags in enumerate(pos_tags):
            for tag in tags:
                if tag in pos_tags_dict:
                    pos_columns.at[i, pos_tags_dict[tag]] = True
        
        # Dataset erweitern
        dataset = pd.concat([dataset, pos_columns], axis=1)
        
        # Wichtige Features auswählen
        selected_features = ['personal pronoun', 'NumberStart', 'noun, singular', 
                            'determiner', 'NoNumber', 'cardinal digit']
        
        # Scaler initialisieren
        scaler = StandardScaler()
        scaler.fit(dataset[selected_features].astype(float))
        
        print("Vorverarbeitungskomponenten erfolgreich initialisiert")

def add_number_columns(headline):
    """Fügt Spalten für Zahlen in der Überschrift hinzu."""
    no_number = not bool(re.search(r'\d+', headline))
    number_start = bool(re.match(r'^\d+', headline))
    number_middle = bool(re.search(r'\d+', headline) and not number_start)
    
    return pd.Series([no_number, number_start, number_middle])

def predict_clickbait_neural(headline):
    """Vorhersage ob eine Überschrift Clickbait ist mit neuronalem Netz."""
    global neural_model, tokenizer, scaler, padded_shape
    
    if not neural_model or not tokenizer or not scaler:
        return {"error": "Neuronales Modell wurde noch nicht geladen."}
    
    try:
        # Tokenize and POS tag the input headline
        tokens = nltk.word_tokenize(headline)
        pos_tags = [tag for word, tag in nltk.pos_tag(tokens)]
        
        # Create a DataFrame with the same structure as the dataset
        test_data = pd.DataFrame(index=[0])
        test_data['headline'] = headline
        
        # Add columns for the selected POS tags and initialize them to False
        selected_tags = ['personal pronoun', 'NumberStart', 'noun, singular', 
                        'determiner', 'NoNumber', 'cardinal digit']
        for tag in selected_tags:
            test_data[tag] = False
        
        # Check if the headline starts with a number and set NumberStart to True if it does
        if re.match(r'^\d+', headline):
            test_data.at[0, 'NumberStart'] = True
        
        # Check if the headline does not contain any numbers and set NoNumber to True if it doesn't
        if not bool(re.search(r'\d+', headline)):
            test_data.at[0, 'NoNumber'] = True
        
        # Set the corresponding POS tag columns to True if the tag is present in the headline
        for tag in pos_tags:
            if tag in pos_tags_dict and pos_tags_dict[tag] in selected_tags:
                test_data.at[0, pos_tags_dict[tag]] = True
        
        # Tokenize and pad the input headline
        sequence = tokenizer.texts_to_sequences([headline])
        padded_sequence = pad_sequences(sequence, maxlen=padded_shape)
        
        # Prepare additional features
        extra_features = test_data[selected_tags].astype(float)
        extra_features = scaler.transform(extra_features)
        
        # Predict the probability of clickbait
        prediction = neural_model.predict([padded_sequence, extra_features])
        prediction_value = float(prediction[0][0])
        
        return {
            "headline": headline,
            "probability": prediction_value,
            "is_clickbait": bool(prediction_value > 0.5),
            "model": "neural"
        }
    except Exception as e:
        print(f"Fehler bei der neuronalen Vorhersage: {str(e)}")
        return {"error": str(e)}

def predict_clickbait_rule_based(headline):
    """Vorhersage ob eine Überschrift Clickbait ist basierend auf regelbasierten Klassifikator."""
    if not rule_based_model:
        return {"error": "Regelbasiertes Modell wurde noch nicht geladen."}
    
    try:
        # Parse und Vorverarbeitung der Eingabe
        tokens = nltk.word_tokenize(headline)
        pos_tags = [tag for word, tag in nltk.pos_tag(tokens)]
        
        # Extrahiere Features
        number_start = bool(re.match(r'^\d+', headline))
        no_number = not bool(re.search(r'\d+', headline))
        
        # Initialisiere POS tag Variablen
        has_personal_pronoun = False
        has_determiner = False
        has_noun_singular = False
        
        # Setze entsprechende POS tag Variablen
        for tag in pos_tags:
            if tag == 'PRP':
                has_personal_pronoun = True
            elif tag == 'DT':
                has_determiner = True
            elif tag == 'NN':
                has_noun_singular = True
        
        # Wende regelbasierte Klassifikation an
        result = rule_based_clickbait_classifier(number_start, has_personal_pronoun, 
                                              no_number, has_determiner, 
                                              has_noun_singular)
        
        is_clickbait = bool(result)
        
        # Da dieser Klassifikator keine Wahrscheinlichkeit gibt, setzen wir 0.1 für nicht-Clickbait und 0.9 für Clickbait
        probability = 0.9 if is_clickbait else 0.1
        
        return {
            "headline": headline,
            "probability": probability,
            "is_clickbait": is_clickbait,
            "model": "rule_based"
        }
    except Exception as e:
        print(f"Fehler bei der regelbasierten Vorhersage: {str(e)}")
        return {"error": str(e)}

# Hauptseite - leitet zum Frontend weiter
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/frontend/index.html">
        <title>Weiterleitung...</title>
    </head>
    <body>
        <p>Weiterleitung zum Frontend...</p>
    </body>
    </html>
    """

# API-Endpunkt für die Clickbait-Erkennung
@app.post("/predict")
async def predict(message: str = Form(...), model: str = Form("neural")):
    if model == "neural":
        result = predict_clickbait_neural(message)
    else:
        result = predict_clickbait_rule_based(message)
    return JSONResponse(content=result)

# Server starten, wenn direkt ausgeführt
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)