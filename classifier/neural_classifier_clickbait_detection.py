import pandas as pd
import nltk
import numpy as np
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Input, Concatenate, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Model, load_model
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import re
import os.path
from pathlib import Path

# Replace 'your_clickbait_dataset.csv' with the path to your dataset
current_dir = Path(__file__).parent
dataset = pd.read_csv(current_dir / 'clickbait_data.csv')

# Turn the clickbait column into a boolean column
dataset['clickbait'] = dataset['clickbait'].astype(bool)

# Display information about the dataset
print(dataset.head())

# %%

def add_number_columns(headline: str) -> pd.Series:
    # Add columns for headlines with no numbers, numbers at the start, and numbers in the middle
    no_number = not bool(re.search(r'\d+', headline))
    number_start = bool(re.match(r'^\d+', headline))
    number_middle = bool(re.search(r'\d+', headline) and not number_start)
    
    return pd.Series([no_number, number_start, number_middle])

# Apply the add_number_columns function to each headline in the dataset
dataset[['NoNumber', 'NumberStart', 'NumberMiddle']] = dataset['headline'].apply(add_number_columns)

# %%
def add_special_character_columns(headline: str) -> pd.Series:
    """
    Add columns for special characters '-', '=', "'", and '.'.
    These characters were chosen because they are the top 4 special characters
    found in clickbait and non-clickbait headlines, as shown in the plot above.
    """
    has_minus = '-' in headline
    has_equals = '=' in headline
    has_apostrophe = "'" in headline
    has_period = '.' in headline
    
    return pd.Series([has_minus, has_equals, has_apostrophe, has_period])

# Apply the add_special_character_columns function to each headline in the dataset
dataset[['HasMinus', 'HasEquals', 'HasApostrophe', 'HasPeriod']] = dataset['headline'].apply(add_special_character_columns)

# %%

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

# Tokenize and POS tag each headline
tokens = dataset['headline'].apply(nltk.word_tokenize)
pos_tags = tokens.apply(lambda tokens: [tag for word, tag in nltk.pos_tag(tokens)])

# Define the POS tags and their descriptions
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

# Create a DataFrame to store the POS tag columns
pos_columns = pd.DataFrame(index=dataset.index)

# Add columns for each POS tag and initialize them to False
for tag, description in pos_tags_dict.items():
    pos_columns[description] = False

# Set the corresponding POS tag columns to True for each headline
for i, tags in enumerate(pos_tags):
    for tag in tags:
        if tag in pos_tags_dict:
            pos_columns.at[i, pos_tags_dict[tag]] = True

# Add the POS tag columns to the original dataset
dataset = pd.concat([dataset, pos_columns], axis=1)

# %%

# Drop the 'headline' column before calculating the correlation matrix
correlation_matrix = dataset.drop(columns=['headline']).corr()

# Get the correlation of each feature with the 'clickbait' column
correlation_with_clickbait = correlation_matrix['clickbait'].abs().sort_values(ascending=False)

# Remove the 'clickbait' column from the list
correlation_with_clickbait = correlation_with_clickbait.drop('clickbait')

# Get the top 20 features with the highest correlation
top_20_features = correlation_with_clickbait.index[:6]

# Keep only the top 5 features + 'clickbait' and 'headline'
dataset = dataset[['headline', 'clickbait'] + list(correlation_with_clickbait.index[:6])]

# %%
dataset.head()

# %%

# Parameters
max_words = 10000
embedding_dim = 100

# Tokenizer erstellen und auf Daten fitten
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(dataset['headline'])

def build_and_save_neural_model(base_dir=None):
    """Baut, trainiert und speichert das neuronale Modell"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Stelle sicher, dass das Modellverzeichnis existiert
    model_dir = base_dir / 'model'
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'clickbait_detection_model.h5'
    
    print("Erstelle neues neuronales Modell...")
    
    # Textdaten vorbereiten
    sequences = tokenizer.texts_to_sequences(dataset['headline'])
    padded = pad_sequences(sequences)
    
    # Zusätzliche Features vorbereiten
    extra_features = dataset[['personal pronoun', 'NumberStart', 'noun, singular', 
                            'determiner', 'NoNumber', 'cardinal digit']].astype(float)
    
    # Normierung der numerischen Features
    scaler = StandardScaler()
    extra_features = scaler.fit_transform(extra_features)
    
    # **Modell mit zwei Eingängen definieren**
    # Input 1: Textdaten
    input_text = Input(shape=(padded.shape[1],), name="text_input")
    embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim)(input_text)
    conv_layer = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
    pooling_layer = GlobalMaxPooling1D()(conv_layer)
    lstm_layer = LSTM(128)(embedding_layer)
    
    # Input 2: Zusätzliche numerische Features
    input_extra = Input(shape=(extra_features.shape[1],), name="extra_input")
    extra_dense = Dense(32, activation='relu')(input_extra)
    
    # Zusammenführen beider Pfade
    concatenated = Concatenate()([pooling_layer, lstm_layer, extra_dense])
    dense_layer = Dense(64, activation='relu')(concatenated)
    dropout_layer = Dropout(0.3)(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(dropout_layer)
    
    # Modell erstellen
    model = Model(inputs=[input_text, input_extra], outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Modell trainieren
    model.fit([padded, extra_features], dataset['clickbait'], epochs=7, batch_size=32, validation_split=0.2)
    
    # Modell speichern
    model.save(model_path)
    print(f"Neuronales Modell gespeichert unter: {model_path}")
    
    # Speichere auch Tokenizer und Scaler
    with open(model_dir / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return model

# Wenn das Skript direkt ausgeführt wird, erstelle und speichere das Modell
if __name__ == "__main__":
    # Baue und speichere das Modell
    build_and_save_neural_model()

# %%
def predict_clickbait(headline):
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
    padded_sequence = pad_sequences(sequence, maxlen=padded.shape[1])

    # Prepare additional features
    extra_features = test_data[['personal pronoun', 'NumberStart', 'noun, singular', 
                                'determiner', 'NoNumber', 'cardinal digit']].astype(float)
    extra_features = scaler.transform(extra_features)

    # Predict the probability of clickbait
    prediction = model.predict([padded_sequence, extra_features])

    # Return the prediction as a percentage
    print(f'The model predicts a {prediction[0][0] * 100:.2f}% chance that the headline is clickbait.')

# # %%

# # Test the model with a sample clickbait headline

# predict_clickbait("You won't believe what happens next!")

# # %%

# # Ensure that the model and auxiliary files were created
# loading from:
# first try
# print('[INFO] Loading Dataset...')

# # Test the model with a sample clickbait headline
# predict_clickbait("You won't believe what happens next!")

# # Allow user to input a headline for evaluation
# user_headline = input("Please enter a headline to evaluate: ")
# predict_clickbait(user_headline)