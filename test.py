import pandas as pd
import spacy
import random
from spacy.training.example import Example

def load_training_data(csv_file):
    df = pd.read_csv(csv_file)
    training_data = []

    for _, row in df.iterrows():
        text = row['Paragraph']
        entities = []

        for label, column in [('FEATURE', 'Features'), ('MATERIAL', 'Materials'), ('STYLE', 'Styles')]:
            if pd.notna(row[column]):
                for item in str(row[column]).split(','):
                    item = item.strip()
                    start = text.lower().find(item.lower())
                    if start != -1:
                        end = start + len(item)
                        entities.append((start, end, label))
        training_data.append((text, {"entities": entities}))
    return training_data

def train_model(train_data, output_dir='my_custom_model', n_iter=30):
    nlp = spacy.blank("en")  # start with a blank English model
    ner = nlp.add_pipe("ner")

    for _, annotations in train_data:
        for start, end, label in annotations["entities"]:
            ner.add_label(label)

    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.3, losses=losses)
        print(f"Epoch {i+1}: {losses}")

    nlp.to_disk(output_dir)
    print(f"✅ Model trained and saved to '{output_dir}'")

if __name__ == "__main__":
    csv_path = "Annotated_Monuments.csv"  # <-- Your CSV path
    train_data = load_training_data(csv_path)
    train_model(train_data)
