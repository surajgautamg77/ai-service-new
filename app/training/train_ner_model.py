import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
import json
import os
import shutil

def train_ner_model(file_path: str):
    MODEL_NAME = "xlm-roberta-base" # Or any other base model for NER
    MODEL_PATH = "app/models/entity_extraction/xlmr_ner"
    
    # Remove existing model directory if it exists
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Load CSV
    df = pd.read_csv(file_path)

    # Convert labels from string to list of strings
    df["labels"] = df["labels"].apply(lambda x: x.split())

    # Collect all unique labels to create a label map
    unique_labels = sorted(list(set(label for sublist in df["labels"] for label in sublist)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_labels = len(unique_labels)

    # Save label map
    with open(os.path.join(MODEL_PATH, "label_map.json"), "w") as f:
        json.dump(id_to_label, f)
    
    print("Detected NER labels:", unique_labels)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization and label alignment function
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"], truncation=True, is_split_into_words=False, padding="max_length", max_length=128
        )
        labels = []
        for i, label_list in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label_list[word_idx]])
                # For the other tokens in a word, we set the label to -100.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Train/test split
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_PATH, "results"),
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        # tokenizer=tokenizer, # For NER, tokenizer is crucial for Trainer
    )

    trainer.train()

    # Save trained model
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    print(f"✅ NER Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    # Example usage:
    # Make sure you have data/ner_train.csv with 'text' and 'labels' columns
    train_ner_model("data/ner_train.csv")
