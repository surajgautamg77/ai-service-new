import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
import json
import os
import shutil

def train_model(file_path: str):
    MODEL_NAME = "xlm-roberta-base"
    MODEL_PATH = "app/models/intent_recognition/xlmr"
    
    # Remove existing model directory if it exists
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    
    os.makedirs(MODEL_PATH, exist_ok=True)


    # Load CSV
    df = pd.read_csv(file_path)

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])
    num_labels = len(label_encoder.classes_)

    print("Detected intents:", list(label_encoder.classes_))

    # Save label map
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(MODEL_PATH, "label_map.json"), "w") as f:
        json.dump(label_map, f)

    # Convert to HF dataset
    dataset = Dataset.from_pandas(df[["text", "label"]])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Train/test split
    dataset = dataset.train_test_split(test_size=0.2)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
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
        load_best_model_at_end=False # Reverted to False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # Removed tokenizer=tokenizer as it causes TypeError
    )

    trainer.train()

    # Save trained model
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model("data/train.csv")
