import os
import numpy as np
import pandas as pd
import evaluate

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODEL_NAME = "google-bert/bert-base-uncased"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
    return df[["review", "label"]]


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=256,
    )


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }


def main():
    os.makedirs("models/bert_sentiment", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    data_path = "data/raw/IMDB Dataset.csv"
    df = load_data(data_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    training_args = TrainingArguments(
        output_dir="models/bert_sentiment/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    trainer.save_model("models/bert_sentiment")
    tokenizer.save_pretrained("models/bert_sentiment")

    with open("outputs/reports/bert_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    print("\nBERT modeli kaydedildi: models/bert_sentiment")
    print("Rapor kaydedildi: outputs/reports/bert_results.txt")


if __name__ == "__main__":
    main()