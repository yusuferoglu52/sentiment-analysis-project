import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["review"] = df["review"].astype(str)
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df


def plot_confusion_matrix(cm, model_name, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Negative", "Positive"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix - {model_name}",
        ylabel="True label",
        xlabel="Predicted label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test, raw_test_texts):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    errors = []
    misclassified_idx = np.where(y_test.to_numpy() != y_pred)[0]

    for idx in misclassified_idx:
        errors.append({
            "text": raw_test_texts.iloc[idx],
            "true_label": int(y_test.iloc[idx]),
            "pred_label": int(y_pred[idx])
        })

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "errors": errors,
        "predictions": y_pred
    }


def save_errors(errors, output_path, limit=30):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, err in enumerate(errors[:limit], start=1):
            f.write(f"Example {i}\n")
            f.write(f"True Label: {err['true_label']}\n")
            f.write(f"Predicted Label: {err['pred_label']}\n")
            f.write(f"Text: {err['text']}\n")
            f.write("-" * 80 + "\n")


def save_disagreements(raw_texts, y_true, pred_a, pred_b, output_path, limit=30):
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(raw_texts)):
            if pred_a[i] != pred_b[i]:
                f.write(f"Example {count + 1}\n")
                f.write(f"True Label: {int(y_true.iloc[i])}\n")
                f.write(f"Logistic Prediction: {int(pred_a[i])}\n")
                f.write(f"SVM Prediction: {int(pred_b[i])}\n")
                f.write(f"Text: {raw_texts.iloc[i]}\n")
                f.write("-" * 80 + "\n")
                count += 1
                if count >= limit:
                    break


def main():
    data_path = "data/raw/IMDB Dataset.csv"
    report_path = "outputs/reports/model_comparison.txt"

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    print("Veri yükleniyor...")
    df = load_data(data_path)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/imdb_clean.csv", index=False, encoding="utf-8")
    print("Temizlenmiş veri kaydedildi: data/processed/imdb_clean.csv")

    X = df["clean_review"]
    y = df["label"]
    raw_text = df["review"]

    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X,
    y,
    raw_text,
    test_size=0.2,
    random_state=42,
    stratify=y
)

    # 🔥 processed klasörünü oluştur
    os.makedirs("data/processed", exist_ok=True)

    # 🔥 train dataframe
    train_df = pd.DataFrame({
        "review": raw_train.values,
        "clean_review": X_train.values,
        "label": y_train.values
    })

    # 🔥 test dataframe
    test_df = pd.DataFrame({
        "review": raw_test.values,
        "clean_review": X_test.values,
        "label": y_test.values
    })

    # 🔥 kaydet
    train_df.to_csv("data/processed/train.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/processed/test.csv", index=False, encoding="utf-8")

    print("Train verisi kaydedildi: data/processed/train.csv")
    print("Test verisi kaydedildi: data/processed/test.csv")

    

    logreg_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=2.0
        ))
    ])

    svm_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(C=1.0))
    ])

    results = []
    results.append(evaluate_model(
        "Logistic Regression",
        logreg_model,
        X_train, X_test, y_train, y_test, raw_test
    ))

    results.append(evaluate_model(
        "Linear SVM",
        svm_model,
        X_train, X_test, y_train, y_test, raw_test
    ))

    best_result = max(results, key=lambda x: x["accuracy"])

    for result in results:
        model_name = result["name"].lower().replace(" ", "_")
        model_path = f"models/{model_name}.joblib"
        joblib.dump(result["model"], model_path)
        print(f"Model kaydedildi: {model_path}")

        cm_path = f"outputs/figures/{model_name}_confusion_matrix.png"
        plot_confusion_matrix(result["confusion_matrix"], result["name"], cm_path)
        print(f"Confusion matrix kaydedildi: {cm_path}")

        error_path = f"outputs/reports/{model_name}_errors.txt"
        save_errors(result["errors"], error_path, limit=30)
        print(f"Hatalı örnekler kaydedildi: {error_path}")

    save_disagreements(
        raw_test,
        y_test,
        results[0]["predictions"],
        results[1]["predictions"],
        "outputs/reports/model_disagreements.txt",
        limit=30
    )
    print("Model ayrışma raporu kaydedildi: outputs/reports/model_disagreements.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"{'=' * 20} {result['name']} {'=' * 20}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(result["report"])
            f.write("\nConfusion Matrix:\n")
            f.write(str(result["confusion_matrix"]))
            f.write("\n\n")

        f.write(f"Best Model: {best_result['name']} | Accuracy: {best_result['accuracy']:.4f}\n")

    print(f"\nEn iyi model: {best_result['name']} | Accuracy: {best_result['accuracy']:.4f}")
    print(f"Karşılaştırma raporu kaydedildi: {report_path}")


if __name__ == "__main__":
    main()