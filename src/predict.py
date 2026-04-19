import os
import joblib
from transformers import pipeline


def load_joblib_model(path: str):
    if not os.path.exists(path):
        print(f"Model bulunamadı: {path}")
        return None
    return joblib.load(path)


def load_bert_pipeline(model_dir: str):
    if not os.path.exists(model_dir):
        print(f"BERT modeli bulunamadı: {model_dir}")
        return None

    return pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir
    )


def format_label_binary(pred: int) -> str:
    return "Pozitif" if pred == 1 else "Negatif"


def format_label_bert(label: str) -> str:
    label = label.lower()
    if "positive" in label or label == "label_1":
        return "Pozitif"
    return "Negatif"


def main():
    logreg_path = "models/logistic_regression.joblib"
    svm_path = "models/linear_svm.joblib"
    bert_path = "models/bert_sentiment"

    logreg_model = load_joblib_model(logreg_path)
    svm_model = load_joblib_model(svm_path)
    bert_model = load_bert_pipeline(bert_path)

    if logreg_model is None and svm_model is None and bert_model is None:
        print("Hiçbir model bulunamadı. Önce modelleri eğit.")
        return

    print("Üç modelle duygu analizi tahmini")
    print("Çıkmak için q yaz.\n")

    while True:
        text = input("Review gir: ").strip()

        if text.lower() == "q":
            print("Çıkılıyor.")
            break

        if not text:
            print("Boş giriş yaptın.\n")
            continue

        print("\n=== TAHMİN SONUÇLARI ===")

        predictions = []

        # Logistic Regression
        if logreg_model is not None:
            logreg_pred = logreg_model.predict([text])[0]
            logreg_label = format_label_binary(logreg_pred)
            predictions.append(logreg_label)

            print(f"Logistic Regression: {logreg_label}")

            if hasattr(logreg_model, "predict_proba"):
                probs = logreg_model.predict_proba([text])[0]
                confidence = probs[logreg_pred]
                print(f"Logistic Regression Güven: {confidence:.4f}")

        # Linear SVM
        if svm_model is not None:
            svm_pred = svm_model.predict([text])[0]
            svm_label = format_label_binary(svm_pred)
            predictions.append(svm_label)

            print(f"Linear SVM: {svm_label}")

            if hasattr(svm_model, "decision_function"):
                decision_score = svm_model.decision_function([text])[0]
                print(f"Linear SVM Decision Score: {decision_score:.4f}")

        # BERT
        if bert_model is not None:
            bert_result = bert_model(text)[0]
            bert_label = format_label_bert(bert_result["label"])
            bert_score = bert_result["score"]
            predictions.append(bert_label)

            print(f"BERT: {bert_label}")
            print(f"BERT Güven: {bert_score:.4f}")

        # Final değerlendirme
        if predictions:
            positive_count = predictions.count("Pozitif")
            negative_count = predictions.count("Negatif")

            print("\n=== FİNAL DEĞERLENDİRME ===")

            if positive_count > negative_count:
                print("Ortak Sonuç: Pozitif")
            elif negative_count > positive_count:
                print("Ortak Sonuç: Negatif")
            else:
                print("Ortak Sonuç: Berabere / Belirsiz")

            unique_preds = set(predictions)
            if len(unique_preds) == 1:
                print("Durum: Tüm modeller aynı tahmini verdi.")
            else:
                print("Durum: Modeller arasında ayrışma var, tahmin dikkatle yorumlanmalı.")

        print()


if __name__ == "__main__":
    main()