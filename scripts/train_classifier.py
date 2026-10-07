"""
Based on: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
          https://scikit-learn.org/stable/modules/compose.html#featureunion-composite-feature-spaces
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "backend" / "models" / "intent_classifier.joblib"
RESULTS = ROOT / "results"


def main():
    train = pd.read_csv(DATA / "train.csv").fillna("")
    val = pd.read_csv(DATA / "val.csv").fillna("")
    test = pd.read_csv(DATA / "test.csv").fillna("")

    model = Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2))),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=5)),
    ])
    model.fit(train["clean_text"], train["label"])

    lines = []
    for name, part in [("validation", val), ("test", test)]:
        pred = model.predict(part["clean_text"])
        acc = accuracy_score(part["label"], pred)
        lines.append(f"== {name}: accuracy {acc:.1%} ({len(part)} samples)")
        lines.append(classification_report(part["label"], pred, zero_division=0))
        wrong = part[pred != part["label"]]
        for (_, row), p in zip(wrong.iterrows(), pred[pred != part["label"]]):
            lines.append(f"   wrong: {row['text']!r}  true={row['label']}  predicted={p}")
    report = "\n".join(lines)
    print(report)

    # Retrain on train+val for the final model used by the app (test stays unseen).
    full = pd.concat([train, val])
    model.fit(full["clean_text"], full["label"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "classifier_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
