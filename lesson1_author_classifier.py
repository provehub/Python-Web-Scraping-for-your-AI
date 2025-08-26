# --- imports ---
import os
import pathlib
import joblib
import pandas as pd

from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- config ---
# Point to your Postgres DB (works with what you set in pgAdmin)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://quotes_user:quotes_pass@localhost:5432/quotes",
)

MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "author_clf.joblib"

MIN_QUOTE_LEN = 10        # filter super-short texts (noise)
MIN_SAMPLES_PER_CLASS = 5 # drop authors with too few examples to learn from

def load_data() -> pd.DataFrame:
    """Load quotes from Postgres into a DataFrame."""
    engine = create_engine(DATABASE_URL, future=True)
    with engine.connect() as con:
        # Basic sanity: ensure table exists and user has privileges
        # If this fails, recheck your GRANTs in pgAdmin.
        df = pd.read_sql(text("SELECT text, author FROM quotes"), con)
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning + class pruning so the baseline learns something."""
    df = df.dropna(subset=["text", "author"])
    # Normalize whitespace
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    # Remove super short quotes
    df = df[df["text"].str.len() >= MIN_QUOTE_LEN].copy()

    # Keep authors with at least N samples (simple way to avoid extreme imbalance)
    vc = df["author"].value_counts()
    keep_authors = set(vc[vc >= MIN_SAMPLES_PER_CLASS].index)
    df = df[df["author"].isin(keep_authors)].copy()

    # Shuffle (not strictly needed, but nice)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def build_pipeline() -> Pipeline:
    """Create a text → TF-IDF → LogisticRegression pipeline."""
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),      # unigrams + bigrams often help
                min_df=2,                # ignore very rare terms
                max_df=0.9,              # ignore overly common terms
                lowercase=True,
                strip_accents="unicode"
            )),
            ("clf", LogisticRegression(
                max_iter=200,            # ensure convergence
                n_jobs=None,             # scikit-learn handles parallelism internally
                class_weight="balanced"  # a simple guard against imbalance
            )),
        ]
    )
    return pipe

def main():
    print("Loading data from Postgres…")
    df = load_data()
    print(f"Raw rows: {len(df)}")

    df = clean_df(df)
    print(f"Rows after cleaning/pruning: {len(df)}")
    print("Class distribution (top 10):")
    print(df["author"].value_counts().head(10))

    if df["author"].nunique() < 2:
        raise SystemExit("Need at least 2 authors after cleaning to train a classifier.")

    # Train/Val split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["author"],
        test_size=0.2,
        random_state=42,
        stratify=df["author"]
    )

    # Build model
    pipe = build_pipeline()

    # Quick cross-validation on the training set (stratified for fairness)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running 5-fold CV on train…")
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV accuracy: mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full train
    print("Fitting pipeline on full training data…")
    pipe.fit(X_train, y_train)

    # Evaluate on held-out test
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # Save the whole pipeline (vectorizer + classifier)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH.resolve()}")

    # Try a quick demo prediction
    demo = "The world as we have created it is a process of our thinking."
    print("\nDemo prediction:")
    print("Text:", demo)
    print("Predicted author:", pipe.predict([demo])[0])

if __name__ == "__main__":
    main()