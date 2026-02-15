"""Initial bag of words baseline test using flat classification"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def flatBagOfWords(df_valid, label_columns, BATCH_SIZE = 10000, RESULTS_FILE = "bagOfWords_classification_reports.csv"):
    # --- STORE RESULTS ---
    all_results = []

    # --- TRAINING FUNCTION ---
    def train_and_evaluate(df_valid, label, vectorizer, batch_mode=False):
        """Train + evaluate a classifier for a given label with a given vectorizer."""
        print(f"\n=== Training model for {label.upper()} using {vectorizer.__class__.__name__} ===")

        # Drop NaNs
        df_label = df_valid.dropna(subset=[label]).reset_index(drop=True)
        if df_label.empty or df_label[label].nunique() < 2:
            print(f"⚠ Skipping {label}, not enough valid classes.")
            return

        texts = df_label["text"].tolist()
        y = df_label[label]

        stratify_opt = y if y.value_counts().min() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=stratify_opt
        )

        clf = SGDClassifier(loss="log_loss", max_iter=5)

        if batch_mode:  
            # --- Incremental training with HashingVectorizer ---
            classes = list(set(y_train))
            n_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Training on {len(X_train)} samples in {n_batches} batches...")

            for i, start in enumerate(range(0, len(X_train), BATCH_SIZE), 1):
                end = start + BATCH_SIZE
                X_batch = vectorizer.transform(X_train[start:end])
                y_batch = y_train.iloc[start:end]
                clf.partial_fit(X_batch, y_batch, classes=classes)
                print(f"  ✅ Finished batch {i}/{n_batches}")

            X_test_vec = vectorizer.transform(X_test)

        else:
            # --- Standard training (Count/TFIDF) ---
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            clf.fit(X_train_vec, y_train)

        # --- Evaluation ---
        print("Evaluating model...")
        y_pred = clf.predict(X_test_vec)

        # Get classification report as dict
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        # Flatten into DataFrame-friendly format
        for cls, metrics in report_dict.items():
            if isinstance(metrics, dict):
                row = {
                    "label_type": label,
                    "vectorizer": vectorizer.__class__.__name__,
                    "class": cls,
                    "precision": metrics.get("precision", None),
                    "recall": metrics.get("recall", None),
                    "f1-score": metrics.get("f1-score", None),
                    "support": metrics.get("support", None),
                }
                all_results.append(row)

        print(classification_report(y_test, y_pred))
        print(f"=== Done training {label.upper()} with {vectorizer.__class__.__name__} ===\n")


    # --- RUN EXPERIMENTS ---
    for label in label_columns:
        # HashingVectorizer (streaming / batch mode)
        hashing_vec = HashingVectorizer(
            n_features=2**18, stop_words="english", alternate_sign=False
        )
        train_and_evaluate(df_valid, label, hashing_vec, batch_mode=True)

        # CountVectorizer
        count_vec = CountVectorizer(max_features=20000, stop_words="english")
        train_and_evaluate(df_valid, label, count_vec, batch_mode=False)

        # TfidfVectorizer
        tfidf_vec = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1,2))
        train_and_evaluate(df_valid, label, tfidf_vec, batch_mode=False)

    # --- SAVE RESULTS ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_FILE, sep=";", index=False)
        print(f"📊 Saved all classification reports to {RESULTS_FILE}")