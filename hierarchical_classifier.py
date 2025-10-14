import os
import random
import re
import warnings
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
)
from sentence_transformers.losses import TripletLoss
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")


def cleaner(text: str) -> str:
    """Replace URLs with a condensed placeholder."""
    if not text:
        return text

    url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"

    def replace_url(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                important_path = "/".join(path_parts[:2])
                return f"<url>: ({domain}/{important_path})"
            return f"<url>: ({domain})"
        except Exception:
            return "<url>: (unknown)"

    return re.sub(url_pattern, replace_url, str(text))


def load_test_data() -> pd.DataFrame:
    """Load test data."""
    print("Loading test data...")
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    print(f"Loaded {len(test_df)} test examples")
    print(f"Unique rules: {test_df['rule'].nunique()}")
    return test_df


def collect_all_texts(test_df: pd.DataFrame) -> List[str]:
    """Collect all unique texts from test set."""
    print("\nCollecting all texts for embedding...")

    all_texts = set()

    for body in test_df["body"]:
        if pd.notna(body):
            all_texts.add(cleaner(str(body)))

    example_cols = [
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    ]

    for col in example_cols:
        for example in test_df[col]:
            if pd.notna(example):
                all_texts.add(cleaner(str(example)))

    all_texts = list(all_texts)
    print(f"Collected {len(all_texts)} unique texts")
    return all_texts


def generate_embeddings(texts: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Generate embeddings for all texts."""
    print(f"Generating embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        sentences=texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings)


def create_test_triplet_dataset(
    test_df: pd.DataFrame,
    augmentation_factor: int = 2,
    random_seed: int = 42,
    subsample_fraction: float = 1.0,
) -> Dataset:
    """Create triplet dataset from test data."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    anchors: List[str] = []
    positives: List[str] = []
    negatives: List[str] = []

    print("Creating rule-aligned triplets from test data...")

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Processing test rows"
    ):
        rule = cleaner(str(row["rule"]))

        pos_examples: List[str] = []
        neg_examples: List[str] = []

        for neg_col in ["negative_example_1", "negative_example_2"]:
            if pd.notna(row[neg_col]):
                pos_examples.append(cleaner(str(row[neg_col])))

        for pos_col in ["positive_example_1", "positive_example_2"]:
            if pd.notna(row[pos_col]):
                neg_examples.append(cleaner(str(row[pos_col])))

        for pos_ex in pos_examples:
            for neg_ex in neg_examples:
                anchors.append(rule)
                positives.append(pos_ex)
                negatives.append(neg_ex)

    if augmentation_factor > 0:
        print(f"Adding {augmentation_factor}x augmentation...")

        rule_positives: Dict[str, List[str]] = {}
        rule_negatives: Dict[str, List[str]] = {}

        for rule in test_df["rule"].unique():
            rule_df = test_df[test_df["rule"] == rule]

            pos_pool: List[str] = []
            neg_pool: List[str] = []

            for _, row in rule_df.iterrows():
                for neg_col in ["negative_example_1", "negative_example_2"]:
                    if pd.notna(row[neg_col]):
                        pos_pool.append(cleaner(str(row[neg_col])))
                for pos_col in ["positive_example_1", "positive_example_2"]:
                    if pd.notna(row[pos_col]):
                        neg_pool.append(cleaner(str(row[pos_col])))

            rule_positives[rule] = list(set(pos_pool))
            rule_negatives[rule] = list(set(neg_pool))

        for rule in test_df["rule"].unique():
            clean_rule = cleaner(str(rule))
            pos_pool = rule_positives[rule]
            neg_pool = rule_negatives[rule]

            n_samples = min(
                augmentation_factor * len(pos_pool), len(pos_pool) * len(neg_pool)
            )

            for _ in range(n_samples):
                if pos_pool and neg_pool:
                    anchors.append(clean_rule)
                    positives.append(random.choice(pos_pool))
                    negatives.append(random.choice(neg_pool))

    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)

    original_count = len(combined)
    if subsample_fraction < 1.0:
        n_samples = int(len(combined) * subsample_fraction)
        combined = combined[:n_samples]
        print(
            f"Subsampled {original_count} -> {len(combined)} triplets "
            f"({subsample_fraction*100:.1f}%)"
        )

    anchors, positives, negatives = zip(*combined) if combined else ([], [], [])

    print(f"Created {len(anchors)} triplets from test data")

    dataset = Dataset.from_dict(
        {
            "anchor": list(anchors),
            "positive": list(positives),
            "negative": list(negatives),
        }
    )

    return dataset


def fine_tune_model(
    model: SentenceTransformer,
    train_dataset: Dataset,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    margin: float = 0.25,
    output_dir: str = "./models/test-finetuned-bge",
) -> SentenceTransformer:
    """Fine-tune the sentence transformer model using triplet loss."""

    print(f"Fine-tuning model on {len(train_dataset)} triplets...")

    loss = TripletLoss(model=model, triplet_margin=margin)

    dataset_size = len(train_dataset)
    steps_per_epoch = max(1, dataset_size // batch_size)
    max_steps = steps_per_epoch * epochs

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=0,
        learning_rate=learning_rate,
        logging_steps=max(1, max_steps // 4),
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()

    final_model_path = f"{output_dir}/final"
    print(f"Saving fine-tuned model to {final_model_path}...")
    model.save_pretrained(final_model_path)

    return model


def load_or_create_finetuned_model(test_df: pd.DataFrame) -> SentenceTransformer:
    """Load fine-tuned model if exists, otherwise fine-tune a new one."""

    fine_tuned_path = "./models/test-finetuned-bge/final"

    if os.path.exists(fine_tuned_path):
        print(f"Loading existing fine-tuned model from {fine_tuned_path}...")
        try:
            word_embedding_model = models.Transformer(
                fine_tuned_path, max_seq_length=128, do_lower_case=True
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode="mean",
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            print("Loaded fine-tuned model with explicit pooling")
        except Exception:
            model = SentenceTransformer(fine_tuned_path)
            print("Loaded fine-tuned model with default configuration")
        model.half()
        return model

    print("Fine-tuned model not found. Creating new one...")

    print("Loading base BGE embedding model...")
    try:
        model_path = "/kaggle/input/baai/transformers/bge-base-en-v1.5/1"
        word_embedding_model = models.Transformer(
            model_path, max_seq_length=256, do_lower_case=True
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        base_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("Loaded base model from Kaggle path with explicit pooling")
    except Exception:
        model_path = ""
        word_embedding_model = models.Transformer(
            model_path, max_seq_length=256, do_lower_case=True
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        base_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("Loaded base model from local path with explicit pooling")

    triplet_dataset = create_test_triplet_dataset(
        test_df, augmentation_factor=2, subsample_fraction=1.0
    )

    fine_tuned_model = fine_tune_model(
        model=base_model,
        train_dataset=triplet_dataset,
        epochs=1,
        batch_size=16,
        learning_rate=2e-5,
        margin=0.25,
    )

    print("Fine-tuning completed. Model saved to: ./models/test-finetuned-bge/final")
    fine_tuned_model.half()
    return fine_tuned_model


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Safely normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _cluster_embeddings(
    embeddings: List[np.ndarray],
    max_clusters: int = 3,
) -> List[np.ndarray]:
    """Cluster embeddings and return centroids."""
    if not embeddings:
        return []

    matrix = np.vstack(embeddings)
    if matrix.shape[0] == 1:
        return [_normalize(matrix[0])]

    n_clusters = min(max_clusters, matrix.shape[0])

    if n_clusters <= 1:
        return [_normalize(matrix.mean(axis=0))]

    # 変更理由: 正例・負例それぞれで複数クラスタを持たせ、
    #          後段の1-NN距離比較に使うためAgglomerativeClusteringで代表点を抽出。
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clusterer.fit_predict(matrix)

    centroids: List[np.ndarray] = []
    for cluster_id in np.unique(labels):
        cluster_embeddings = matrix[labels == cluster_id]
        centroid = _normalize(cluster_embeddings.mean(axis=0))
        centroids.append(centroid)

    return centroids


def create_rule_centroids(
    test_df: pd.DataFrame,
    text_to_embedding: Dict[str, np.ndarray],
    max_clusters: int = 3,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Create multiple centroids per rule for compliant/violating examples."""
    print("\nCreating rule centroids with multi-cluster 1-NN scoring...")

    rule_centroids: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for rule in test_df["rule"].unique():
        rule_data = test_df[test_df["rule"] == rule]

        compliant_embeddings: List[np.ndarray] = []
        violating_embeddings: List[np.ndarray] = []

        for _, row in rule_data.iterrows():
            for col in ["negative_example_1", "negative_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        compliant_embeddings.append(text_to_embedding[clean_text])
            for col in ["positive_example_1", "positive_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        violating_embeddings.append(text_to_embedding[clean_text])

        if not compliant_embeddings or not violating_embeddings:
            continue

        compliant_centroids = _cluster_embeddings(
            compliant_embeddings, max_clusters=max_clusters
        )
        violating_centroids = _cluster_embeddings(
            violating_embeddings, max_clusters=max_clusters
        )

        rule_centroids[rule] = {
            "compliant_centroids": compliant_centroids,
            "violating_centroids": violating_centroids,
        }

        print(
            # 変更理由: セントロイド数を明示してログ出力し、
            #          学習データのばらつきとクラスタ分割の具合を確認しやすくする。
            f"  Rule: {rule[:50]}... - compliant={len(compliant_centroids)}, "
            f"violating={len(violating_centroids)}"
        )

    print(f"Created centroids for {len(rule_centroids)} rules")
    return rule_centroids


def _nearest_distance(
    embedding: np.ndarray, centroids: List[np.ndarray]
) -> float:
    if not centroids:
        return float("inf")
    distances = [np.linalg.norm(embedding - centroid) for centroid in centroids]
    return float(min(distances))


def predict_test_set_with_centroids(
    test_df: pd.DataFrame,
    text_to_embedding: Dict[str, np.ndarray],
    rule_centroids: Dict[str, Dict[str, List[np.ndarray]]],
) -> np.ndarray:
    """Predict test set using 1-NN distance over centroids."""
    print("\nMaking predictions on test set with centroid 1-NN scoring...")

    row_ids: List[int] = []
    predictions: List[float] = []

    for rule in test_df["rule"].unique():
        rule_data = test_df[test_df["rule"] == rule]

        if rule not in rule_centroids:
            continue

        compliant_centroids = rule_centroids[rule]["compliant_centroids"]
        violating_centroids = rule_centroids[rule]["violating_centroids"]

        for _, row in rule_data.iterrows():
            body = cleaner(str(row["body"]))
            row_id = row["row_id"]

            if body not in text_to_embedding:
                continue

            body_embedding = text_to_embedding[body]

            compliant_distance = _nearest_distance(body_embedding, compliant_centroids)
            violating_distance = _nearest_distance(body_embedding, violating_centroids)

            # 変更理由: 各ルールで最も近い正クラスタと負クラスタの距離差を
            #          直接スコア化することで、1-NN判定の鋭さを活かしAUC向上を狙う。
            rule_prediction = violating_distance - compliant_distance

            row_ids.append(row_id)
            predictions.append(rule_prediction)

    print(f"Made predictions for {len(predictions)} test examples")
    return row_ids, np.array(predictions)


def main():
    """Main inference pipeline."""
    print("=" * 70)
    print("SIMILARITY CLASSIFIER - MULTI-CENTROID 1-NN")
    print("=" * 70)

    test_df = load_test_data()

    print("\n" + "=" * 50)
    print("MODEL PREPARATION PHASE")
    print("=" * 50)
    model = load_or_create_finetuned_model(test_df)

    all_texts = collect_all_texts(test_df)

    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION PHASE")
    print("=" * 50)
    all_embeddings = generate_embeddings(all_texts, model)

    text_to_embedding = {text: emb for text, emb in zip(all_texts, all_embeddings)}

    rule_centroids = create_rule_centroids(
        test_df,
        text_to_embedding,
        max_clusters=3,
    )

    print("\n" + "=" * 50)
    print("PREDICTION PHASE")
    print("=" * 50)
    row_ids, predictions = predict_test_set_with_centroids(
        test_df, text_to_embedding, rule_centroids
    )

    submission_df = pd.DataFrame({"row_id": row_ids, "rule_violation": predictions})

    submission_df.to_csv("submission.csv", index=False)
    print(f"\nSaved predictions for {len(submission_df)} test examples to submission.csv")

    print("\n" + "=" * 70)
    print("MULTI-CENTROID 1-NN INFERENCE COMPLETED")
    print("Model: Fine-tuned BGE on test data triplets")
    print("Method: Per-rule centroid clustering with 1-NN distance scoring")
    print(f"Predicted on {len(test_df)} test examples")
    if len(predictions) > 0:
        print(
            f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, "
            f"mean={predictions.mean():.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
