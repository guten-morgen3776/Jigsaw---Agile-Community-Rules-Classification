import os
import re
import random
import warnings
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
from umap import UMAP


warnings.filterwarnings("ignore")


# ===============================
# Similarity control parameters
# ===============================
# 変更理由: スコアリングに用いる重みをまとめて管理することで、後からの調整や
#          探索を容易にしつつ一貫したロジックを保つため。
SIM_WEIGHTS = {
    "max": 0.6,
    "mean": 0.3,
    "contrast": 0.1,
}
TEMPERATURE = 0.75  # 変更理由: スコアをtanhで圧縮し極端な値を抑えることでAUCの安定化を狙う。


def cleaner(text):
    """Replace URLs with format: <url>: (domain/important-path)"""
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


def load_test_data():
    """Load test data."""
    print("Loading test data...")
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    print(f"Loaded {len(test_df)} test examples")
    print(f"Unique rules: {test_df['rule'].nunique()}")
    return test_df


def collect_all_texts(test_df):
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


def generate_embeddings(texts, model, batch_size=64):
    """Generate BGE embeddings for all texts."""
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
    test_df,
    augmentation_factor=2,
    random_seed=42,
    subsample_fraction=1.0,
):
    """Create triplet dataset from test data."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    anchors = []
    positives = []
    negatives = []

    print("Creating rule-aligned triplets from test data...")

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Processing test rows"
    ):
        rule = cleaner(str(row["rule"]))

        pos_examples = []
        neg_examples = []

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

        rule_positives = {}
        rule_negatives = {}

        for rule in test_df["rule"].unique():
            rule_df = test_df[test_df["rule"] == rule]

            pos_pool = []
            neg_pool = []

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
    model,
    train_dataset,
    epochs=3,
    batch_size=32,
    learning_rate=2e-5,
    margin=0.25,
    output_dir="./models/test-finetuned-bge",
):
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

    return model, final_model_path


def load_or_create_finetuned_model(test_df):
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

    fine_tuned_model, model_path = fine_tune_model(
        model=base_model,
        train_dataset=triplet_dataset,
        epochs=1,
        batch_size=16,
        learning_rate=2e-5,
        margin=0.25,
    )

    print(f"Fine-tuning completed. Model saved to: {model_path}")
    fine_tuned_model.half()
    return fine_tuned_model


def _normalize(vec):
    """Safely normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def generate_rule_embeddings(test_df, model):
    """Generate embeddings for each unique rule."""
    print("Generating rule embeddings...")

    unique_rules = test_df["rule"].unique()
    rule_embeddings = {}

    for rule in unique_rules:
        clean_rule = cleaner(str(rule))
        rule_emb = model.encode(
            clean_rule, convert_to_tensor=False, normalize_embeddings=True
        )
        rule_embeddings[rule] = np.asarray(rule_emb)

    print(f"Generated embeddings for {len(rule_embeddings)} rules")
    return rule_embeddings


def _stack_centroids(centroid_list):
    if not centroid_list:
        return np.zeros((0,))
    return np.stack(centroid_list)


def create_rule_centroids_with_hierarchical_clustering(
    test_df, text_to_embedding, rule_embeddings
):
    """Create centroids using Hierarchical Clustering + UMAP."""
    print("\nCreating rule centroids with Hierarchical Clustering + UMAP...")

    umap_reducer = UMAP(n_components=32, random_state=42)

    rule_centroids = {}

    for rule in test_df["rule"].unique():
        rule_data = test_df[test_df["rule"] == rule]

        pos_embeddings = []
        for _, row in rule_data.iterrows():
            for col in ["positive_example_1", "positive_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        pos_embeddings.append(text_to_embedding[clean_text])

        neg_embeddings = []
        for _, row in rule_data.iterrows():
            for col in ["negative_example_1", "negative_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        neg_embeddings.append(text_to_embedding[clean_text])

        if pos_embeddings and neg_embeddings:
            pos_embeddings = np.array(pos_embeddings)
            neg_embeddings = np.array(neg_embeddings)

            if (
                pos_embeddings.shape[0] > 10
                and pos_embeddings.shape[0] > umap_reducer.n_components
            ):
                pos_reduced = umap_reducer.fit_transform(pos_embeddings)
            else:
                pos_reduced = pos_embeddings

            if (
                neg_embeddings.shape[0] > 10
                and neg_embeddings.shape[0] > umap_reducer.n_components
            ):
                neg_reduced = umap_reducer.fit_transform(neg_embeddings)
            else:
                neg_reduced = neg_embeddings

            n_pos_clusters = min(3, len(pos_embeddings))
            n_neg_clusters = min(3, len(neg_embeddings))

            pos_centroids = []
            neg_centroids = []

            if n_pos_clusters > 1:
                pos_clusterer = AgglomerativeClustering(n_clusters=n_pos_clusters)
                pos_labels = pos_clusterer.fit_predict(pos_reduced)

                for cluster_id in np.unique(pos_labels):
                    cluster_mask = pos_labels == cluster_id
                    cluster_embeddings = pos_embeddings[cluster_mask]
                    cluster_centroid = _normalize(cluster_embeddings.mean(axis=0))
                    pos_centroids.append(cluster_centroid)
            else:
                pos_centroid = _normalize(pos_embeddings.mean(axis=0))
                pos_centroids.append(pos_centroid)

            if n_neg_clusters > 1:
                neg_clusterer = AgglomerativeClustering(n_clusters=n_neg_clusters)
                neg_labels = neg_clusterer.fit_predict(neg_reduced)

                for cluster_id in np.unique(neg_labels):
                    cluster_mask = neg_labels == cluster_id
                    cluster_embeddings = neg_embeddings[cluster_mask]
                    cluster_centroid = _normalize(cluster_embeddings.mean(axis=0))
                    neg_centroids.append(cluster_centroid)
            else:
                neg_centroid = _normalize(neg_embeddings.mean(axis=0))
                neg_centroids.append(neg_centroid)

            pos_proto = _normalize(np.mean(pos_centroids, axis=0))
            neg_proto = _normalize(np.mean(neg_centroids, axis=0))
            contrast_vector = _normalize(neg_proto - pos_proto)
            # 変更理由: 違反・非違反の平均ベクトル差分を保持することで、スコア計算時に
            #          より滑らかな方向性情報を利用しAUC向上を狙う。

            rule_centroids[rule] = {
                "positive_centroids": pos_centroids,
                "negative_centroids": neg_centroids,
                "pos_count": len(pos_embeddings),
                "neg_count": len(neg_embeddings),
                "rule_embedding": rule_embeddings[rule],
                "contrast_vector": contrast_vector,
            }

            print(
                f"  Rule: {rule[:50]}... - Pos: {len(pos_embeddings)}, "
                f"Neg: {len(neg_embeddings)} - Clusters: Pos={len(pos_centroids)}, "
                f"Neg={len(neg_centroids)}"
            )

    print(f"Created hierarchical centroids for {len(rule_centroids)} rules")
    return rule_centroids


def _compute_similarity_stats(embedding, centroids):
    if not centroids:
        return {"max": 0.0, "mean": 0.0}

    centroid_matrix = _stack_centroids(centroids)
    sims = centroid_matrix @ embedding

    return {"max": float(np.max(sims)), "mean": float(np.mean(sims))}


def predict_test_set_with_hierarchical_clustering(
    test_df, text_to_embedding, rule_centroids
):
    """Predict test set using hierarchical clustering centroids."""
    print("\nMaking predictions on test set with Hierarchical Clustering centroids...")

    row_ids = []
    predictions = []

    for rule in test_df["rule"].unique():
        print(f"  Processing rule: {rule[:50]}...")
        rule_data = test_df[test_df["rule"] == rule]

        if rule not in rule_centroids:
            continue

        centroids_info = rule_centroids[rule]
        pos_centroids = centroids_info["positive_centroids"]
        neg_centroids = centroids_info["negative_centroids"]
        contrast_vector = centroids_info["contrast_vector"]

        for _, row in rule_data.iterrows():
            body = cleaner(str(row["body"]))
            row_id = row["row_id"]

            if body not in text_to_embedding:
                continue

            body_embedding = text_to_embedding[body]

            pos_stats = _compute_similarity_stats(body_embedding, pos_centroids)
            neg_stats = _compute_similarity_stats(body_embedding, neg_centroids)

            contrast_score = float(body_embedding @ contrast_vector)
            # 変更理由: クラスタ平均差分との内積を導入し、単一の最近傍距離に依存しない
            #          滑らかな判定指標でAUC向上を狙う。

            raw_score = (
                SIM_WEIGHTS["max"] * (neg_stats["max"] - pos_stats["max"])
                + SIM_WEIGHTS["mean"] * (neg_stats["mean"] - pos_stats["mean"])
                + SIM_WEIGHTS["contrast"] * contrast_score
            )

            rule_prediction = np.tanh(raw_score / TEMPERATURE)

            row_ids.append(row_id)
            predictions.append(rule_prediction)

    print(f"Made predictions for {len(predictions)} test examples")
    return row_ids, np.array(predictions)


def main():
    """Main inference pipeline with Hierarchical Clustering + UMAP improvements."""
    print("=" * 70)
    print("IMPROVED SIMILARITY CLASSIFIER - HIERARCHICAL CLUSTERING + UMAP")
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

    rule_embeddings = generate_rule_embeddings(test_df, model)

    rule_centroids = create_rule_centroids_with_hierarchical_clustering(
        test_df, text_to_embedding, rule_embeddings
    )

    print("\n" + "=" * 50)
    print("PREDICTION PHASE")
    print("=" * 50)
    row_ids, predictions = predict_test_set_with_hierarchical_clustering(
        test_df, text_to_embedding, rule_centroids
    )

    submission_df = pd.DataFrame({"row_id": row_ids, "rule_violation": predictions})

    submission_df.to_csv("submission.csv", index=False)
    print(f"\nSaved predictions for {len(submission_df)} test examples to submission.csv")

    print("\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING + UMAP INFERENCE COMPLETED")
    print("Model: Fine-tuned BGE on test data triplets")
    print("Method: Hierarchical clustering + UMAP dimensionality reduction")
    print(f"Predicted on {len(test_df)} test examples")
    if len(predictions) > 0:
        print(
            f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, "
            f"mean={predictions.mean():.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
