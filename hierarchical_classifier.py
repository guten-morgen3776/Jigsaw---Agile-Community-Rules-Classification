import os
import random
import re
import warnings
from typing import Dict, List, Tuple, TypedDict
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


def prepare_clean_text_columns(test_df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned text columns once to avoid repeated `cleaner`呼び出し."""

    columns = [
        "body",
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
        "rule",
    ]

    for col in columns:
        clean_col = f"{col}_clean"

        # 変更理由: 大規模データで毎回 cleaner を呼ぶとボトルネックになるため、
        #          事前に正規化済みテキスト列を持たせて後段処理を高速化する。
        test_df.loc[:, clean_col] = test_df[col].map(
            lambda val: cleaner(str(val)) if pd.notna(val) else None
        )

    return test_df


def collect_all_texts(test_df: pd.DataFrame) -> List[str]:
    """Collect all unique texts from test set."""
    print("\nCollecting all texts for embedding...")

    all_texts = set()

    body_col = "body_clean" if "body_clean" in test_df.columns else "body"

    for body in test_df[body_col]:
        if pd.notna(body):
            text = str(body) if body_col.endswith("_clean") else cleaner(str(body))
            all_texts.add(text)

    example_cols = [
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    ]

    for col in example_cols:
        clean_col = f"{col}_clean" if f"{col}_clean" in test_df.columns else col
        for example in test_df[clean_col]:
            if pd.notna(example):
                text = (
                    str(example)
                    if clean_col.endswith("_clean")
                    else cleaner(str(example))
                )
                all_texts.add(text)

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


class RuleCentroidSummary(TypedDict):
    compliant_centroids: np.ndarray
    violating_centroids: np.ndarray
    compliant_counts: np.ndarray
    violating_counts: np.ndarray
    compliant_radii: np.ndarray
    violating_radii: np.ndarray
    compliant_weight_inv: np.ndarray
    violating_weight_inv: np.ndarray


def softmin(dist_mat: np.ndarray, tau: float = 0.08) -> np.ndarray:
    """Temperature-controlled smooth minimum for distance aggregation."""

    if tau <= 0:
        raise ValueError("tau must be positive for softmin")

    distances = np.asarray(dist_mat, dtype=np.float64)
    if distances.ndim == 1:
        distances = distances[None, :]

    x = -distances / max(1e-12, tau)
    m = x.max(axis=1, keepdims=True)
    sm = np.exp(x - m).sum(axis=1, keepdims=True)
    result = (-tau) * (np.log(sm) + m)

    # 変更理由: squeeze() だとサンプル数が1件のときに0次元スカラーへ潰れてしまい、
    #          後段処理でイテラブルとして扱えずエラーになるため、axis=1を固定で
    #          取り出して (M,) 形状を維持する。
    return result[:, 0]


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Safely normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _cluster_embeddings(
    embeddings: List[np.ndarray],
    max_clusters: int = 3,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster embeddings and return centroid statistics as numpy arrays."""

    if not embeddings:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    matrix = np.vstack(embeddings).astype(np.float32)

    if matrix.shape[0] == 1:
        centroid = _normalize(matrix[0])
        radii = np.array([0.0], dtype=np.float32)
        counts = np.array([1], dtype=np.int32)
        weight_inv = (radii + eps) / counts.astype(np.float32)
        return centroid[None, :], counts, radii, weight_inv

    n_clusters = min(max_clusters, matrix.shape[0])

    if n_clusters <= 1:
        centroid = _normalize(matrix.mean(axis=0))
        diff = matrix - centroid
        distances = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32))
        radius = float(distances.mean())
        radii = np.array([radius], dtype=np.float32)
        counts = np.array([matrix.shape[0]], dtype=np.int32)
        weight_inv = (radii + eps) / counts.astype(np.float32)
        return centroid[None, :], counts, radii, weight_inv

    # 変更理由: 正例・負例それぞれで複数クラスタを持たせ、
    #          後段の1-NN距離比較に使うためAgglomerativeClusteringで代表点を抽出。
    #          Ward法 (デフォルト) で分割し、クラスタ内サンプル数も控えて解析できるようにする。
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clusterer.fit_predict(matrix)

    centroid_list: List[np.ndarray] = []
    count_list: List[int] = []
    radius_list: List[float] = []
    for cluster_id in np.unique(labels):
        cluster_embeddings = matrix[labels == cluster_id]
        centroid = _normalize(cluster_embeddings.mean(axis=0))
        centroid_list.append(centroid)
        count_list.append(int(cluster_embeddings.shape[0]))
        diff = cluster_embeddings - centroid
        distances = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32))
        radius_list.append(float(distances.mean()))

    centroids = np.vstack(centroid_list).astype(np.float32)
    counts = np.asarray(count_list, dtype=np.int32)
    radii = np.asarray(radius_list, dtype=np.float32)
    weight_inv = (radii + eps) / np.maximum(counts.astype(np.float32), 1.0)

    return centroids, counts, radii, weight_inv


def create_rule_centroids(
    test_df: pd.DataFrame,
    text_to_embedding: Dict[str, np.ndarray],
    max_clusters: int = 3,
) -> Dict[str, RuleCentroidSummary]:
    """Create multiple centroids per rule for compliant/violating examples.

    各ルール × (準拠 / 違反) の埋め込み集合に対し ``_cluster_embeddings`` を適用し、
    AgglomerativeClustering で生成した複数クラスタの中心とクラスタサイズを保存します。
    """
    print("\nCreating rule centroids with multi-cluster 1-NN scoring...")

    rule_centroids: Dict[str, RuleCentroidSummary] = {}

    compliant_cols = [
        f"{col}_clean" if f"{col}_clean" in test_df.columns else col
        for col in ["negative_example_1", "negative_example_2"]
    ]
    violating_cols = [
        f"{col}_clean" if f"{col}_clean" in test_df.columns else col
        for col in ["positive_example_1", "positive_example_2"]
    ]

    for rule, rule_data in test_df.groupby("rule", sort=False):
        compliant_embeddings: List[np.ndarray] = []
        violating_embeddings: List[np.ndarray] = []

        for row in rule_data.itertuples(index=False):
            for col in compliant_cols:
                value = getattr(row, col)
                if not value:
                    continue
                key = value if col.endswith("_clean") else cleaner(str(value))
                if key in text_to_embedding:
                    compliant_embeddings.append(text_to_embedding[key])
            for col in violating_cols:
                value = getattr(row, col)
                if not value:
                    continue
                key = value if col.endswith("_clean") else cleaner(str(value))
                if key in text_to_embedding:
                    violating_embeddings.append(text_to_embedding[key])

        if not compliant_embeddings or not violating_embeddings:
            continue

        (
            compliant_centroids,
            compliant_counts,
            compliant_radii,
            compliant_weight_inv,
        ) = _cluster_embeddings(
            compliant_embeddings, max_clusters=max_clusters
        )
        (
            violating_centroids,
            violating_counts,
            violating_radii,
            violating_weight_inv,
        ) = _cluster_embeddings(
            violating_embeddings, max_clusters=max_clusters
        )

        rule_centroids[rule] = {
            "compliant_centroids": compliant_centroids,
            "violating_centroids": violating_centroids,
            "compliant_counts": compliant_counts,
            "violating_counts": violating_counts,
            "compliant_radii": compliant_radii,
            "violating_radii": violating_radii,
            "compliant_weight_inv": compliant_weight_inv,
            "violating_weight_inv": violating_weight_inv,
        }

        print(
            # 変更理由: セントロイド数を明示してログ出力し、
            #          学習データのばらつきとクラスタ分割の具合を確認しやすくする。
            f"  Rule: {rule[:50]}... - compliant={len(compliant_centroids)}"
            f" (sizes={compliant_counts}), violating={len(violating_centroids)}"
            f" (sizes={violating_counts})"
        )

    print(f"Created centroids for {len(rule_centroids)} rules")
    return rule_centroids


def predict_test_set_with_centroids(
    test_df: pd.DataFrame,
    text_to_embedding: Dict[str, np.ndarray],
    rule_centroids: Dict[str, RuleCentroidSummary],
    tau: float = 0.08,
) -> np.ndarray:
    """Predict test set using softmin-aggregated, weighted centroid distances."""

    print("\nMaking predictions on test set with centroid softmin scoring...")

    row_ids: List[int] = []
    predictions: List[float] = []

    body_col = "body_clean" if "body_clean" in test_df.columns else "body"

    for rule, rule_data in test_df.groupby("rule", sort=False):
        if rule not in rule_centroids:
            continue

        summary = rule_centroids[rule]
        compliant_centroids = summary["compliant_centroids"]
        violating_centroids = summary["violating_centroids"]

        if compliant_centroids.size == 0 or violating_centroids.size == 0:
            continue

        compliant_weight_inv = summary["compliant_weight_inv"]
        violating_weight_inv = summary["violating_weight_inv"]

        body_embeddings: List[np.ndarray] = []
        body_row_ids: List[int] = []

        for row in rule_data.itertuples(index=False):
            body_value = getattr(row, body_col)
            if not body_value:
                continue

            body_key = (
                body_value if body_col.endswith("_clean") else cleaner(str(body_value))
            )

            embedding = text_to_embedding.get(body_key)
            if embedding is None:
                continue

            body_embeddings.append(embedding)
            body_row_ids.append(getattr(row, "row_id"))

        if not body_embeddings:
            continue

        body_matrix = np.vstack(body_embeddings).astype(np.float32)

        # 変更理由: ループを排して (M, C, D) のテンソル計算に置き換えることで、
        #          数万件規模の推論でも NumPy のベクトル化によりスループットを向上させる。
        compliant_diff = body_matrix[:, None, :] - compliant_centroids[None, :, :]
        compliant_dist = np.sqrt(
            np.sum(compliant_diff * compliant_diff, axis=2, dtype=np.float32)
        )
        compliant_adjusted = compliant_dist * compliant_weight_inv[None, :]

        violating_diff = body_matrix[:, None, :] - violating_centroids[None, :, :]
        violating_dist = np.sqrt(
            np.sum(violating_diff * violating_diff, axis=2, dtype=np.float32)
        )
        violating_adjusted = violating_dist * violating_weight_inv[None, :]

        compliant_scores = np.asarray(softmin(compliant_adjusted, tau=tau), dtype=np.float64)
        violating_scores = np.asarray(softmin(violating_adjusted, tau=tau), dtype=np.float64)
        rule_scores = violating_scores - compliant_scores

        rule_mean = float(rule_scores.mean())
        rule_std = float(rule_scores.std())

        if rule_std > 1e-6:
            normalized_scores = (rule_scores - rule_mean) / rule_std
        else:
            normalized_scores = rule_scores - rule_mean

        row_ids.extend(body_row_ids)
        predictions.extend(normalized_scores.astype(float))

    print(f"Made predictions for {len(predictions)} test examples")
    return row_ids, np.array(predictions)


def main():
    """Main inference pipeline."""
    print("=" * 70)
    print("SIMILARITY CLASSIFIER - SOFTMIN WEIGHTED CENTROIDS")
    print("=" * 70)

    test_df = load_test_data()
    test_df = prepare_clean_text_columns(test_df)

    print("\n" + "=" * 50)
    print("MODEL PREPARATION PHASE")
    print("=" * 50)
    model = load_or_create_finetuned_model(test_df)

    all_texts = collect_all_texts(test_df)

    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION PHASE")
    print("=" * 50)
    all_embeddings = generate_embeddings(all_texts, model)

    text_to_embedding = {
        text: np.asarray(emb, dtype=np.float32) for text, emb in zip(all_texts, all_embeddings)
    }

    rule_centroids = create_rule_centroids(
        test_df,
        text_to_embedding,
        max_clusters=3,
    )

    print("\n" + "=" * 50)
    print("PREDICTION PHASE")
    print("=" * 50)
    row_ids, predictions = predict_test_set_with_centroids(
        test_df, text_to_embedding, rule_centroids, tau=0.08
    )

    submission_df = pd.DataFrame({"row_id": row_ids, "rule_violation": predictions})

    submission_df.to_csv("submission.csv", index=False)
    print(f"\nSaved predictions for {len(submission_df)} test examples to submission.csv")

    print("\n" + "=" * 70)
    print("SOFTMIN-WEIGHTED CENTROID INFERENCE COMPLETED")
    print("Model: Fine-tuned BGE on test data triplets")
    print(
        "Method: Per-rule centroid clustering with cluster-weighted softmin scoring"
    )
    print(f"Predicted on {len(test_df)} test examples")
    if len(predictions) > 0:
        print(
            f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, "
            f"mean={predictions.mean():.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
