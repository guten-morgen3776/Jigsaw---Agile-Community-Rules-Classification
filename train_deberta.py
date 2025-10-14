import os
import pandas as pd
import torch
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.model_selection import train_test_split  # noqa: F401
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)

from utils import get_dataframe_to_train, url_to_semantics, extract_head_tail


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:
    model_name_or_path = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./deberta_v3_small_final_model"

    EPOCHS = 3
    LEARNING_RATE = 2e-5

    MAX_LENGTH = 512
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    FOCAL_GAMMA = 2.0


class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, sample_weights=None):
        self.encodings = encodings
        self.labels = labels
        self.sample_weights = sample_weights

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        if self.sample_weights is not None:
            item['sample_weight'] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def get_llrd_param_groups(
    model,
    base_lr=2e-5,
    layerwise_decay=0.9,
    head_lr_mult=10.0,
    weight_decay=0.01
):
    if hasattr(model, 'deberta'):
        layers = model.deberta.encoder.layer
        encoder_name = 'deberta.encoder.layer'
        embed_prefix = 'deberta.embeddings'
    elif hasattr(model, 'deberta_v2'):
        layers = model.deberta_v2.encoder.layer
        encoder_name = "deberta_v2.encoder.layer"
        embed_prefix = "deberta_v2.embeddings"
    else:
        raise ValueError("LLRD: This model is not recognized as DeBERTa family.")

    n_layers = len(layers)

    def is_no_decay(n):
        return n.endswith('.bias') or 'LayerNorm.weight' in n

    groups = []

    head_params = [(n, p) for n, p in model.named_parameters()
                   if ("classifier" in n or "score" in n) and p.requires_grad]
    if head_params:
        groups.append({"params": [p for n, p in head_params if not is_no_decay(n)],
                       "lr": base_lr * head_lr_mult, "weight_decay": weight_decay})
        groups.append({"params": [p for n, p in head_params if is_no_decay(n)],
                       "lr": base_lr * head_lr_mult, "weight_decay": 0.0})

    embed_params = [(n, p) for n, p in model.named_parameters()
                    if n.startswith(embed_prefix) and p.requires_grad]
    if embed_params:
        lr = base_lr * (layerwise_decay ** n_layers)
        groups.append({"params": [p for n, p in embed_params if not is_no_decay(n)],
                       "lr": lr, "weight_decay": weight_decay})
        groups.append({"params": [p for n, p in embed_params if is_no_decay(n)],
                       "lr": lr, "weight_decay": 0.0})

    for i in range(n_layers):
        prefix = f"{encoder_name}.{i}."
        layer_params = [(n, p) for n, p in model.named_parameters()
                        if n.startswith(prefix) and p.requires_grad]
        if not layer_params:
            continue
        lr = base_lr * (layerwise_decay ** (n_layers - 1 - i))
        groups.append({"params": [p for n, p in layer_params if not is_no_decay(n)],
                       "lr": lr, "weight_decay": weight_decay})
        groups.append({"params": [p for n, p in layer_params if is_no_decay(n)],
                       "lr": lr, "weight_decay": 0.0})
    return groups


def compute_num_training_steps(n_samples, epochs, batch_size, grad_accum=1):
    import math
    steps_per_epoch = math.ceil(n_samples / batch_size / grad_accum)
    return steps_per_epoch * epochs


@dataclass
class FocalLossConfig:
    gamma: float = 2.0


class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_config: Optional[FocalLossConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_config = focal_config or FocalLossConfig()

    def compute_loss(self, model, inputs, return_outputs=False):
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels") if "labels" in inputs else None
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = None
        if labels is not None:
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            labels = labels.view(-1)
            loss = self.focal_loss(logits, labels, sample_weight)

        return (loss, outputs) if return_outputs else loss

    def focal_loss(self, logits, labels, sample_weight=None):
        import torch.nn.functional as F

        gamma = self.focal_config.gamma
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        labels = labels.long()
        focal_weight = (1 - probs.gather(1, labels.unsqueeze(1)).squeeze()) ** gamma

        ce_loss = F.nll_loss(log_probs, labels, reduction='none')
        loss = focal_weight * ce_loss

        if sample_weight is not None:
            sample_weight = sample_weight.to(loss.device)
            loss = loss * sample_weight

        return loss.mean()


def build_input_text(rule: str, body: str) -> str:
    head, tail = extract_head_tail(body)
    segments = ["<RULE>", rule.strip(), "<BODY_HEAD>", head.strip()]
    tail = tail.strip()
    if tail:
        segments.extend(["<BODY_TAIL>", tail])
    return " ".join([seg for seg in segments if seg])


def prepare_special_tokens(tokenizer, model):
    special_tokens = {"additional_special_tokens": ["<RULE>", "<BODY_HEAD>", "<BODY_TAIL>"]}
    newly_added = tokenizer.add_special_tokens(special_tokens)
    if newly_added > 0:
        model.resize_token_embeddings(len(tokenizer))


def main():
    seed_everything(42)
    training_data_df = get_dataframe_to_train(CFG.data_path)
    print(f"Training dataset (from examples only) size: {len(training_data_df)}")

    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")

    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    training_data_df['input_text'] = training_data_df.apply(
        lambda row: build_input_text(row['rule'], row['body_with_url']), axis=1
    )

    rule_counts = training_data_df.groupby(['rule', 'rule_violation']).size().unstack(fill_value=0)
    def compute_alpha(row):
        pos = row.get(1, 0)
        neg = row.get(0, 0)
        total = pos + neg
        if total == 0:
            return {0: 0.5, 1: 0.5}
        pos_alpha = neg / total if total > 0 else 0.5
        neg_alpha = pos / total if total > 0 else 0.5
        if pos == 0:
            pos_alpha = 0.5
        if neg == 0:
            neg_alpha = 0.5
        return {0: neg_alpha, 1: pos_alpha}

    alpha_map = {rule: compute_alpha(counts) for rule, counts in rule_counts.iterrows()}
    training_data_df['sample_alpha'] = training_data_df.apply(
        lambda row: alpha_map.get(row['rule'], {0: 0.5, 1: 0.5}).get(row['rule_violation'], 0.5),
        axis=1
    )

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)
    prepare_special_tokens(tokenizer, model)

    train_encodings = tokenizer(
        training_data_df['input_text'].tolist(),
        truncation=True,
        padding=True,
        max_length=CFG.MAX_LENGTH
    )
    train_labels = training_data_df['rule_violation'].tolist()
    train_weights = training_data_df['sample_alpha'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels, train_weights)

    param_groups = get_llrd_param_groups(
        model,
        base_lr=CFG.LEARNING_RATE,
        layerwise_decay=0.92,
        head_lr_mult=8.0,
        weight_decay=0.01
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    num_training_steps = compute_num_training_steps(
        n_samples=len(training_data_df),
        epochs=CFG.EPOCHS,
        batch_size=CFG.BATCH_SIZE,
        grad_accum=CFG.GRADIENT_ACCUMULATION_STEPS
    )
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=0.0,
        weight_decay=0.0,
        report_to="none",
        save_strategy="no",
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, scheduler),
        focal_config=FocalLossConfig(gamma=CFG.FOCAL_GAMMA)
    )

    trainer.train()

    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['input_text'] = test_df_for_prediction.apply(
        lambda row: build_input_text(row['rule'], row['body_with_url']), axis=1
    )

    test_encodings = tokenizer(
        test_df_for_prediction['input_text'].tolist(),
        truncation=True,
        padding=True,
        max_length=CFG.MAX_LENGTH
    )
    test_dataset = JigsawDataset(test_encodings)

    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": probs
    })
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
