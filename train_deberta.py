import os
import pandas as pd
import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)

from utils import get_dataframe_to_train, url_to_semantics


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
    SEEDS = [42, 2025, 196]


class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
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
        embed = model.deberta.embeddings
        encoder_name = 'deberta.encoder.layer'
        embed_prefix = 'deberta.embeddings'
    elif hasattr(model, 'deberta_v2'):
        layers = model.deberta_v2.encoder.layer
        embed = model.deberta_v2.embeddings
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


def train_and_predict_for_seed(seed, train_dataset, test_dataset, training_data_df):
    seed_everything(seed)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)

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
        grad_accum=1
    )
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    run_output_dir = os.path.join(CFG.output_dir, f"seed_{seed}")
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.0,
        weight_decay=0.0,
        report_to="none",
        save_strategy="no",
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    return probs


def main():
    training_data_df = get_dataframe_to_train(CFG.data_path)
    print(f"Training dataset (from examples only) size: {len(training_data_df)}")

    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")

    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    train_encodings = tokenizer(training_data_df['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    train_labels = training_data_df['rule_violation'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels)

    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['input_text'] = test_df_for_prediction['rule'] + "[SEP]" + test_df_for_prediction['body_with_url']

    test_encodings = tokenizer(test_df_for_prediction['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    test_dataset = JigsawDataset(test_encodings)

    seed_probabilities = []
    for seed in CFG.SEEDS:
        print(f"Starting training for seed {seed}")
        probs = train_and_predict_for_seed(seed, train_dataset, test_dataset, training_data_df)
        seed_probabilities.append(probs)

    averaged_probs = np.mean(np.stack(seed_probabilities, axis=0), axis=0)

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": averaged_probs
    })
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
