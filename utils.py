import pandas as pd
import re


def url_to_semantics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)

    if not urls:
        return ""

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()

        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3:
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()]

        for part in path_parts:
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""

    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]].copy())

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"

            if col_name in train_dataset.columns:
                sub_dataset = train_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0

                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]

                if not sub_dataset.empty:
                    flatten.append(sub_dataset)

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"

            if col_name in test_dataset.columns:
                sub_dataset = test_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0

                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]

                if not sub_dataset.empty:
                    flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body', 'rule'], keep='first', inplace=True)

    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
