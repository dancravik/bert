import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HTML_ENTITY_RE = re.compile(r"&\w+;")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Лёгкая очистка твита: убираем url, @упоминания, html-сущности, схлопываем пробелы.
    Хэштеги не выбрасываем целиком, а только убираем символ '#' -- само слово часто несёт сигнал."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_ENTITY_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def load_raw(cfg: dict):
    encoding = cfg["data"].get("encoding", "latin1")
    train_df = pd.read_csv(cfg["data"]["train_path"], encoding=encoding)
    test_df = pd.read_csv(cfg["data"]["test_path"], encoding=encoding)
    return train_df, test_df


def prepare_datasets(cfg: dict, label_encoder: LabelEncoder = None):
    """Возвращает (train_df, val_df, test_df, label_encoder).
    val_df -- стратифицированный сплит из train (доля data.val_size).
    Все df содержат колонку 'label_id' с числовым классом."""
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    train_df, test_df = load_raw(cfg)

    for df in (train_df, test_df):
        df.dropna(subset=[text_col, label_col], inplace=True)
        if cfg["data"].get("clean_text", True):
            df[text_col] = df[text_col].apply(clean_text)
        else:
            df[text_col] = df[text_col].astype(str)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df[label_col])

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["label_id"] = label_encoder.transform(train_df[label_col])
    test_df["label_id"] = label_encoder.transform(test_df[label_col])

    val_size = cfg["data"].get("val_size", 0.1)
    seed = cfg["data"].get("seed", 42)
    train_part, val_part = train_test_split(
        train_df, test_size=val_size, random_state=seed, stratify=train_df["label_id"]
    )
    return (
        train_part.reset_index(drop=True),
        val_part.reset_index(drop=True),
        test_df.reset_index(drop=True),
        label_encoder,
    )


class TextClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, max_length: int):
        self.texts = df[text_col].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["idx"] = idx
        return item
