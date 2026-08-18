import inspect

import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TransformerWithHead(nn.Module):
    """Берём ЛЮБОЙ энкодер из transformers через AutoModel (то есть без исходной
    pretraining-головы: MLM/NSP/RTD и т.п. уже отсутствуют), поверх [CLS]-представления
    ставим свою классификационную голову. Один и тот же класс используется и для
    бейзлайна (bert-base-uncased), и для всех 4 модификаций -- это даёт честное сравнение,
    т.к. отличается только backbone, а голова и способ пулинга всегда одинаковые."""

    def __init__(
        self,
        checkpoint: str,
        num_labels: int,
        dropout: float = 0.1,
        hidden_dim: int = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.config.hidden_size

        if hidden_dim:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # У разных моделей разный форвард-сигнатур (например, DistilBERT не принимает
        # token_type_ids вообще). Запоминаем, какие ключи backbone реально понимает,
        # чтобы не падать на моделях без сегментных эмбеддингов.
        self._backbone_arg_names = set(inspect.signature(self.backbone.forward).parameters.keys())

    def forward(self, batch: dict):
        backbone_inputs = {
            k: v
            for k, v in batch.items()
            if k in self._backbone_arg_names and k in ("input_ids", "attention_mask", "token_type_ids")
        }
        outputs = self.backbone(**backbone_inputs)
        # Универсальный пулинг: берём представление [CLS]-токена (первая позиция).
        # Работает одинаково для BERT/RoBERTa/DistilBERT/ALBERT/ELECTRA, в отличие
        # от outputs.pooler_output, который есть не у всех моделей и по-разному устроен.
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_repr)
        return logits

    def trainable_parameter_groups(self, lr_backbone: float, lr_head: float, weight_decay: float = 0.01):
        """Разные lr для backbone и головы + отключаем weight decay на bias/LayerNorm,
        как рекомендовано в оригинальных статьях BERT/RoBERTa."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm", "LayerNorm.bias"]

        def is_no_decay(name: str) -> bool:
            return any(nd in name for nd in no_decay)

        groups = []

        backbone_trainable = [(n, p) for n, p in self.backbone.named_parameters() if p.requires_grad]
        if backbone_trainable:
            groups.append(
                {
                    "params": [p for n, p in backbone_trainable if not is_no_decay(n)],
                    "lr": lr_backbone,
                    "weight_decay": weight_decay,
                }
            )
            groups.append(
                {
                    "params": [p for n, p in backbone_trainable if is_no_decay(n)],
                    "lr": lr_backbone,
                    "weight_decay": 0.0,
                }
            )

        head_named = list(self.head.named_parameters())
        groups.append(
            {
                "params": [p for n, p in head_named if not is_no_decay(n)],
                "lr": lr_head,
                "weight_decay": weight_decay,
            }
        )
        groups.append(
            {
                "params": [p for n, p in head_named if is_no_decay(n)],
                "lr": lr_head,
                "weight_decay": 0.0,
            }
        )

        return [g for g in groups if len(g["params"]) > 0]
