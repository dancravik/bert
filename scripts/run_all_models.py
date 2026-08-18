"""
Шаг 2: обучаем baseline (BERT-base, с гиперпараметрами из sweep) и 4 модификации
BERT на тех же данных, тем же способом (одна и та же голова, один и тот же цикл
обучения/оценки/error-анализа) -- строим лидерборд и определяем лучшую модель.

Если scripts/sweep_baseline.py ещё не запускался, используются значения
training.* из config.yaml по умолчанию.

Запуск:
    python scripts/run_all_models.py --config configs/config.yaml
"""
import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import run_experiment
from src.utils import load_config


def load_best_baseline_hparams(cfg: dict) -> dict:
    path = os.path.join(cfg["output"]["dir"], "best_baseline_config.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            hparams = yaml.safe_load(f)
        print(f"[i] Использую гиперпараметры из {path}: {hparams}")
        return hparams

    print(
        "[!] outputs/best_baseline_config.yaml не найден -- использую значения из "
        "config.yaml (training.*). Рекомендуется сначала запустить scripts/sweep_baseline.py"
    )
    return {
        "freeze_backbone": cfg["training"].get("freeze_backbone", False),
        "lr": cfg["training"]["lr"],
        "head_lr": cfg["training"].get("head_lr", 1e-3),
        "batch_size": cfg["training"]["batch_size"],
    }


def main(config_path: str):
    cfg = load_config(config_path)
    best_hparams = load_best_baseline_hparams(cfg)

    all_models = [cfg["models"]["baseline"]] + cfg["models"]["modifications"]
    leaderboard = []

    for i, model_cfg in enumerate(all_models):
        is_baseline = i == 0
        run_cfg = {
            "checkpoint": model_cfg["checkpoint"],
            "display_name": model_cfg["display_name"],
            "freeze_backbone": model_cfg.get("freeze_backbone", best_hparams["freeze_backbone"]),
            "lr": model_cfg.get("lr", best_hparams["lr"]),
            "head_lr": model_cfg.get("head_lr", best_hparams["head_lr"]),
            "batch_size": model_cfg.get("batch_size", best_hparams["batch_size"]),
            "epochs": cfg["training"]["epochs"],
            "tags": ["final", "baseline" if is_baseline else "modification"],
        }
        print(f"\n{'=' * 70}\nОбучаю: {run_cfg['display_name']} ({run_cfg['checkpoint']})\n{'=' * 70}")
        result = run_experiment(cfg, run_cfg)

        row = {
            "display_name": result["display_name"],
            "checkpoint": result["checkpoint"],
            "is_baseline": is_baseline,
            "freeze_backbone": result["freeze_backbone"],
            "out_dir": result["out_dir"],
        }
        row.update({f"test_{k}": v for k, v in result["test_metrics"].items() if isinstance(v, (int, float))})
        leaderboard.append(row)

    lb_df = pd.DataFrame(leaderboard).sort_values("test_f1_macro", ascending=False)
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    lb_path = os.path.join(out_dir, "leaderboard.csv")
    lb_df.to_csv(lb_path, index=False)

    print("\n" + "=" * 70)
    print("ЛИДЕРБОРД (по test_f1_macro)")
    print("=" * 70)
    cols = ["display_name", "is_baseline", "test_accuracy", "test_f1_macro", "test_mcc", "test_roc_auc_ovr_macro"]
    cols = [c for c in cols if c in lb_df.columns]
    print(lb_df[cols].to_string(index=False))

    best_row = lb_df.iloc[0]
    print(
        f"\nЛучшая модель: {best_row['display_name']} "
        f"(test_f1_macro={best_row['test_f1_macro']:.4f}, "
        f"test_accuracy={best_row['test_accuracy']:.4f})"
    )
    print(f"\nПолная таблица сохранена в {lb_path}")
    print(f"Детали по каждой модели (метрики, top/bottom ошибки, веса) -- в {out_dir}/<model>_<mode>/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
