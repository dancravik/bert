"""
Шаг 1: находим бейзлайн.

Прогоняем bert-base-uncased в двух режимах:
  - frozen_backbone : учим только новую голову, backbone заморожен
  - full_finetune    : учим всё целиком

В каждом режиме перебираем несколько lr и batch_size (короткое обучение,
sweep.epochs эпох). Всё логируется в Comet отдельными экспериментами
(теги: sweep, baseline, <режим>). Результат:
  outputs/sweep_results.csv         -- таблица со всеми прогонами
  outputs/best_baseline_config.yaml -- лучшая связка (freeze/lr/head_lr/batch_size)

Запуск:
    python scripts/sweep_baseline.py --config configs/config.yaml
"""
import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import run_experiment
from src.utils import load_config


def main(config_path: str):
    cfg = load_config(config_path)
    baseline = cfg["models"]["baseline"]
    sweep_cfg = cfg["sweep"]
    epochs = sweep_cfg.get("epochs", 2)

    modes = [
        ("frozen_backbone", True, sweep_cfg["frozen_backbone"]),
        ("full_finetune", False, sweep_cfg["full_finetune"]),
    ]

    results = []
    for mode_name, freeze, grid in modes:
        for lr in grid["lr_values"]:
            for bs in grid["batch_size_values"]:
                run_cfg = {
                    "checkpoint": baseline["checkpoint"],
                    "display_name": f"{baseline['display_name']}-sweep-{mode_name}-lr{lr}-bs{bs}",
                    "freeze_backbone": freeze,
                    # при заморозке backbone главный параметр -- head_lr, backbone lr не используется,
                    # при полном fine-tune наоборот: главный параметр -- lr backbone
                    "lr": lr if not freeze else cfg["training"].get("lr", 2e-5),
                    "head_lr": lr if freeze else cfg["training"].get("head_lr", 1e-3),
                    "batch_size": bs,
                    "epochs": epochs,
                    "tags": ["sweep", "baseline", mode_name],
                }
                print(f"\n=== SWEEP: {mode_name} | lr={lr} | batch_size={bs} ===")
                result = run_experiment(cfg, run_cfg)
                results.append(
                    {
                        "mode": mode_name,
                        "freeze_backbone": freeze,
                        "lr": lr,
                        "batch_size": bs,
                        "val_f1_macro": result["val_f1_macro"],
                        "test_f1_macro": result["test_metrics"]["f1_macro"],
                        "test_accuracy": result["test_metrics"]["accuracy"],
                        "test_loss": result["test_metrics"].get("log_loss"),
                    }
                )

    results_df = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, "sweep_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ SWEEP (отсортировано по val_f1_macro)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    best_cfg = {
        "freeze_backbone": bool(best["freeze_backbone"]),
        "lr": float(best["lr"]) if not best["freeze_backbone"] else float(cfg["training"].get("lr", 2e-5)),
        "head_lr": float(best["lr"]) if best["freeze_backbone"] else float(cfg["training"].get("head_lr", 1e-3)),
        "batch_size": int(best["batch_size"]),
    }
    with open(os.path.join(out_dir, "best_baseline_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True)

    frozen_best = results_df[results_df["mode"] == "frozen_backbone"].iloc[0]
    finetune_best = results_df[results_df["mode"] == "full_finetune"].iloc[0]

    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ РЕЖИМОВ (лучшая конфигурация в каждом)")
    print("=" * 70)
    print(
        f"frozen_backbone : val_f1_macro={frozen_best['val_f1_macro']:.4f} | "
        f"test_f1_macro={frozen_best['test_f1_macro']:.4f} | "
        f"lr_head={frozen_best['lr']} | batch_size={frozen_best['batch_size']}"
    )
    print(
        f"full_finetune   : val_f1_macro={finetune_best['val_f1_macro']:.4f} | "
        f"test_f1_macro={finetune_best['test_f1_macro']:.4f} | "
        f"lr={finetune_best['lr']} | batch_size={finetune_best['batch_size']}"
    )
    diff = finetune_best["val_f1_macro"] - frozen_best["val_f1_macro"]
    winner = "full_finetune" if diff > 0 else "frozen_backbone"
    print(
        f"\nРазница val_f1_macro (full_finetune - frozen_backbone) = {diff:+.4f} "
        f"-> в качестве бейзлайна выбран режим: {winner}"
    )
    print(f"\nЛучшая конфигурация сохранена в {out_dir}/best_baseline_config.yaml:")
    print(best_cfg)
    print(
        "\nДальше запусти scripts/run_all_models.py -- он подхватит эту конфигурацию "
        "и обучит бейзлайн + 4 модификации BERT."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
