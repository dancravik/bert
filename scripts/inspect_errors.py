"""
Печатает в консоль топ худших и лучших (по loss) примеров для конкретного запуска.
Помогает делать выводы: на каких текстах модель ошибается больше/меньше всего.

Использование:
    python scripts/inspect_errors.py --run outputs/BERT-base_finetune --top 15
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="папка вида outputs/<run_name>")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--text-col", default="OriginalTweet")
    args = parser.parse_args()

    worst = pd.read_csv(os.path.join(args.run, "top_worst_losses.csv")).head(args.top)
    best = pd.read_csv(os.path.join(args.run, "top_best_losses.csv")).head(args.top)

    def show(df, title):
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        for _, row in df.iterrows():
            text = str(row.get(args.text_col, ""))[:200]
            print(
                f"[loss={row['loss']:.3f} conf={row['confidence']:.2f}] "
                f"true={row['true_label']!r} pred={row['pred_label']!r}\n  text: {text}\n"
            )

    show(worst, f"ТОП-{args.top} ХУДШИХ ПРЕДСКАЗАНИЙ (наибольший loss)")
    show(best, f"ТОП-{args.top} ЛУЧШИХ ПРЕДСКАЗАНИЙ (наименьший loss)")


if __name__ == "__main__":
    main()
