# BERT text classification — сравнение BERT и его модификаций

Задача: классификация тональности твитов про covid, 5 классов
(`Extremely Negative`, `Negative`, `Neutral`, `Positive`, `Extremely Positive`).
Датасет — [Coronavirus tweets NLP](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)
(`Corona_NLP_train.csv` / `Corona_NLP_test.csv`), тот же, что используется в
[исходном ноутбуке](https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert).

## Идея пайплайна

Один и тот же код (`src/pipeline.py`) используется для **всех** моделей: и для
бейзлайна (`bert-base-uncased`), и для 4 модификаций. Меняется только чекпоинт
в `transformers.AutoModel` — голова классификации, пулинг, цикл обучения,
метрики и анализ ошибок везде одинаковые. Это важно для честного сравнения:
разница в метриках объясняется архитектурой backbone, а не случайными
отличиями в коде.

Голова — своя, не встроенная: берём `AutoModel` (а не `AutoModelForSequenceClassification`),
то есть исходная pretraining-голова (MLM/NSP/RTD/SOP и т.п.) отброшена, поверх
[CLS]-представления навешен `Dropout -> Linear(hidden_size, num_labels)`.

## Шаг 1. Бейзлайн: заморозка vs полный fine-tune

```bash
python scripts/sweep_baseline.py --config configs/config.yaml
```

Что делает:
- Обучает `bert-base-uncased` в двух режимах — `frozen_backbone` (учится только
  голова) и `full_finetune` (учится всё) — коротким циклом (`sweep.epochs`).
- В каждом режиме перебирает несколько `lr` и `batch_size` (см. `sweep:` в
  `config.yaml`). Диапазоны lr разные не случайно: у замороженного backbone
  учится только голова, ей обычно нужен lr на 1–2 порядка выше, чем при
  полном fine-tune всей модели (иначе голова просто не успеет обучиться).
- Использует `AdamW` с раздельными lr для backbone и головы и без weight decay
  на `bias`/`LayerNorm` (стандартная практика для BERT).
- Каждый прогон логируется в Comet отдельным экспериментом (теги `sweep`,
  `baseline`, режим).
- В конце печатает сравнение лучшего `frozen_backbone` и лучшего
  `full_finetune` по `val_f1_macro` и объясняет, какой режим выбран в
  качестве бейзлайна.
- Сохраняет `outputs/sweep_results.csv` (все прогоны) и
  `outputs/best_baseline_config.yaml` (победившая связка freeze/lr/batch_size)
  — их подхватит следующий шаг.

Почему `f1_macro`, а не `accuracy`: классы в датасете распределены неравномерно
(`Positive`/`Negative` встречаются намного чаще `Extremely Positive`), а
macro-усреднение не даёт мажоритарным классам "спрятать" низкое качество на
редких классах.

## Шаг 2. Бейзлайн + 4 модификации BERT

```bash
python scripts/run_all_models.py --config configs/config.yaml
```

Обучает `bert-base-uncased` (с гиперпараметрами из шага 1) и 4 модификации на
полном числе эпох (`training.epochs`), строит лидерборд
(`outputs/leaderboard.csv`), печатает лучшую модель. Для каждой модели в
`outputs/<model>_<mode>/` сохраняются:
- `test_metrics.json` — полный набор метрик (см. ниже)
- `top_worst_losses.csv` / `top_best_losses.csv` — тексты, на которых модель
  ошибается больше/меньше всего
- `per_sample_errors.csv` — предсказания по всем тестовым примерам
- `model.pt` — веса модели

Посмотреть топ ошибок человекочитаемо:

```bash
python scripts/inspect_errors.py --run outputs/BERT-base_finetune --top 15
```

### Модификации BERT — что выбрано и почему

Выбраны 4 модели, которые чаще всего фигурируют в сравнительных работах про
BERT и у которых наибольшее число цитирований среди производных BERT в
литературе:

| Модель | Что изменили относительно BERT |
|---|---|
| **RoBERTa** (Liu et al., 2019) | Убрали задачу NSP (next sentence prediction) как бесполезную, обучали дольше и на существенно большем корпусе, с большими батчами, динамическим (не статическим) маскированием токенов при каждом проходе по данным, byte-level BPE токенизация. По сути — "тот же BERT, но с гораздо более тщательным rецептом претрейна". |
| **DistilBERT** (Sanh et al., 2019) | Дистилляция: маленькая модель (6 слоёв вместо 12) обучается имитировать выходы полного BERT (soft targets + классический MLM loss + cosine embedding loss). На выходе ~40% меньше параметров, ~60% быстрее инференс, при этом сохраняет заявлено около 97% качества BERT на GLUE. |
| **ALBERT** (Lan et al., 2020) | Две вещи для уменьшения числа параметров: (1) факторизация embedding-матрицы (embedding size отделён от hidden size), (2) shared parameters между всеми слоями трансформера (cross-layer parameter sharing). Также заменили NSP на SOP (sentence order prediction) — сложнее и полезнее задачу. |
| **ELECTRA** (Clark et al., 2020) | Вместо MLM (предсказать замаскированный токен) — replaced token detection: маленький генератор подменяет часть токенов правдоподобными, а основная модель (дискриминатор) учится определять, какие токены настоящие, а какие подменены. Обучающий сигнал получается с *каждого* токена, а не только с ~15% замаскированных — заметно эффективнее по компьютерным ресурсам при той же выборке данных. |

Хотелось бы также отметить: точное число цитирований со временем меняется,
но именно этот набор (RoBERTa/DistilBERT/ALBERT/ELECTRA) — стандартный выбор
в сравнительных статьях про BERT-архитектуры, что подтверждает их значимость
в области.

Список чекпоинтов и их HF-имена — в `configs/config.yaml` (`models:`). Можно
добавить/заменить модификацию, просто дописав пункт в `models.modifications`
(достаточно, чтобы у чекпоинта был выход `last_hidden_state` — подходит любой
encoder-only трансформер из `AutoModel`).

## Метрики (максимальный набор для классификации)

Для каждой модели считается (`src/metrics.py`):
- `accuracy`
- `precision` / `recall` / `f1` — macro, micro, weighted
- `precision` / `recall` / `f1` — по каждому классу отдельно (`per_class`)
- `Matthews correlation coefficient (MCC)` — устойчива к дисбалансу классов
- `Cohen's kappa`
- `log loss`
- `ROC-AUC` (one-vs-rest, macro)
- `confusion matrix`

Confusion matrix и таблицы top/bottom ошибок автоматически логируются в Comet
(`Confusion Matrix` и `Assets` в UI эксперимента).

## Структура репозитория

```
configs/config.yaml       # все гиперпараметры, comet-настройки, список моделей
requirements.txt
src/
  data.py                 # загрузка csv, очистка текста, train/val/test сплит, Dataset
  model.py                # AutoModel + своя голова, заморозка backbone, param groups для AdamW
  metrics.py               # весь набор метрик классификации
  comet_utils.py            # обёртка над comet_ml (+ заглушка, если ключ не задан)
  pipeline.py               # обучение + eval + error-анализ одного запуска (используется везде)
  utils.py                 # сиды, конфиг, device
scripts/
  sweep_baseline.py         # шаг 1: freeze vs full fine-tune, подбор lr/batch_size
  run_all_models.py          # шаг 2: baseline + 4 модификации, лидерборд
  inspect_errors.py           # человекочитаемый просмотр top/bottom ошибок
outputs/                      # результаты прогонов (в git не попадает, см. .gitignore)
```

## Как запускать на Kaggle

1. Закоммить репозиторий в свой GitHub (см. ниже).
2. В `configs/config.yaml` пропиши:
   - `comet.api_key` — свой ключ с [comet.com/account-settings](https://www.comet.com/account-settings)
   - `comet.workspace` — имя своего воркспейса в Comet
   - `data.train_path` / `data.test_path` — под то, как Kaggle смонтирует датасет
     (подключи датасет через **Add Data** → *Covid-19 NLP Text Classification*
     (datatattle), путь будет вида `/kaggle/input/<slug>/Corona_NLP_train.csv`)
3. В новом Kaggle-ноутбуке (GPU-акселератор включён, интернет включён —
   Settings → Internet → On):

```python
!git clone https://github.com/<your_username>/<your_repo>.git
%cd <your_repo>
!pip install -q -r requirements.txt

!python scripts/sweep_baseline.py --config configs/config.yaml
!python scripts/run_all_models.py --config configs/config.yaml
```

4. Дальше смотри метрики/сравнение прямо в выводе ячейки, полную картину — в
   Comet (confusion matrix, кривые обучения по эпохам, таблицы ошибок), и
   таблицы `outputs/sweep_results.csv` / `outputs/leaderboard.csv` в файлах
   ноутбука.

### Как закоммитить в репозиторий

```bash
cd bert-text-classification
git init                          # если ещё не инициализирован
git add .
git commit -m "BERT + 4 модификации: baseline sweep, обучение, метрики, error-анализ"
git remote add origin https://github.com/<your_username>/<your_repo>.git
git push -u origin main
```

Не забудь: `configs/config.yaml` с реальным `comet.api_key` **не стоит**
пушить в публичный репозиторий как есть. Варианты: (а) держать репозиторий
приватным, (б) вынести ключ в переменную окружения и прочитать его в
`src/comet_utils.py` через `os.environ`, (в) хранить `config.yaml` с ключом
только локально/на Kaggle (через Kaggle Secrets), а в репозитории — версию с
плейсхолдером `YOUR_COMET_API_KEY` (как сейчас).

## Что можно покрутить дальше

- `data.max_length` — больше 128 может улучшить качество на длинных твитах, но
  дороже по памяти/времени.
- `head.hidden_dim` — попробовать голову с одним скрытым слоем вместо линейной.
- `training.epochs` / `early_stopping_patience` — сейчас настроено под
  разумный баланс скорости и качества для Kaggle-сессии.
- Добавить ещё модификации (DeBERTa, XLNet, MobileBERT) — просто дописав их в
  `models.modifications` в конфиге.
