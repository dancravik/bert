import os

try:
    import comet_ml
except ImportError:
    comet_ml = None


class _DummyExperiment:
    """Заглушка: если ключ не задан или comet_ml не установлен, весь код работает
    как обычно, просто ничего никуда не логируется (и явно печатается warning)."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def build_experiment(cfg: dict, name: str, tags=None):
    comet_cfg = cfg.get("comet", {})

    if comet_cfg.get("disabled", False):
        print("[comet] логирование отключено в config.yaml (comet.disabled=true)")
        return _DummyExperiment()

    if comet_ml is None:
        print("[comet] пакет comet_ml не установлен -> логирование отключено")
        return _DummyExperiment()

    # Приоритет: переменная окружения COMET_API_KEY (удобно и безопаснее для публичного
    # репозитория) -> значение из config.yaml.
    api_key = os.environ.get("COMET_API_KEY") or comet_cfg.get("api_key")
    if not api_key or api_key == "YOUR_COMET_API_KEY":
        print(
            "[comet] api_key не задан (ни COMET_API_KEY, ни configs/config.yaml) "
            "-> логирование отключено"
        )
        return _DummyExperiment()

    experiment = comet_ml.Experiment(
        api_key=api_key,
        project_name=comet_cfg.get("project_name", "bert-text-classification"),
        workspace=comet_cfg.get("workspace"),
        auto_metric_logging=False,
        auto_param_logging=False,
    )
    experiment.set_name(name)
    if tags:
        experiment.add_tags(tags)
    return experiment


def log_confusion_matrix(experiment, y_true, y_pred, labels):
    try:
        experiment.log_confusion_matrix(
            y_true=[int(v) for v in y_true],
            y_predicted=[int(v) for v in y_pred],
            labels=labels,
        )
    except Exception as e:  # логирование не должно ронять обучение
        print(f"[comet] не удалось залогировать confusion matrix: {e}")


def log_error_table(experiment, df, name: str):
    try:
        experiment.log_table(f"{name}.csv", tabular_data=df, headers=list(df.columns))
    except Exception as e:
        print(f"[comet] не удалось залогировать таблицу {name}: {e}")
