import numpy as np
import pandas as pd
import pytest

from kaggle_ltv import (
    TabularPreprocessor,
    calculate_metrics,
    choose_threshold,
    flatten_metrics,
    split_data,
)


def test_preprocessor_fits_statistics_only_on_train_and_normalizes_categories():
    train = pd.DataFrame(
        {
            "client_id": [1, 2, 3],
            "amount": [10.0, np.nan, 30.0],
            "city": ["MOSCOW", None, "Moscow"],
        }
    )
    validation = pd.DataFrame(
        {"client_id": [4], "amount": [1000.0], "city": [None]}
    )

    preprocessor = TabularPreprocessor(["client_id"]).fit(train)
    transformed = preprocessor.transform(validation)

    assert preprocessor.numeric_medians["amount"] == 20.0
    assert transformed.loc[0, "amount"] == 1000.0
    assert transformed.loc[0, "city"] == "__missing__"
    assert not bool(transformed.isna().to_numpy().any())
    assert "client_id" not in transformed.columns
    assert train.loc[0, "city"] == "MOSCOW"


def test_split_data_is_reproducible_and_stratified():
    df = pd.DataFrame(
        {"feature": range(100), "target": [0] * 80 + [1] * 20},
        index=range(1000, 1100),
    )
    first = split_data(df, "target", seed=42, test_size=0.2, val_size=0.2)
    second = split_data(df, "target", seed=42, test_size=0.2, val_size=0.2)

    for first_part, second_part in zip(first, second):
        if isinstance(first_part, pd.DataFrame):
            pd.testing.assert_frame_equal(first_part, second_part)
        else:
            pd.testing.assert_series_equal(first_part, second_part)
    assert set(first[0].index).isdisjoint(first[1].index)
    assert set(first[0].index).isdisjoint(first[2].index)
    assert set(first[1].index).isdisjoint(first[2].index)
    assert first[3].mean() == pytest.approx(0.2, abs=0.02)


def test_threshold_and_metrics_on_perfect_example():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = choose_threshold(y_true, probabilities)
    metrics = calculate_metrics(y_true, probabilities, threshold)

    assert 0.05 <= threshold <= 0.95
    assert metrics["classification_report"]["1"]["f1-score"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_flatten_metrics_converts_nested_report():
    nested = {
        "validation": {
            "threshold": 0.5,
            "roc_auc": 0.8,
            "classification_report": {
                "1": {"precision": 0.7, "recall": 0.6, "f1-score": 0.65}
            },
        }
    }
    flattened = flatten_metrics(nested)

    assert flattened["validation_threshold"] == 0.5
    assert flattened["validation_roc_auc"] == 0.8
    assert flattened["validation_classification_report_1_f1-score"] == 0.65


def test_split_data_rejects_invalid_sizes():
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
    with pytest.raises(ValueError):
        split_data(df, "target", seed=42, test_size=0.8, val_size=0.3)
