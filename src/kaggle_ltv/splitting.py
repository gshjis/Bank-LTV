"""Dataset splitting utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame, target: str, seed: int, test_size: float, val_size: float
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split a labeled dataframe into stratified train, validation and test sets."""
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test-size и val-size должны быть между 0 и 1")
    if test_size + val_size >= 1:
        raise ValueError("Сумма test-size и val-size должна быть меньше 1")

    x = df.drop(columns=[target])
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test  # type: ignore[return-value]
