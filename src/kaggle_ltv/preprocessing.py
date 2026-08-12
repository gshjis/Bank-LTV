"""Leakage-safe tabular preprocessing."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


class TabularPreprocessor:
    """Preprocessor whose learned statistics come from the training split only."""

    def __init__(self, drop_columns: list[str]) -> None:
        self.drop_columns: list[str] = drop_columns.copy()
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.numeric_medians: dict[str, float] = {}
        self.categorical_modes: dict[str, str] = {}

    @staticmethod
    def _normalize_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            result[column] = (
                result[column]
                .where(result[column].notna(), "__MISSING__")
                .astype(str)
                .str.lower()
            )
        return result

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        clean = df.drop(columns=self.drop_columns, errors="ignore").copy()
        self.categorical_columns = clean.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.numeric_columns = [
            column for column in clean.columns if column not in self.categorical_columns
        ]
        clean = self._normalize_categories(clean, self.categorical_columns)

        for column in self.numeric_columns:
            numeric_series = cast(
                pd.Series, pd.to_numeric(clean[column], errors="coerce")
            )
            median_value = float(numeric_series.median())
            self.numeric_medians[column] = median_value if median_value == median_value else 0.0
        for column in self.categorical_columns:
            mode = clean[column].mode(dropna=False)
            self.categorical_modes[column] = (
                str(mode.iloc[0]).lower() if not mode.empty else "__MISSING__"
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.drop(columns=self.drop_columns, errors="ignore").copy()
        expected = self.numeric_columns + self.categorical_columns
        missing = sorted(set(expected) - set(result.columns))
        if missing:
            raise ValueError(f"В данных отсутствуют признаки: {missing}")
        result = cast(pd.DataFrame, result[expected])
        result = self._normalize_categories(result, self.categorical_columns)
        for column in self.numeric_columns:
            numeric_series = cast(
                pd.Series, pd.to_numeric(result[column], errors="coerce")
            )
            result[column] = numeric_series.fillna(self.numeric_medians[column])
        for column in self.categorical_columns:
            result[column] = result[column].fillna(self.categorical_modes[column])
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "drop_columns": self.drop_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_medians": self.numeric_medians,
            "categorical_modes": self.categorical_modes,
        }
