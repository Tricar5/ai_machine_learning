from typing import Union, Optional
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


def business_metric(y_true, y_pred, tolerance: float = 0.1):
    # Считаем отклонение от реальной цены в абсолютных значениях
    relative_errors = np.abs((y_true - y_pred) / y_true)
    # По маске считаем долю
    within_tolerance = np.mean(relative_errors <= tolerance)

    return within_tolerance


def preprocess_col(feature_val: str) -> float:
    if isinstance(feature_val, str):
        try:
            return float(feature_val.split()[0])
        except ValueError:
            # Если нельзя сконвертировать - возвращаем np.nan
            return np.nan
    return feature_val


def remove_measurements(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].apply(preprocess_col).astype(float)
    return df


def convert_to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].astype(float)
    return df


class TokenExtractorTransformer(TransformerMixin):
    def __init__(
        self,
        source_col: str,
        target_col: str,
        token_pos: int = 1,
    ) -> None:
        self.src = source_col
        self.tgt = target_col
        self.pos = token_pos
        self.tokens = {"unknown"}
        self.is_fitted_ = False

    def fit(self, X, y=None) -> "TokenExtractorTransformer":
        for x in X[self.src].values:
            self.tokens.add(x.split()[self.pos])
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X[self.tgt] = X[self.src].apply(lambda x: self.extract_tokens(x))
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def extract_tokens(self, text: Optional[str]) -> Union[np.nan, str]:
        toks = text.split()
        for token in toks:
            if token in self.tokens:
                return token
        return "unknown"


class MyStateFullFillerTransformer(TransformerMixin):
    def __init__(
        self,
        strategy: str,
        cols: set[str],
    ) -> None:
        self.strategy = strategy
        self.cols = set(cols)
        self.state: dict[str, Union[float, int]] = {}

    def fit(
        self, X: pd.DataFrame, y=None, cols: Optional[list[str]] = None
    ) -> "MyStateFullFillerTransformer":
        if cols:
            self.cols = self.cols | set(cols)

        for col in self.cols:
            # если для колонки уже посчитано состояние (мы обновляем наш трансформер)
            if col in self.state:
                continue

            if self.strategy == "mean":
                self.state[col] = np.nanmean(X[col])
            elif self.strategy == "median":
                self.state[col] = np.nanmedian(X[col])

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            X[col] = X[col].fillna(value=self.state[col])
        return X

    def fit_transform(self, X, y=None, **fit_params) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, y)
