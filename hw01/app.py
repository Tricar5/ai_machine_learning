import pathlib
from typing import Final
from sklearn.metrics import mean_absolute_percentage_error

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from pkg import (
    TokenExtractorTransformer,
    MyStateFullFillerTransformer,
    remove_measurements,
    business_metric,
)

st.set_page_config(page_title="Auto Predictions", page_icon="🎯", layout="wide")

MODEL_PATH = pathlib.Path("hw01/model.pkl")

CAT_ANNOTATIONS = pathlib.Path("hw01/cat_features.json")
CATEGORICAL = {}


@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model


@st.cache_resource
def load_annotations():
    with open(CAT_ANNOTATIONS, "rb") as f:
        cat_annotations = json.load(f)
    return cat_annotations


FEATURE_NAMES: Final[list[str]] = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]


def prepare_features(
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    df = df[feature_names]
    measurable_cols = ["mileage", "engine", "max_power"]
    df = remove_measurements(df, measurable_cols)

    return df


# Загружаем модель
try:
    MODEL = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

try:
    CATEGORICAL = load_annotations()
except Exception as e:
    st.error(f"X Ошибка загрузки аннотаций к категориальным типам")

# --- Основной интерфейс ---
st.title("Предсказание цен на автомобили")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

try:
    prepared_df = prepare_features(df, FEATURE_NAMES)
    predictions = MODEL.predict(prepared_df)

    prepared_df["prediction"] = predictions
    prepared_df["selling_price"] = df["selling_price"]
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()

# --- Метрики ---
st.subheader("📊 Результаты")

col1, col2 = st.columns(2)
with col1:
    st.metric("Всего автомобилей", len(df))
with col2:
    mean_predicted_price = np.mean(prepared_df["prediction"])
    st.metric(f"Средняя предсказанная цена ", f"{mean_predicted_price:.2f}")

    if "selling_price" in prepared_df:
        positive_rate = (
            business_metric(
                prepared_df["prediction"],
                prepared_df["selling_price"],
            )
            * 100
        )
        st.metric(
            "Средний процент удовлетворенности предсказанной ценой",
            f"{positive_rate:.1f}%",
        )
        mape = mean_absolute_percentage_error(
            prepared_df["selling_price"], prepared_df["prediction"]
        )
        st.metric("Среднее отклонение по предсказанной цене", f"{mape:.1f}%")
# --- Визуализации ---
st.subheader("📈 Визуализации")

pred_counts = prepared_df["prediction"].value_counts().sort_index()

abs_coefs = [abs(coef) for coef in MODEL["estimator"].coef_]
features_out = MODEL["transformer"].get_feature_names_out()

top_20_indices = np.argsort(abs_coefs)[-20:]

fig = px.bar(
    x=[features_out[i] for i in top_20_indices],
    y=[MODEL["estimator"].coef_[i] for i in top_20_indices],
    title="Коэффициенты модели по фичам",
    barmode="group",
)

st.plotly_chart(fig, use_container_width=True)

# Реальные значения
prepared_df["error"] = (
    (prepared_df["selling_price"] - prepared_df["prediction"])
    / prepared_df["selling_price"]
) * 100
fig_err = px.histogram(prepared_df["error"].values)

# График остатков
st.plotly_chart(fig_err, use_container_width=True)

# --- Форма для предсказания ---
st.subheader("🔮 Сделать предсказание для своего автомобиля")

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)
    input_data = {}

    with col_left:
        st.write("**Категориальные:**")

        for col, unique_vals in CATEGORICAL.items():
            input_data[col] = st.selectbox(col, unique_vals, key=f"{col}")

    with col_right:
        st.write("**Числовые:**")
        for col in FEATURE_NAMES:
            if col in CATEGORICAL:
                continue
            if prepared_df[col].dtype not in ("object", "bool"):
                val = int(prepared_df[col].median())
                input_data[col] = st.number_input(col, value=val, key=f"{col}")

    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        prepared_input = prepare_features(input_df, FEATURE_NAMES)
        prediction = MODEL.predict(prepared_input)[0]

        st.success(f"**Предсказанная цена:** {prediction:.2f}")
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")
