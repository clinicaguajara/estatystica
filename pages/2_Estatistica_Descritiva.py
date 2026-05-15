# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit as st

from modules.descriptive_stats import describe_categorical_column, describe_numeric_column
from utils.dataframe_state import select_active_dataframe
from utils.design import load_css

# PAGE 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

# Título e instruções iniciais
st.title("Estatísticas Descritivas")

st.caption("""
A seção **Estatísticas Descritivas** fornece uma análise detalhada da distribuição de variáveis numéricas, incluindo medidas de **tendência central** —média, moda e mediana—, medidas de **dispersão** e **froma** —desvio padrão, amplitude, assimetria e curtose.
Também permite gerar gráficos interativos —histograma, boxplot e curvas de densidade.
""")

selected_df_name, df = select_active_dataframe(
    state_key="selected_df_name",
    label="Selecione o dataframe para análise:",
    widget_key="descriptive_selected_df",
)

st.divider()

# BODY ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# Controle do número de linhas com incremento nativo
st.write("### Descrição por coluna")
num_rows = st.number_input(
    "Número de linhas para inspeção visual:",
    min_value=5,
    max_value=100,
    value=5,
    step=5,
    format="%d",
)

# Visualização do dataframe selecionado
st.write(f"Visualizando as primeiras {num_rows} linhas de **{selected_df_name}**:")
st.dataframe(df.head(num_rows), use_container_width=True)

st.divider()
analysis_flow = st.radio(
    "Escolha o fluxo de análise:",
    ["Numérica", "Categórica"],
    horizontal=True,
    key="descriptive_analysis_flow",
)


if analysis_flow == "Numérica":
    describe_numeric_column(df, selected_df_name)
else:
    describe_categorical_column(df, selected_df_name)
