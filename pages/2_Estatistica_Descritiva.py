# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit as st

from modules.descriptive_stats import describe_categorical_column, describe_numeric_column
from utils.design import load_css

# PAGE 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

# Título e instruções iniciais
st.title("Estatísticas Descritivas")

st.caption("""
A seção **Estatísticas Descritivas** fornece uma análise detalhada da distribuição de variáveis numéricas, incluindo medidas de **tendência central** —média, moda e mediana—, medidas de **dispersão** e **froma** —desvio padrão, amplitude, assimetria e curtose.
Também permite gerar gráficos interativos —histograma, boxplot e curvas de densidade.
""")

# Verify dataframe
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Nenhum dataframe carregado.")
    st.stop()

# Seleção do dataframe para visualização
df_names = list(st.session_state.dataframes.keys())

if not df_names:
    st.warning("Nenhum dataframe disponível.")
    st.stop()

selected_df_name = st.session_state.get("selected_df_name")

if selected_df_name not in df_names:
    selected_df_name = df_names[0]

selected_df_name = st.selectbox(
    "Selecione o dataframe para análise:",
    df_names,
    index=df_names.index(selected_df_name),
)
st.session_state["selected_df_name"] = selected_df_name

df = st.session_state.dataframes[selected_df_name]

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
