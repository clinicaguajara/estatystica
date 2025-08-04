# PAGE 4 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from boruta import BorutaPy

st.set_page_config(page_title="Seleção de Variáveis - Boruta", layout="centered")
st.title("Seleção de Variáveis com Boruta")

st.caption("""
O algoritmo **Boruta** é uma técnica robusta de *feature selection* baseada em florestas aleatórias.
Ele compara a importância das variáveis reais com versões aleatórias (chamadas *shadow features*) e seleciona apenas aquelas que são consistentemente mais importantes que o acaso.
Ideal para reduzir dimensionalidade sem perder variáveis relevantes para o modelo.
""")

# Verifica se há dataframes carregados
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Nenhum dataframe carregado.")
    st.stop()

# Seleção do dataframe
df_name = st.selectbox("Selecione o dataframe para análise:", list(st.session_state.dataframes.keys()))
df = st.session_state.dataframes[df_name]

# Seleção da variável alvo
target_column = st.selectbox("Selecione a variável alvo (y):", options=df.select_dtypes(include='number').columns)

# Parâmetros ajustáveis
with st.expander("⚙️ Ajustar parâmetros do modelo", expanded=False):
    st.markdown("""
    **Sobre os parâmetros:**
    - `max_depth`: controla a profundidade máxima das árvores na floresta. Valores maiores permitem modelos mais complexos, mas com maior risco de overfitting.
    - `n_estimators`: número de árvores de decisão construídas no Random Forest. Mais árvores aumentam a robustez, mas também o tempo de execução.
    - `random_state`: define a semente aleatória para garantir reprodutibilidade dos resultados. Use o mesmo valor para obter os mesmos resultados em execuções futuras.
    """)
    
    max_depth = st.slider("Profundidade máxima da árvore (max_depth)", 2, 30, value=5)
    n_estimators = st.number_input("Número de árvores (n_estimators)", min_value=50, max_value=1000, value=300, step=50)
    random_state = st.number_input("Semente aleatória (random_state)", min_value=0, value=42, step=1)

# Tipo de análise
analysis_type = st.radio(
    "Como deseja tratar a variável alvo?",
    options=["Regressão contínua", "Classificação"],
    horizontal=True
)

if analysis_type == "Classificação":
    discretization_method = st.radio(
        "Método de discretização:",
        options=[
            "Quantis balanceados (qcut)",
            "Intervalos fixos (cut)",
            "Ponto de corte teórico (>= limiar)"
        ]
    )

    if discretization_method == "Ponto de corte teórico (>= limiar)":
        threshold = st.number_input(
            "Valor de corte (ex: 36 significa que y ≥ 36 será classificado como 'alto')",
            value=36,
            help="Esse valor será usado para separar os grupos: 'baixo' (< limiar) e 'alto' (≥ limiar)"
        )


# Executar Boruta
if st.button("🚀 Executar Boruta", use_container_width=True):
    try:
        df_clean = df.dropna()
        X = df_clean.drop(columns=[target_column]).select_dtypes(include='number')

        # Define y e rf de acordo com o tipo de análise
        if analysis_type == "Classificação":
            # Discretização
            if discretization_method == "Quantis balanceados (qcut)":
                y = pd.qcut(df_clean[target_column], q=3, labels=["baixo", "médio", "alto"])
            elif discretization_method == "Intervalos fixos (cut)":
                y = pd.cut(df_clean[target_column], bins=3, labels=["baixo", "médio", "alto"])
            else:  # Ponto de corte
                y = df_clean[target_column].apply(lambda x: "baixo" if x < threshold else "alto")

            # Classificador
            rf = RandomForestClassifier(
                n_jobs=-1,
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=random_state
            )

        else:
            # Regressão contínua
            y = df_clean[target_column].values
            rf = RandomForestRegressor(
                n_jobs=-1,
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=random_state
            )

        # Aplica Boruta
        boruta = BorutaPy(estimator=rf, n_estimators='auto', verbose=1, random_state=random_state)
        boruta.fit(X.values, y)

        # Formata resultados
        selecionada_label = ["✅ Sim" if sel else "❌ Não" for sel in boruta.support_]
        result_df = pd.DataFrame({
            "Variável": X.columns,
            "Selecionada": selecionada_label,
            "Ranking": boruta.ranking_
        }).sort_values("Ranking")

        st.success("Análise finalizada com sucesso ✅")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar resultados (.csv)", data=csv,
                           file_name="boruta_resultados.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Erro ao executar Boruta: {e}")