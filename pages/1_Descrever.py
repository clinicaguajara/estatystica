# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit   as st
import pandas      as pd
import scipy.stats as stats

from utils.design import load_css

# CUSTOM FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_normality(df: pd.DataFrame):
    """
    Executa testes de normalidade sobre variáveis numéricas selecionadas.
    """

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("Não há colunas numéricas no dataframe.")
        return

    selected_cols = st.multiselect("Selecione colunas para testar:", numeric_cols)
    test_options = st.multiselect(
        "Escolha os testes a serem aplicados:",
        ["Shapiro-Wilk", "Kolmogorov-Smirnov", "D’Agostino-Pearson", "Anderson-Darling"]
    )

    st.caption("Testes de normalidade realizados com [SciPy](https://docs.scipy.org/doc/scipy/) v.1.16.1")
    
    if selected_cols and test_options:
        for col in selected_cols:
            col_data = df[col].dropna()
            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"#### 📌 {col}")

            for test in test_options:
                if test == "Shapiro-Wilk":
                    stat, p = stats.shapiro(col_data)
                    st.write("##### 🟣 Shapiro-Wilk: ideal para < 5000 amostras.")
                elif test == "Kolmogorov-Smirnov":
                    stat, p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                    st.write("##### 🟣 Kolmogorov-Smirnov: compara com normal padrão.")
                elif test == "D’Agostino-Pearson":
                    stat, p = stats.normaltest(col_data)
                    st.write("##### 🟣 D’Agostino-Pearson: avalia simetria e curtose.")
                elif test == "Anderson-Darling":
                    result = stats.anderson(col_data, dist='norm')
                    st.write("##### 🟣 Anderson-Darling: fornece estatística crítica.")
                    st.write(f"Estatística: {result.statistic:.4f}")
                    for i in range(len(result.critical_values)):
                        st.write(f"Nível {result.significance_level[i]}% → valor crítico: {result.critical_values[i]}")
                    continue

                st.write(f"Estatística: `{stat:.4f}`, valor-p: `{p:.4g}`")
                if p < 0.05:
                    st.error("⛔ Rejeita normalidade (p < 0.05)")
                else:
                    st.success("✅ Distribuição compatível com normalidade (p ≥ 0.05)")
            

def describe_numeric_column(df: pd.DataFrame, df_name="selected_df_name"):
    """
    <docstrings>
    Exibe estatísticas descritivas e visualizações para uma coluna numérica selecionada de um DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a ser descrito.

    Calls:
        df.select_dtypes(): Seleciona colunas numéricas | método do DataFrame.
        st.selectbox(): Seleciona coluna a ser descrita | instanciado por st.
        st.table(): Exibe estatísticas como tabela | instanciado por st.
        plt.subplots(): Criação de gráficos | método do matplotlib.
        pd.Series.mode(): Retorna o(s) valor(es) mais frequentes | método de Series.
        pd.Series.quantile(): Calcula percentis da distribuição | método de Series.
        pd.Series.skew(): Calcula a assimetria | método de Series.
        pd.Series.kurtosis(): Calcula a curtose | método de Series.

    Raises:
        st.warning: Caso o dataframe não tenha colunas numéricas.
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import io

    # Verificação de integridade antes da renderização
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] == 0:
        st.warning(f"O dataframe '{df_name}' está vazio ou inválido.")
        st.stop()

    st.subheader("Descrição por coluna")

    # ───────────────────────────────────────────────────────
    # Verifica e seleciona coluna numérica
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("Este dataframe não possui colunas numéricas.")
        return

    selected_col = st.selectbox(f"Selecione uma coluna numérica para descrever em **{df_name}**:", numeric_cols)
    col_data = df[selected_col].dropna()

    # ───────────────────────────────────────────────────────
    # Estatísticas descritivas
    mode_value = col_data.mode()
    moda = mode_value.iloc[0] if not mode_value.empty else None

    q1 = col_data.quantile(0.25)
    q2 = col_data.quantile(0.50)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1

    # Tendência Central
    tendencia_central = {
        "Média": col_data.mean(),
        "Mediana": col_data.median(),
        "Moda": moda,
        "Q1 (25%)": q1,
        "Q2 (50%)": q2,
        "Q3 (75%)": q3,
    }

    # Dispersão e Forma
    dispersao = {
        "Desvio Padrão": col_data.std(),
        "Variância": col_data.var(),
        "IQR (Q3 - Q1)": iqr,
        "Amplitude": col_data.max() - col_data.min(),
        "Mínimo": col_data.min(),
        "Máximo": col_data.max(),
        "Assimetria (Skewness)": col_data.skew(),
        "Curtose": col_data.kurtosis(),
        "Valores Ausentes": df[selected_col].isna().sum(),
        "Valores Únicos": col_data.nunique()
    }

    # ───────────────────────────────────────────────────────
    # Visualização gráfica
    plot_type = st.radio("Escolha o tipo de gráfico:", [ "Curva de Densidade", "Histograma", "Boxplot"], horizontal=True)

    col_data_clean = col_data.dropna()
    dark_bg = "#0E1117"
    white = "#FFFFFF"
    purple = "#7159c1"

    # Gráfico modo escuro
    fig, ax = plt.subplots(facecolor=dark_bg)
    ax.set_facecolor(dark_bg)
    if plot_type == "Histograma":
            # Frequência por valor único (exato, sem bins)
            valores_unicos = sorted(col_data_clean.unique())
            counts = col_data_clean.value_counts().sort_index()

            bars = ax.bar(valores_unicos, counts, color=purple, edgecolor=white, width=0.6)

            ax.set_title(f"Histograma de {selected_col}", color=white)
            ax.set_xlabel(selected_col, color=white)
            ax.set_ylabel("Frequência", color=white)
            ax.tick_params(colors=white)
            # Anota frequências em cada barra
            for rect in bars:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, height, int(height),
                        ha='center', va='bottom', color=white)


    elif plot_type == "Boxplot":
        ax.boxplot(
            col_data_clean, vert=False,
            boxprops=dict(color=white),
            capprops=dict(color=white),
            whiskerprops=dict(color=white),
            flierprops=dict(markeredgecolor=purple),
            medianprops=dict(color=white)
        )
        ax.plot(col_data_clean.mean(), 1, 'o', color=purple, label='Média')
        ax.set_xlabel(selected_col, color=white)

    elif plot_type == "Curva de Densidade":
        col_data_clean.plot(kind='density', ax=ax, color=purple)
        ax.set_xlabel(selected_col, color=white)
        ax.set_ylabel("Densidade", color=white)

    ax.tick_params(colors=white)
    for spine in ax.spines.values():
        spine.set_edgecolor(white)

    plt.tight_layout()
    st.pyplot(fig)

    # ───────────────────────────────────────────────────────
    # Download dos gráficos

    dark_buf = io.BytesIO()
    fig.savefig(dark_buf, format="png")
    dark_buf.seek(0)

    plt.style.use("default")
    light_fig, light_ax = plt.subplots(facecolor="white")
    light_ax.set_facecolor("white")

    if plot_type == "Histograma":
        light_ax.hist(col_data_clean, bins=20, color=purple, edgecolor="black")
        light_ax.set_title(f"Histograma de {selected_col}")
        light_ax.set_xlabel(selected_col)
        light_ax.set_ylabel("Frequência")

    elif plot_type == "Boxplot":
        light_ax.boxplot(
            col_data_clean, vert=False,
            boxprops=dict(color="black"),
            capprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            flierprops=dict(markeredgecolor=purple),
            medianprops=dict(color="black")
        )
        light_ax.set_title(f"Boxplot de {selected_col}")
        light_ax.set_xlabel(selected_col)

    elif plot_type == "Curva de Densidade":
        col_data_clean.plot(kind='density', ax=light_ax, color=purple)
        light_ax.set_title(f"Curva de Densidade de {selected_col}")
        light_ax.set_xlabel(selected_col)

    light_fig.tight_layout()
    light_buf = io.BytesIO()
    light_fig.savefig(light_buf, format="png", facecolor="white")
    light_buf.seek(0)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download (tema escuro)",
            data=dark_buf,
            file_name=f"{selected_col}_{plot_type.lower().replace(' ', '_')}_dark.png",
            mime="image/png",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📥 Download (tema claro)",
            data=light_buf,
            file_name=f"{selected_col}_{plot_type.lower().replace(' ', '_')}_light.png",
            mime="image/png",
            use_container_width=True
        )

    # McKinney, 2010
    st.info(
        """**[W. McKinney. *Data Structures for Statistical Computing in Python* (2010)](https://proceedings.scipy.org/articles/Majora-92bf1922-00a.pdf)**  
    \nO autor argumenta que a integração de pandas com NumPy, SciPy, Matplotlib e outras bibliotecas científicas torna o Python uma opção cada vez mais atraente para análise de dados estatísticos, especialmente em comparação com R. O artigo aponta a evolução futura da biblioteca e seu papel central em um ecossistema de modelagem estatística em Python.
    """,
        icon="📜"
    )

    # Renderiza as tabelas
    st.write("### Tendência central")
    st.caption("Métricas que resumem a localização dos dados na distribuição.")
    st.table(pd.DataFrame(tendencia_central.items(), columns=["Estatística", "Valor"]))

    st.write("### Dispersão e forma")
    st.caption("Indicadores de variabilidade, amplitude e o formato da distribuição.")
    st.table(pd.DataFrame(dispersao.items(), columns=["Estatística", "Valor"]))

    st.caption("Powered by [Pandas](https://pandas.pydata.org/docs/) v.2.3.1, [Streamlit](https://docs.streamlit.io/) v.1.35.0, [Matplotlib](https://matplotlib.org/stable/index.html) v3.10.5")

# PAGE 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

# Título e instruções iniciais
st.title("Estatísticas Descritivas")

st.caption("""
A seção **Estatísticas Descritivas** fornece uma análise detalhada da distribuição de variáveis numéricas, incluindo medidas de **tendência central** (média, mediana, moda), **dispersão** (desvio padrão, IQR, amplitude) e **forma** da distribuição (assimetria e curtose). 
Também permite gerar gráficos interativos (histograma, boxplot e curva de densidade). 
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


selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names, index=df_names.index(selected_df_name))
df = st.session_state.dataframes[selected_df_name]
st.write(f"**Dimensões:** {df.shape[0]} × {df.shape[1]}")

st.divider()

# Controle do número de linhas com incremento nativo
st.write("### Inspeção visual")
num_rows = st.number_input(
    "Número de linhas para inspeção visual:",
    min_value=5,
    max_value=100,
    value=5,
    step=5,
    format="%d"
)

# Visualização do dataframe selecionado
st.write(f"Visualizando as primeiras {num_rows} linhas de **{selected_df_name}**:")
st.dataframe(df.head(num_rows), use_container_width=True)

describe_numeric_column(df, selected_df_name)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
        st.stop()

st.write("### Normalidade")
st.caption("""
Além disso, estão disponíveis testes clássicos de **normalidade** — como Shapiro-Wilk, Kolmogorov-Smirnov e D’Agostino-Pearson — para verificar se os dados seguem uma distribuição normal. 
Ideal para exploração inicial de dados, identificação de padrões e avaliação da adequação para testes estatísticos posteriores.
""")

with st.expander("Executar testes de normalidade"):
    st.markdown("<br>", unsafe_allow_html=True)
    test_normality(df)
    st.markdown("<br>", unsafe_allow_html=True)
