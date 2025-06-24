# ───────────────────────────────────────────────────────────
# REQUIRED IMPORTS
# ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import io
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from utils.design import load_css


# ───────────────────────────────────────────────────────────
# FUNÇÃO 1 — Testes de Normalidade
# ───────────────────────────────────────────────────────────



def test_normality(df: pd.DataFrame):
    """
    Executa testes de normalidade sobre variáveis numéricas selecionadas.
    """

    st.subheader("Teste de Normalidade")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("Não há colunas numéricas no dataframe.")
        return

    selected_cols = st.multiselect("Selecione colunas para testar:", numeric_cols)
    test_options = st.multiselect(
        "Escolha os testes a serem aplicados:",
        ["Shapiro-Wilk", "Kolmogorov-Smirnov", "D’Agostino-Pearson", "Anderson-Darling"]
    )

    if selected_cols and test_options:
        for col in selected_cols:
            col_data = df[col].dropna()
            st.markdown(f"**📌 Coluna: `{col}`**")

            for test in test_options:
                if test == "Shapiro-Wilk":
                    stat, p = stats.shapiro(col_data)
                    st.caption("Shapiro-Wilk: ideal para < 5000 amostras.")
                elif test == "Kolmogorov-Smirnov":
                    stat, p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                    st.caption("Kolmogorov-Smirnov: compara com normal padrão.")
                elif test == "D’Agostino-Pearson":
                    stat, p = stats.normaltest(col_data)
                    st.caption("D’Agostino-Pearson: avalia simetria e curtose.")
                elif test == "Anderson-Darling":
                    result = stats.anderson(col_data, dist='norm')
                    st.caption("Anderson-Darling: fornece estatística crítica.")
                    st.write(f"Estatística: {result.statistic:.4f}")
                    for i in range(len(result.critical_values)):
                        st.write(f"Nível {result.significance_level[i]}% → valor crítico: {result.critical_values[i]}")
                    continue

                st.write(f"→ Estatística: `{stat:.4f}`, valor-p: `{p:.4g}`")
                if p < 0.05:
                    st.error("🚫 Rejeita normalidade (p < 0.05)")
                else:
                    st.success("✅ Distribuição compatível com normalidade (p ≥ 0.05)")

            st.divider()

# ───────────────────────────────────────────────────────────
# FUNÇÃO 2 — Análise de Correlação
# ───────────────────────────────────────────────────────────
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import io

def correlation_analysis(df: pd.DataFrame):
    """
    Calcula e exibe uma matriz de correlação com coeficientes e valores-p combinados,
    além de um heatmap visual com download em tema escuro e claro.
    """

    st.subheader("🔗 Correlação entre Variáveis")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("É necessário pelo menos duas variáveis numéricas.")
        return

    selected_cols = st.multiselect("Variáveis para matriz de correlação:", numeric_cols, default=numeric_cols)

    if selected_cols:
        method = st.radio("Método:", ["Pearson", "Spearman", "Kendall"], horizontal=True)

        # Inicializa matrizes
        matrix_display = pd.DataFrame(index=selected_cols, columns=selected_cols)
        matrix_r = pd.DataFrame(index=selected_cols, columns=selected_cols, dtype=float)

        for col1 in selected_cols:
            for col2 in selected_cols:
                x = df[col1].dropna()
                y = df[col2].dropna()
                x, y = x.align(y, join="inner")

                if method == "Pearson":
                    r, p = pearsonr(x, y)
                elif method == "Spearman":
                    r, p = spearmanr(x, y)
                elif method == "Kendall":
                    r, p = kendalltau(x, y)

                matrix_r.loc[col1, col2] = r
                matrix_display.loc[col1, col2] = f"{r:.2f} (p={p:.4g})"

        # ─────────────────────────────────────────────────────
        # Tabela formatada
        st.dataframe(matrix_display, use_container_width=True)

        # ─────────────────────────────────────────────────────
        # Heatmap com tema escuro
        with st.expander("Heatmap das Correlações"):
            
            st.markdown("<br>", unsafe_allow_html=True)

            dark_bg = "#0E1117"
            fig_dark, ax_dark = plt.subplots(figsize=(6, 4), facecolor=dark_bg)
            ax_dark.set_facecolor(dark_bg)
            
            heatmap = sns.heatmap(
                matrix_r.astype(float),
                annot=True, fmt=".2f", cmap="magma",
                cbar=True, square=True, linewidths=0.5,
                annot_kws={"color": "white", "size": 9},
                ax=ax_dark
            )

            # Título e eixos escuros
            ax_dark.set_title("Matriz de Correlação", color="white")
            ax_dark.tick_params(colors="white")
            for spine in ax_dark.spines.values():
                spine.set_edgecolor("white")

            # Corrigir colorbar (barra lateral)
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(colorbar.ax.yaxis.get_majorticklabels(), color='white')


            plt.tight_layout()
            st.pyplot(fig_dark)

            # Buffer para download (tema escuro)
            dark_buf = io.BytesIO()
            fig_dark.savefig(dark_buf, format="png", facecolor=dark_bg)
            dark_buf.seek(0)

            # Heatmap tema claro
            fig_light, ax_light = plt.subplots(figsize=(6, 4), facecolor="white")
            ax_light.set_facecolor("white")

            sns.heatmap(
                matrix_r.astype(float),
                annot=True, fmt=".2f", cmap="coolwarm",
                cbar=True, square=True, linewidths=0.5,
                annot_kws={"color": "white", "size": 9},
                ax=ax_light
            )
            ax_light.set_title("Matriz de Correlação")
            plt.tight_layout()

            light_buf = io.BytesIO()
            fig_light.savefig(light_buf, format="png", facecolor="white")
            light_buf.seek(0)

            # Botões de download
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Baixar Heatmap (Tema Escuro)",
                    data=dark_buf,
                    file_name="heatmap_correlacao_dark.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📥 Baixar Heatmap (Tema Claro)",
                    data=light_buf,
                    file_name="heatmap_correlacao_light.png",
                    mime="image/png",
                    use_container_width=True
                )


# ───────────────────────────────────────────────────────────
# FUNÇÃO 3 — Gráfico de Dispersão
# ───────────────────────────────────────────────────────────
def scatter_visualizer(df: pd.DataFrame):
    """
    Exibe gráfico de dispersão entre duas variáveis numéricas, com opção de download nos temas claro e escuro.
    """

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("É necessário pelo menos duas variáveis numéricas.")
        return

    col1 = st.selectbox("Eixo X:", numeric_cols, key="scatter_x")
    col2 = st.selectbox("Eixo Y:", [c for c in numeric_cols if c != col1], key="scatter_y")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ───────────────────────────────────────────────────────
    # Gráfico modo escuro
    dark_bg = "#0E1117"
    white = "#FFFFFF"
    purple = "#7159c1"

    fig, ax = plt.subplots(facecolor=dark_bg)
    ax.set_facecolor(dark_bg)

    ax.scatter(df[col1], df[col2], color=purple, alpha=0.7, edgecolors=white)
    ax.set_xlabel(col1, color=white)
    ax.set_ylabel(col2, color=white)

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

    # Tema claro
    plt.style.use("default")
    light_fig, light_ax = plt.subplots(facecolor="white")
    light_ax.set_facecolor("white")
    light_ax.scatter(df[col1], df[col2], color=purple, alpha=0.7, edgecolors="white")
    light_ax.set_title(f"{col2} vs {col1}")
    light_ax.set_xlabel(col1)
    light_ax.set_ylabel(col2)

    light_fig.tight_layout()
    light_buf = io.BytesIO()
    light_fig.savefig(light_buf, format="png", facecolor="white")
    light_buf.seek(0)

    # ───────────────────────────────────────────────────────
    # Botões de download
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Baixar Gráfico (Tema Escuro)",
            data=dark_buf,
            file_name=f"{col2}_vs_{col1}_dark.png",
            mime="image/png",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label="📥 Baixar Gráfico (Tema Claro)",
            data=light_buf,
            file_name=f"{col2}_vs_{col1}_light.png",
            mime="image/png",
            use_container_width=True
        )



# ───────────────────────────────────────────────────────────
# FUNÇÃO 4 — Regressões
# ───────────────────────────────────────────────────────────

def linear_regression_analysis(df: pd.DataFrame):
    """
    Realiza regressão linear simples entre duas variáveis numéricas, com visualização gráfica e download.
    """

    st.header("📈 Regressão Linear Simples")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("É necessário pelo menos duas variáveis numéricas.")
        return

    col_x = st.selectbox("Variável Independente (X):", numeric_cols, key="reg_x")
    col_y = st.selectbox("Variável Dependente (Y):", [c for c in numeric_cols if c != col_x], key="reg_y")

    x = df[col_x].dropna()
    y = df[col_y].dropna()
    x, y = x.align(y, join="inner")

    X_const = sm.add_constant(x)  # Adiciona intercepto
    model = sm.OLS(y, X_const).fit()

    intercept = model.params['const']
    slope = model.params[col_x]
    r_squared = model.rsquared
    p_value = model.pvalues[col_x]

    st.markdown(f"**Equação:** `{col_y} = {intercept:.3f} + {slope:.3f} × {col_x}`")
    st.markdown(f"**R²:** `{r_squared:.3f}`")
    st.markdown(f"**Valor-p (coef. X):** `{p_value:.4g}`")
    
    st.subheader("Linha de Regressão")

    # ───────────────────────────────────────────────────────
    # Gráfico com tema escuro
    dark_bg = "#0E1117"
    white = "#FFFFFF"
    purple = "#7159c1"

    fig_dark, ax_dark = plt.subplots(facecolor=dark_bg)
    ax_dark.set_facecolor(dark_bg)

    # Pontos
    ax_dark.scatter(x, y, color=purple, edgecolors=white, alpha=0.7)

    # Linha de regressão
    x_line = pd.Series(sorted(x))
    y_line = intercept + slope * x_line
    ax_dark.plot(x_line, y_line, color=white, linewidth=2, label="Regressão Linear")

    ax_dark.set_xlabel(col_x, color=white)
    ax_dark.set_ylabel(col_y, color=white)
    ax_dark.legend(facecolor=dark_bg, edgecolor=white, labelcolor=white)

    ax_dark.tick_params(colors=white)
    for spine in ax_dark.spines.values():
        spine.set_edgecolor(white)

    plt.tight_layout()
    st.pyplot(fig_dark)

    # Buffer para download (dark)
    dark_buf = io.BytesIO()
    fig_dark.savefig(dark_buf, format="png", facecolor=dark_bg)
    dark_buf.seek(0)

    # ───────────────────────────────────────────────────────
    # Gráfico tema claro
    plt.style.use("default")
    fig_light, ax_light = plt.subplots()
    ax_light.scatter(x, y, color=purple, edgecolors="black", alpha=0.7)
    ax_light.plot(x_line, y_line, color="black", linewidth=2, label="Regressão Linear")
    ax_light.set_title(f"{col_y} em função de {col_x}")
    ax_light.set_xlabel(col_x)
    ax_light.set_ylabel(col_y)
    ax_light.legend()
    plt.tight_layout()

    light_buf = io.BytesIO()
    fig_light.savefig(light_buf, format="png", facecolor="white")
    light_buf.seek(0)

    # ───────────────────────────────────────────────────────
    # Botões de download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Baixar Gráfico (Tema Escuro)",
            data=dark_buf,
            file_name=f"regressao_{col_y}_vs_{col_x}_dark.png",
            mime="image/png",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📥 Baixar Gráfico (Tema Claro)",
            data=light_buf,
            file_name=f"regressao_{col_y}_vs_{col_x}_light.png",
            mime="image/png",
            use_container_width=True
        )


# ───────────────────────────────────────────────────────────
# PÁGINA PRINCIPAL
# ───────────────────────────────────────────────────────────

load_css()
st.title("Estatística Inferencial")
st.subheader("Regressões, testes de hipótese e magnitude.")
st.divider()

if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Nenhum dataframe disponível. Volte à página inicial e carregue um arquivo.")
    st.stop()

df_names = list(st.session_state.dataframes.keys())
selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names)
df = st.session_state.dataframes[selected_df_name]

# Executa os módulos de inferência

st.subheader("Scatter Plot")
st.caption("Visualize a relação entre duas variáveis numéricas em um plano cartesiano.")
scatter_visualizer(df)
test_normality(df)
correlation_analysis(df)
linear_regression_analysis(df)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)