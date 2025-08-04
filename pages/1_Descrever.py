# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.design import load_css

# CUSTOM FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def batch_map_categorical_values(df_key_input="raw_df", df_key_output="numeric_df", df_name="selected_df_name"):
    """
    <docstrings>
    Aplica um mapeamento numérico compartilhado a múltiplas colunas categóricas de um dataframe salvo no session_state.

    Args:
        df_key_input (str): Nome da chave no session_state contendo o dataframe original.
        df_key_output (str): Nome da chave onde o dataframe transformado será salvo.
        df_name (str): Nome do dataframe selecionado.

    Calls:
        st.session_state.get(): Método para acessar valores salvos no estado | instanciado por st.session_state.
        st.multiselect(): Componente de seleção múltipla para colunas | instanciado por st.
        st.number_input(): Entrada numérica para mapeamento | instanciado por st.
        st.form_submit_button(): Botão de envio do formulário | instanciado por st.
        st.download_button(): Geração de botão para exportar CSV | instanciado por st.

    Raises:
        st.warning: Caso o dataframe de entrada não exista ou não tenha colunas categóricas.
    """
    
    # Título e instruções.
    st.header("Mapeamento em Lote")
    st.caption("Converta múltiplas colunas categóricas para valores numéricos com o mesmo esquema de mapeamento.")

    # Verifica se o dataframe de entrada existe.
    if df_key_input not in st.session_state:
        st.warning("Nenhum dataframe foi carregado ainda.")
        st.stop()

    # Carrega o dataframe.
    df_original = st.session_state[df_key_input].copy()

    # Filtra colunas categóricas.
    categorical_columns = df_original.select_dtypes(include="object").columns.tolist()
    if not categorical_columns:
        st.info("Nenhuma coluna categórica encontrada.")
        st.stop()

    # Seleciona colunas a serem mapeadas.
    selected_columns = st.multiselect(
        f"Selecione as colunas que você deseja mapear em **{df_name}**:",
        categorical_columns
    )

    # Se houver colunas selecionadas, permite criar o dicionário de mapeamento.
    if selected_columns:
        unique_values = set()
        for col in selected_columns:
            unique_values.update(df_original[col].dropna().unique().tolist())

        unique_values = sorted(unique_values)

        st.subheader("Defina o mapeamento numérico:")
        with st.form("batch_mapping_form"):
            value_map = {}
            for val in unique_values:
                value_map[val] = st.number_input(
                    f"'{val}' →", step=1, format="%d", key=f"map_{val}"
                )

            placeholder_map = st.empty()
            submitted = st.form_submit_button("🧭 Aplicar Mapeamento", use_container_width=True)

        # Aplica o mapeamento após envio.
        if submitted:
            for col in selected_columns:
                df_original[col] = df_original[col].map(value_map)

            st.session_state[df_key_output] = df_original

            # Substitui o dataframe original diretamente, se for cópia temporária.
            if df_key_input == df_key_output == "__temp_df_for_mapping__":
                st.session_state.dataframes[df_name] = df_original

            placeholder_map.success("Mapeamento aplicado com sucesso!")
            st.subheader("Prévia dos dados transformados")
            st.dataframe(df_original[selected_columns].head())

            # Botão para exportar.
            st.download_button(
                label="📥 Baixar CSV transformado",
                data=df_original.to_csv(index=False).encode("utf-8"),
                file_name="dados_transformados.csv",
                mime="text/csv",
                use_container_width=True
            )

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

    st.header("📊 Descrição por Coluna")

    # ───────────────────────────────────────────────────────
    # Verifica e seleciona coluna numérica
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("Nenhuma coluna numérica detectada.")
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
    st.subheader("Visualização Gráfica")
    plot_type = st.radio("Escolha o tipo de gráfico:", ["Histograma", "Boxplot", "Curva de Densidade"], horizontal=True)

    col_data_clean = col_data.dropna()
    dark_bg = "#0E1117"
    white = "#FFFFFF"
    purple = "#7159c1"

    # Gráfico modo escuro
    fig, ax = plt.subplots(facecolor=dark_bg)
    ax.set_facecolor(dark_bg)

    if plot_type == "Histograma":
            counts, bins, patches = ax.hist(col_data_clean, bins=20, color=purple, edgecolor=white)
            ax.set_title(f"Histograma de {selected_col}", color=white)
            ax.set_xlabel(selected_col, color=white)
            ax.set_ylabel("Frequência", color=white)
            ax.tick_params(colors=white)
            # Anota frequências em cada barra
            for rect, count in zip(patches, counts):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, height, int(count),
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
            label="📥 Baixar Gráfico (Tema Escuro)",
            data=dark_buf,
            file_name=f"{selected_col}_{plot_type.lower().replace(' ', '_')}_dark.png",
            mime="image/png",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📥 Baixar Gráfico (Tema Claro)",
            data=light_buf,
            file_name=f"{selected_col}_{plot_type.lower().replace(' ', '_')}_light.png",
            mime="image/png",
            use_container_width=True
        )
    
    
    # Renderiza as tabelas
    st.subheader("Tendência Central")
    st.caption("Métricas que resumem a localização dos dados na distribuição.")
    st.table(pd.DataFrame(tendencia_central.items(), columns=["Estatística", "Valor"]))

    st.subheader("Dispersão e Forma")
    st.caption("Indicadores de variabilidade, amplitude e o formato da distribuição.")
    st.table(pd.DataFrame(dispersao.items(), columns=["Estatística", "Valor"]))


# PAGE 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

# Título e instruções iniciais
st.title("Estatísticas Descritivas")
st.subheader("Visualização, tendência central e dispersão das amostras")
st.divider()

# Verificação da presença de dataframes
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Volte à página inicial e carregue um arquivo .csv para começar.")
    st.stop()

# Seleção do dataframe para visualização
df_names = list(st.session_state.dataframes.keys())
selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names)

# Controle do número de linhas com incremento nativo
num_rows = st.number_input(
    "Número de linhas para inspeção visual:",
    min_value=5,
    max_value=100,
    value=5,
    step=5,
    format="%d"
)

# Visualização do dataframe selecionado
df = st.session_state.dataframes[selected_df_name]
st.write(f"Visualizando as primeiras {num_rows} linhas de **{selected_df_name}**:")
st.dataframe(df.head(num_rows), use_container_width=True)




# —— escolha a ação FORA do form —— 
ação = st.radio(
    "O que você quer remover?",
    ("Linhas", "Colunas"),
    horizontal=True,
    key="acao_remocao"
)

# —— formulário de remoção —— 
with st.form("form_transformacoes"):
    st.markdown("#### Transformações no DataFrame")

    if ação == "Linhas":
        # limpa seleções antigas de colunas, se houver
        st.session_state.pop("sel_colunas", None)
        selec = st.multiselect(
            "Selecione os índices das linhas a excluir:",
            df.index.tolist(),
            key="sel_indices"
        )
    else:
        # limpa seleções antigas de linhas, se houver
        st.session_state.pop("sel_indices", None)
        selec = st.multiselect(
            "Selecione as colunas a excluir:",
            df.columns.tolist(),
            key="sel_colunas"
        )

    # placeholder DENTRO do form, logo antes do botão  
    placeholder_remove = st.empty()

    aplicar = st.form_submit_button("🗑️ Aplicar Remoção", use_container_width=True)

    if aplicar:
        if not selec:
            placeholder_remove.warning(
                f"Nenhuma {('linha' if ação=='Linhas' else 'coluna')} selecionada."
            )
        else:
            # aplica remoção
            if ação == "Linhas":
                df.drop(index=selec, inplace=True)
                df.reset_index(drop=True, inplace=True)
            else:
                df.drop(columns=selec, inplace=True)

            # persiste no session_state  
            st.session_state.dataframes[selected_df_name] = df

            # feedback visual  
            placeholder_remove.success(
                f"{len(selec)} {('linha' if ação=='Linhas' else 'coluna')} removida(s) com sucesso!"
            )

            # prepara CSV para download  
            st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")

# —— fora do form: botão de download —— 
if st.session_state.get("csv_transformado"):
    st.download_button(
        label="📥 Baixar CSV transformado",
        data=st.session_state["csv_transformado"],
        file_name=f"{selected_df_name}_transformado.csv",
        mime="text/csv",
        use_container_width=True
    )







# Executa o mapeamento categórico em lote e sobrescreve o dataframe original
st.session_state["__temp_df_for_mapping__"] = df.copy()

batch_map_categorical_values(
    df_key_input="__temp_df_for_mapping__",
    df_key_output="__temp_df_for_mapping__",
    df_name=selected_df_name
)


describe_numeric_column(df, selected_df_name)


st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)