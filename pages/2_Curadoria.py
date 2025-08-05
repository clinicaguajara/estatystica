# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd

from utils.design import load_css

# CUSTOM FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def remove_rows_with_repeated_value(df: pd.DataFrame, df_name: str):
    """
    <docstrings>
    Remove linhas onde um valor específico se repete com frequência dominante em um conjunto de colunas,
    indicando baixa variação nas respostas (ex: todas as respostas foram "2").

    Args:
        df (pd.DataFrame): DataFrame original.
        df_name (str): Nome do dataframe salvo no session_state.

    Calls:
        st.multiselect(): Seleção de colunas-alvo | instanciado por st.
        st.slider(): Limite de frequência dominante permitida | instanciado por st.
        st.number_input(): Valor a ser monitorado | instanciado por st.
        df.drop(): Remove linhas | método do DataFrame.
        st.session_state.dataframes.__setitem__(): Atualiza o dataframe global | instanciado por session_state.

    Returns:
        None.
    """
    st.write("### Remoção por repetição excessiva de um valor específico")
    st.caption("Remove linhas em que um mesmo valor (ex: 0, 1, 2...) aparece muitas vezes entre os itens selecionados.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("Este dataframe não possui colunas numéricas.")
        return

    selected_cols = st.multiselect("Selecione os itens para verificar:", numeric_cols, key="cols_valor_check")
    if not selected_cols:
        return

    valor_alvo = st.number_input("Valor alvo a ser monitorado:", step=1, key="valor_alvo_check")

    max_freq = st.slider(
        "Remover linhas onde o valor alvo aparece mais que este número de vezes:",
        min_value=1,
        max_value=len(selected_cols),
        value=5,
        key="slider_freq_valor"
    )

    placeholder_valor = st.empty()

    if st.button("🚫 Remover por valor dominante", use_container_width=True):
        try:
            # Conta quantas vezes o valor alvo aparece por linha
            freq_valor = df[selected_cols].apply(lambda row: (row == valor_alvo).sum(), axis=1)

            # Encontra índices onde esse valor aparece mais do que o limite
            indices_repetidos = freq_valor[freq_valor > max_freq].index.tolist()

            if not indices_repetidos:
                placeholder_valor.info("Nenhuma linha excede o limite de repetição do valor alvo.")
            else:
                df.drop(index=indices_repetidos, inplace=True)
                df.reset_index(drop=True, inplace=True)
                st.session_state.dataframes[df_name] = df
                st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")
                placeholder_valor.success(f"{len(indices_repetidos)} linha(s) removida(s) com valor {valor_alvo} dominante.")
        except Exception as e:
            placeholder_valor.error(f"Erro: {e}")

def delete_rows_or_columns(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    UI reutilizável para deletar linhas ou colunas de um DataFrame,
    com feedback visual via placeholder.

    Args:
        df (pd.DataFrame): DataFrame a ser modificado.
        df_name (str): Nome do DataFrame no session_state.

    Returns:
        pd.DataFrame: DataFrame atualizado após a remoção.
    """
    st.subheader("Deleção de linhas ou colunas")
    action = st.radio(
        "O que deseja remover?",
        ("Linhas", "Colunas"),
        horizontal=True,
        key=f"action_{df_name}"
    )
    placeholder = st.empty()

    # Seleção de itens a remover
    if action == "Linhas":
        to_remove = st.multiselect(
            "Selecione índices para remover:",
            df.index.tolist(),
            key=f"idx_remove_{df_name}"
        )
    else:
        to_remove = st.multiselect(
            "Selecione colunas para remover:",
            df.columns.tolist(),
            key=f"col_remove_{df_name}"
        )

    # Botão de execução
    if st.button("🧹 Limpar", use_container_width=True, key=f"btn_remove_{df_name}"):
        # nenhum item selecionado
        if not to_remove:
            placeholder.info(
                f"Nenhuma {'linha' if action == 'Linhas' else 'coluna'} selecionada."
            )
        else:
            try:
                # faz a remoção
                if action == "Linhas":
                    df.drop(index=to_remove, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    # atualiza CSV para download, se necessário
                    st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")
                else:
                    df.drop(columns=to_remove, inplace=True)

                # evita DataFrame sem colunas
                if df.shape[1] == 0:
                    placeholder.error("Todas as colunas foram removidas. DataFrame inválido.")
                else:
                    # persiste no session_state
                    st.session_state.dataframes[df_name] = df
                    placeholder.success(
                        f"{len(to_remove)} {'linha(s)' if action == 'Linhas' else 'coluna(s)'} removida(s) com sucesso."
                    )
            except Exception as e:
                placeholder.error(f"Erro ao remover: {e}")

    return df

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
    st.subheader("Mapear variáveis categóricas")
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

def conditional_row_removal(df: pd.DataFrame, df_name: str):
    """
    <docstrings>
    Permite remover todas as linhas que satisfaçam uma condição lógica definida pelo usuário.

    Args:
        df (pd.DataFrame): DataFrame a ser filtrado.
        df_name (str): Nome do dataframe na session_state.

    Calls:
        st.selectbox(): Seleção de coluna e operador | instanciado por st.
        st.number_input(): Valor a comparar | instanciado por st.
        df.query(): Aplica a condição | método do DataFrame.
        df.drop(): Remove as linhas filtradas | método do DataFrame.
        st.session_state.dataframes.__setitem__(): Atualiza o dataframe no estado global | instanciado por session_state.

    Returns:
        None.
    """
    st.subheader("Remoção condicional de linhas")
    st.caption("Remova todas as linhas onde uma determinada condição seja satisfeita.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("Este dataframe não possui colunas numéricas.")
        return

    col_cond = st.selectbox("Coluna de condição:", numeric_cols, key="cond_col_global")
    operador = st.selectbox("Operador lógico:", ["<", "<=", "==", "!=", ">=", ">"], key="cond_op_global")
    valor = st.number_input("Valor de comparação:", key="cond_val_global")
    placeholder = st.empty()

    if st.button("🧹 Limpar", use_container_width=True, key="btn_remocao_cond_global"):
        try:
            cond = f"`{col_cond}` {operador} {valor}"
            indices = df.query(cond).index.tolist()
            if not indices:
                placeholder.info("Nenhuma linha atende à condição especificada.")
            else:
                df.drop(index=indices, inplace=True)
                df.reset_index(drop=True, inplace=True)
                st.session_state.dataframes[df_name] = df
                st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")
                placeholder.success(f"{len(indices)} linha(s) removida(s).")
        except Exception as e:
            placeholder.error(f"Erro: {e}")

# PAGE 2 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

# Título e instruções iniciais
st.title("Curadoria")

st.caption("""
A seção **Curadoria** oferece ferramentas essenciais para limpeza e transformação de dados antes da análise estatística. 
Permite remover linhas ou colunas manualmente, excluir registros com valores extremos, aplicar filtros condicionais e eliminar padrões de resposta redundantes (ex: repetições excessivas do mesmo valor).
Também disponibiliza um sistema de **mapeamento categórico em lote**, que converte múltiplas variáveis qualitativas em códigos numéricos padronizados. 
Ideal para garantir a qualidade e a consistência dos dados, preparando-os para análises psicométricas, estatísticas ou modelagens mais avançadas.
""")


# Verificação da presença de dataframes
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Nenhum dataframe carregado.")
    st.stop()

# Seleção do dataframe para visualização
df_names = list(st.session_state.dataframes.keys())

# Verify dataframe
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Este dataframe não possui colunas numéricas.")
    st.stop()

selected_df_name = st.session_state.get("selected_df_name")

if selected_df_name not in df_names:
    selected_df_name = df_names[0]


selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names, index=df_names.index(selected_df_name))
df = st.session_state.dataframes[selected_df_name]
st.write(f"**Dimensões:** {df.shape[0]} × {df.shape[1]}")

st.divider()

# BODY ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# Deleção de linhas ou colunas (reutilizável)
df = delete_rows_or_columns(df, selected_df_name)

st.markdown("<br>", unsafe_allow_html=True)

conditional_row_removal(df, selected_df_name)

st.markdown("<br>", unsafe_allow_html=True)

# Executa o mapeamento categórico em lote e sobrescreve o dataframe original
st.session_state["__temp_df_for_mapping__"] = df.copy()

batch_map_categorical_values(
    df_key_input="__temp_df_for_mapping__",
    df_key_output="__temp_df_for_mapping__",
    df_name=selected_df_name
   )


st.markdown("<br>", unsafe_allow_html=True)

remove_rows_with_repeated_value(df, selected_df_name)


st.markdown("<br>", unsafe_allow_html=True)

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
df = st.session_state.dataframes[selected_df_name]

# Verificação de integridade antes da renderização
if df is None or not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] == 0:
    st.warning(f"O dataframe '{selected_df_name}' está vazio ou inválido.")
    st.stop()

st.write(f"Visualizando as primeiras {num_rows} linhas de **{selected_df_name}**:")
st.dataframe(df.head(num_rows), use_container_width=True)

if st.session_state.get("csv_transformado"):
        st.write("### Baixar dataframe transformado")
        st.download_button(
            label="📥 Download (curadoria)",
            data=st.session_state["csv_transformado"],
            file_name=f"{selected_df_name}_filtrado.csv",
            mime="text/csv",
            use_container_width=True
        )
