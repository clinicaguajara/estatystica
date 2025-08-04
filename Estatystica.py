
# REQUIRED IMPORTS ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#.\.venv\Scripts\activate

import streamlit as st
import pandas as pd

from pathlib        import Path
from utils.design   import load_css
from io             import BytesIO

# CUSTOM FUNCTIONS ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def rename_dataframe_form(dataframes: dict):
    """Formulário para renomear um dataframe."""
    with st.form("rename_form"):
        selected_df = st.selectbox("Selecione um dataframe:", list(dataframes.keys()), key="rename_select")
        new_name = st.text_input("Novo nome:", key="rename_input")
        submitted = st.form_submit_button("📝 Renomear", use_container_width=True)

        if submitted:
            if not new_name:
                st.warning("O novo nome não pode estar vazio.")
            elif new_name == selected_df:
                st.warning("O novo nome é idêntico ao atual.")
            elif new_name in dataframes:
                st.warning(f"O nome '{new_name}' já está em uso.")
            else:
                dataframes[new_name] = dataframes.pop(selected_df)

                # Remove também da estrutura original
                st.session_state.dataframes.pop(selected_df, None)
                st.session_state.dataframes[new_name] = dataframes[new_name]

                # Marcar como processado
                st.session_state.processed_files.add(selected_df)
                st.session_state.processed_files.add(new_name)

                # Fechar formulário e sincronizar
                st.session_state.show_rename_form = False
                sync_loaded_data()
                st.rerun()


def delete_dataframe_form(dataframes: dict):
    """Formulário para deletar um dataframe."""
    with st.form("delete_form"):
        selected_df = st.selectbox("Selecione um dataframe para deletar:", list(dataframes.keys()), key="delete_select")
        submitted = st.form_submit_button("🗑️ Excluir", use_container_width=True)

        if submitted:
            del dataframes[selected_df]

            # Remover também de dataframes original
            st.session_state.dataframes.pop(selected_df, None)

            # Marcar como processado
            st.session_state.processed_files.add(selected_df)

            # Fechar formulário e sincronizar
            st.success(f"'{selected_df}' foi removido.")
            st.session_state.show_delete_form = False
            sync_loaded_data()
            st.rerun()




def merge_dataframes_with_fill(dataframes: dict):
    """
    <docstrings>
    Permite unir múltiplos dataframes, preenchendo colunas ausentes com NaN antes da concatenação.

    Args:
        dataframes (dict): Dicionário com dataframes carregados.

    Calls:
        st.multiselect(): Seleciona múltiplos dataframes | instanciado por st.
        pd.concat(): Junta todos os dataframes verticalmente | função definida em pandas.
        st.session_state.dataframes.__setitem__(): Salva o novo dataframe no estado global | método do session_state.

    Returns:
        None. Cria um novo dataframe combinado no session_state.
    """
    st.subheader("Empilhar dataframes")
    st.caption("Colunas iguais serão alinhadas e colunas diferentes serão preenchidas com 'Nan' automaticamente.")

    df_names = list(dataframes.keys())
    if len(df_names) < 2:
        st.info("Você precisa de pelo menos dois dataframes carregados.")
        return

    with st.form("merge_multiple_form"):
        selected_names = st.multiselect(
            "Selecione os dataframes que deseja empilhar:",
            df_names,
            default=[]
        )

        new_name = st.text_input("Nome para o novo dataframe combinado:", value="uniao_dataframes")
        submitted = st.form_submit_button("Empilhar Dataframes", use_container_width=True)

        if submitted:
            if len(selected_names) < 2:
                st.warning("Selecione pelo menos dois dataframes.")
                return

            # Coleta os dataframes
            dfs = [dataframes[name].copy() for name in selected_names]

            # Determina o conjunto total de colunas
            all_columns = set().union(*[df.columns for df in dfs])

            # Preenche colunas ausentes com NaN em cada dataframe
            for i in range(len(dfs)):
                missing_cols = all_columns - set(dfs[i].columns)
                for col in missing_cols:
                    dfs[i][col] = pd.NA
                # Garante mesma ordem de colunas
                dfs[i] = dfs[i][sorted(all_columns)]

            # Concatena tudo
            df_merged = pd.concat(dfs, ignore_index=True, sort=False)

            # Salva no session_state
            st.session_state.dataframes[new_name] = df_merged
            st.session_state.loaded_data[new_name] = df_merged

            st.success(f"{len(selected_names)} dataframes empilhados com sucesso como '{new_name}' ({df_merged.shape[0]} linhas × {df_merged.shape[1]} colunas).")


def render_loaded_dataframes():
    """Exibe os dataframes carregados com opções de renomear e deletar."""

    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = dict(st.session_state.dataframes)

    if "show_rename_form" not in st.session_state:
        st.session_state.show_rename_form = False
    if "show_delete_form" not in st.session_state:
        st.session_state.show_delete_form = False

    dataframes = st.session_state.loaded_data
    df_names = list(dataframes.keys())

    # Tabela
    table_data = [
        {"Nome": name, "Dimensões": f"{df.shape[0]} × {df.shape[1]}"}
        for name, df in dataframes.items()
    ]
    st.table(pd.DataFrame(table_data))

    # Botões de ação
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Renomear", use_container_width=True):
            st.session_state.show_rename_form = True
            st.session_state.show_delete_form = False
    with col2:
        if st.button("Deletar", use_container_width=True):
            st.session_state.show_delete_form = True
            st.session_state.show_rename_form = False

    # Mostrar formulário correspondente
    if st.session_state.show_rename_form:
        rename_dataframe_form(dataframes)

    if st.session_state.show_delete_form:
        delete_dataframe_form(dataframes)

def sync_loaded_data():
    """Adiciona novos dataframes carregados ao wrapper loaded_data, sem reinserir renomeados/excluídos."""
    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = dict(st.session_state.dataframes)
    else:
        for key in st.session_state.dataframes:
            if key not in st.session_state.loaded_data:
                st.session_state.loaded_data[key] = st.session_state.dataframes[key]


# PAGE 0 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

# Título e subtítulo
st.title("Estatystica")
st.subheader("Análise interativa de dados para estatísticas lineares")
st.divider()

# Inicializa dataframes (modo legado)
if "dataframes" not in st.session_state:
    st.session_state["dataframes"] = {}

# Inicializa se necessário
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Upload múltiplo de arquivos CSV
uploaded_files = st.file_uploader(
    "Carregue um ou mais bancos de dados (.csv):",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)

st.write("### 🗃️ Dataframes disponíveis:")

# Processa os arquivos carregados
if uploaded_files:
    for file in uploaded_files:
        try:
            name = Path(file.name).stem  # Nome base sem extensão
            suffix = Path(file.name).suffix.lower()

            if name in st.session_state.processed_files:
                continue

            # Lê conforme o tipo do arquivo
            if suffix == ".csv":
                df = pd.read_csv(file)
            elif suffix in [".xls", ".xlsx"]:
                df = pd.read_excel(BytesIO(file.read()))  # Leitura segura para arquivos binários
            else:
                st.warning(f"Tipo de arquivo não suportado: {file.name}")
                continue

            # Salva no session_state
            st.session_state.dataframes[name] = df
            st.session_state.processed_files.add(name)

        except Exception as e:
            st.error(f"Erro ao carregar '{file.name}': {e}")

# Garante que loaded_data esteja sempre sincronizado com dataframes
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = dict(st.session_state.dataframes)
else:
    # Verifica se há novos dataframes adicionados
    new_keys = set(st.session_state.dataframes.keys()) - set(st.session_state.loaded_data.keys())
    for key in new_keys:
        st.session_state.loaded_data[key] = st.session_state.dataframes[key]

# Exibe os dataframes disponíveis
if "loaded_data" in st.session_state and st.session_state.loaded_data:
    render_loaded_dataframes()
    merge_dataframes_with_fill(st.session_state.dataframes)
else:
    st.info("Nenhum dataframe disponível para análise.")
