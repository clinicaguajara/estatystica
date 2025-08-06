# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas    as pd

from pathlib         import Path
from io              import BytesIO
from utils.design    import load_css
from utils.variables import version

# HOMEPAGE ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Estatystica",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

load_css()

st.title("Estatystica")
st.caption(f"Versão {version}")
st.markdown("""
<h4>O seu software <span class='verde-magico'>B</span><span class='amarelo-brasil'>R</span> para análise de dados e Machine Learning!</h4>
""", unsafe_allow_html=True)

st.divider()

# CUSTOM FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def rename_dataframe_form(dataframes: dict):
    with st.form("rename_form"):
        selected_df = st.selectbox("Selecione um dataframe:", list(dataframes.keys()), key="rename_select")
        new_name = st.text_input("Novo nome:", key="rename_input")
        submitted = st.form_submit_button("📝 Renomear", use_container_width=True)

        if submitted:
            if not new_name:
                st.warning("Preencha o nome do novo dataframe.")
            elif new_name == selected_df:
                st.warning("Os nomes são idênticos.")
            elif new_name in dataframes:
                st.warning(f"O nome '{new_name}' já está em uso.")
            else:
                dataframes[new_name] = dataframes.pop(selected_df)
                st.session_state.dataframes.pop(selected_df, None)
                st.session_state.dataframes[new_name] = dataframes[new_name]
                st.session_state.processed_files.discard(selected_df)
                st.session_state.processed_files.add(new_name)
                st.session_state.show_rename_form = False
                st.session_state.rename_feedback = f"'{selected_df}' foi renomeado para '{new_name}'."
                sync_loaded_data()
                st.rerun()

def delete_dataframe_form(dataframes: dict):
    with st.form("delete_form"):
        selected_df = st.selectbox("Selecione um dataframe para deletar:", list(dataframes.keys()), key="delete_select")
        submitted = st.form_submit_button("🗑️ Excluir", use_container_width=True)

        if submitted:
            del dataframes[selected_df]
            st.session_state.dataframes.pop(selected_df, None)
            st.session_state.processed_files.discard(selected_df)
            st.session_state.show_delete_form = False
            st.session_state.delete_feedback = f"'{selected_df}' foi removido."
            sync_loaded_data()
            st.rerun()

def merge_dataframes_with_fill(dataframes: dict):
    st.divider()
    st.subheader("Empilhar dataframes")
    st.caption("Colunas iguais serão alinhadas e colunas diferentes serão preenchidas com 'Nan'.")

    # ⬇️ Mensagem de sucesso pós-rerun
    if msg := st.session_state.pop("merge_feedback", None):
        st.success(msg)

    df_names = list(dataframes.keys())
    if len(df_names) < 2:
        st.info("Você precisa de pelo menos dois dataframes carregados.")
        return

    with st.form("merge_multiple_form"):
        selected_names = st.multiselect("Selecione os dataframes que deseja empilhar:", df_names, default=[])
        new_name = st.text_input("Nome para o novo dataframe combinado:", value="uniao_dataframes")
        submitted = st.form_submit_button("Empilhar Dataframes", use_container_width=True)

        if submitted:
            if len(selected_names) < 2:
                st.warning("Selecione pelo menos dois dataframes.")
                return

            dfs = [dataframes[name].copy() for name in selected_names]
            all_columns = set().union(*[df.columns for df in dfs])
            for i in range(len(dfs)):
                missing_cols = all_columns - set(dfs[i].columns)
                for col in missing_cols:
                    dfs[i][col] = pd.NA
                dfs[i] = dfs[i][sorted(all_columns)]

            df_merged = pd.concat(dfs, ignore_index=True, sort=False)
            st.session_state.dataframes[new_name] = df_merged
            st.session_state.loaded_data[new_name] = df_merged

            # ⬇️ Armazena mensagem e força rerun
            st.session_state["merge_feedback"] = f"{len(selected_names)} dataframes empilhados como '{new_name}' ({df_merged.shape[0]} linhas × {df_merged.shape[1]} colunas)."
            st.rerun()

def render_loaded_dataframes():
    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = dict(st.session_state.dataframes)
    if "show_rename_form" not in st.session_state:
        st.session_state.show_rename_form = False
    if "show_delete_form" not in st.session_state:
        st.session_state.show_delete_form = False

    dataframes = st.session_state.loaded_data

    table_data = [
        {"Nome": name, "Dimensões": f"{df.shape[0]} × {df.shape[1]}"} for name, df in dataframes.items()
    ]

    st.markdown("<br>", unsafe_allow_html=True)
    
    placeholder_table = st.empty()
    placeholder_feedback = st.empty()

    if table_data:
        placeholder_table.table(pd.DataFrame(table_data))
    else:
        st.session_state.pop("delete_feedback", None)
        st.session_state.pop("rename_feedback", None)
        placeholder_feedback.info("Nenhum dataframe disponível para análise.")
        return

    # Tenta recuperar e remover a mensagem de renomeação do session_state.
    # Se a variável "rename_feedback" existir, ela será removida (pop) e atribuída a "msg".
    # O operador walrus := permite fazer isso em uma única linha.
    if msg := st.session_state.pop("rename_feedback", None):
        placeholder_feedback.success(msg)
        
    elif msg := st.session_state.pop("delete_feedback", None):
        placeholder_feedback.success(msg)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Renomear", use_container_width=True):
            st.session_state.show_rename_form = True
            st.session_state.show_delete_form = False
    with col2:
        if st.button("Deletar", use_container_width=True):
            st.session_state.show_delete_form = True
            st.session_state.show_rename_form = False

    st.caption("Carregue mais de um dataframe para realizar operações com bancos de dados, ou explore o menu lateral para realizar operações entre linhas e colunas de um mesmo dataframe já carregado na sessão.")

    if st.session_state.show_rename_form:
        rename_dataframe_form(dataframes)
    if st.session_state.show_delete_form:
        delete_dataframe_form(dataframes)

def sync_loaded_data():
    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = dict(st.session_state.dataframes)
    else:
        for key in st.session_state.dataframes:
            if key not in st.session_state.loaded_data:
                st.session_state.loaded_data[key] = st.session_state.dataframes[key]

# IMPLEMENTATION ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# Initializing session variables.
if "dataframes" not in st.session_state:
    st.session_state["dataframes"] = {}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Upload dataframes.
uploaded_files = st.file_uploader(
    "Carregue um ou mais dataframes:",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)

# Proccessing
if uploaded_files:
    for file in uploaded_files:
        try:
            name = Path(file.name).stem
            suffix = Path(file.name).suffix.lower()
            if name in st.session_state.processed_files:
                continue

            if suffix == ".csv":
                df = pd.read_csv(file)
            elif suffix in [".xls", ".xlsx"]:
                df = pd.read_excel(BytesIO(file.read()))
            else:
                st.warning(f"Tipo de arquivo não suportado: {file.name}")
                continue

            st.session_state.dataframes[name] = df
            st.session_state.processed_files.add(name)
        except Exception as e:
            st.error(f"Erro ao carregar '{file.name}': {e}")

# Synch and resume.
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = dict(st.session_state.dataframes)
else:
    new_keys = set(st.session_state.dataframes.keys()) - set(st.session_state.loaded_data.keys())
    for key in new_keys:
        st.session_state.loaded_data[key] = st.session_state.dataframes[key]

    render_loaded_dataframes()

if st.session_state.loaded_data:
    merge_dataframes_with_fill(st.session_state.dataframes)
else:
    pass
