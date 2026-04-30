import streamlit as st
import pandas as pd

from utils.design import load_css


load_css()

st.title("Estrutura de Dados")
st.caption(
    "Renomeie, exclua e empilhe estruturas de dados carregadas na sessão atual."
)


if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "rename_feedback" not in st.session_state:
    st.session_state.rename_feedback = None

if "delete_feedback" not in st.session_state:
    st.session_state.delete_feedback = None


def render_table(dataframes: dict):
    table_data = [
        {"Nome": name, "Dimens\u00f5es": f"{df.shape[0]} x {df.shape[1]}"}
        for name, df in dataframes.items()
    ]
    st.table(pd.DataFrame(table_data))


def rename_dataframe_form(dataframes: dict):
    st.divider()
    st.subheader("Renomear estrutura de dados")

    with st.form("rename_form"):
        selected_df = st.selectbox("Selecione um dataframe:", list(dataframes.keys()), key="rename_select")
        new_name = st.text_input("Novo nome:", key="rename_input").strip()
        submitted = st.form_submit_button("\U0001F4DD Renomear", use_container_width=True)

        if submitted:
            if not new_name:
                st.warning("Preencha o nome do novo dataframe.")
                return
            if new_name == selected_df:
                st.warning("Os nomes s\u00e3o id\u00eanticos.")
                return
            if new_name in dataframes:
                st.warning(f"O nome '{new_name}' j\u00e1 est\u00e1 em uso.")
                return

            dataframes[new_name] = dataframes.pop(selected_df)
            st.session_state.processed_files.discard(selected_df)
            st.session_state.processed_files.add(new_name)

            if st.session_state.get("selected_df_name") == selected_df:
                st.session_state["selected_df_name"] = new_name
            if st.session_state.get("ml_selected_df_name") == selected_df:
                st.session_state["ml_selected_df_name"] = new_name

            st.session_state.rename_feedback = f"'{selected_df}' foi renomeado para '{new_name}'."
            st.rerun()


def delete_dataframe_form(dataframes: dict):
    st.divider()    
    st.subheader("Deletar estrutura de dados")

    with st.form("delete_form"):
        selected_df = st.selectbox(
            "Selecione um dataframe para excluir:",
            list(dataframes.keys()),
            key="delete_select",
        )
        submitted = st.form_submit_button("🗑️ Excluir", use_container_width=True)

        if submitted:
            del dataframes[selected_df]
            st.session_state.processed_files.discard(selected_df)

            if st.session_state.get("selected_df_name") == selected_df:
                st.session_state.pop("selected_df_name", None)
            if st.session_state.get("ml_selected_df_name") == selected_df:
                st.session_state.pop("ml_selected_df_name", None)

            st.session_state.delete_feedback = f"'{selected_df}' foi removido."
            st.rerun()


def merge_dataframes_with_fill(dataframes: dict):
    st.divider()
    st.subheader("Empilhar estrutura de dados")
    st.caption(
        "As colunas do primeiro dataframe selecionado ser\u00e3o preservadas na ordem original; "
        "colunas extras dos demais entrar\u00e3o ao final. Colunas ausentes ser\u00e3o preenchidas com 'NaN'."
    )

    if msg := st.session_state.pop("merge_feedback", None):
        st.success(msg)

    df_names = list(dataframes.keys())
    if len(df_names) < 2:
        st.info("Voc\u00ea precisa de pelo menos dois dataframes carregados.")
        return

    with st.form("merge_multiple_form"):
        selected_names = st.multiselect(
            "Selecione os dataframes que deseja empilhar (a ordem de sele\u00e7\u00e3o importa):",
            df_names,
            default=[],
        )
        new_name = st.text_input("Nome para o novo dataframe combinado:", value="uniao_dataframes")
        submitted = st.form_submit_button("Empilhar Dataframes", use_container_width=True)

        if submitted:
            if len(selected_names) < 2:
                st.warning("Selecione pelo menos dois dataframes.")
                return

            dfs = [dataframes[name].copy() for name in selected_names]

            base_cols = list(dfs[0].columns)

            seen = set(base_cols)
            extra_cols = []
            for df in dfs[1:]:
                for col in df.columns:
                    if col not in seen:
                        extra_cols.append(col)
                        seen.add(col)

            final_cols = base_cols + extra_cols

            for i in range(len(dfs)):
                missing = [c for c in final_cols if c not in dfs[i].columns]
                for col in missing:
                    dfs[i][col] = pd.NA
                dfs[i] = dfs[i][final_cols]

            df_merged = pd.concat(dfs, ignore_index=True, sort=False)

            st.session_state.dataframes[new_name] = df_merged
            st.session_state["merge_feedback"] = (
                f"{len(selected_names)} dataframes empilhados como '{new_name}' "
                f"({df_merged.shape[0]} linhas x {df_merged.shape[1]} colunas)."
            )
            st.rerun()


dataframes = st.session_state.dataframes
if not dataframes:
    st.info("Nenhum dataframe carregado na sess\u00e3o.")
    st.stop()

if msg := st.session_state.pop("rename_feedback", None):
    st.success(msg)
elif msg := st.session_state.pop("delete_feedback", None):
    st.success(msg)

render_table(dataframes)

rename_dataframe_form(dataframes)

delete_dataframe_form(dataframes)

merge_dataframes_with_fill(dataframes)
