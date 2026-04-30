import streamlit as st
import pandas as pd

from pathlib import Path
from io import BytesIO
from utils.design import load_css
from utils.variables import version


st.set_page_config(
    page_title="Estatystica",
    page_icon="\U0001F4C8",
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_css()

st.title("Estatystica")
st.caption(f"Vers\u00e3o {version}")
st.markdown(
    """
<h4>O seu software <span class='verde-magico'>B</span><span class='amarelo-brasil'>R</span> para an\u00e1lise de dados e Machine Learning!</h4>
""",
    unsafe_allow_html=True,
)

st.divider()


def render_loaded_dataframes_overview(dataframes: dict):
    table_data = [
        {"Nome": name, "Dimens\u00f5es": f"{df.shape[0]} x {df.shape[1]}"}
        for name, df in dataframes.items()
    ]

    st.markdown("<br>", unsafe_allow_html=True)

    if table_data:
        st.table(pd.DataFrame(table_data))
    else:
        st.info("Nenhum dataframe dispon\u00edvel para an\u00e1lise.")


if "dataframes" not in st.session_state:
    st.session_state["dataframes"] = {}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

uploaded_files = st.file_uploader(
    "Carregue um ou mais dataframes:",
    type=["csv", "xls", "xlsx", "sav", "zsav"],
    accept_multiple_files=True,
)

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

            elif suffix in [".sav", ".zsav"]:
                import tempfile
                import os

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name
                    df = pd.read_spss(tmp_path, convert_categoricals=True)
                except Exception as e_spss:
                    st.error(f"Erro ao carregar '{file.name}' (SPSS): {e_spss}")
                    st.info("Verifique se 'pyreadstat>=1.2.0' est\u00e1 instalado no ambiente.")
                    continue
                finally:
                    if tmp_path:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

            else:
                st.warning(f"Tipo de arquivo n\u00e3o suportado: {file.name}")
                continue

            st.session_state.dataframes[name] = df
            st.session_state.processed_files.add(name)

        except Exception as e:
            st.error(f"Erro ao carregar '{file.name}': {e}")

st.caption(
    "Carregue mais de um dataframe para realizar operações com bancos de dados, ou explore o menu lateral para realizar operações entre linhas e colunas de um mesmo dataframe já carregado na sessão."
)
render_loaded_dataframes_overview(st.session_state.dataframes)