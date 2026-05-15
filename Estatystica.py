import streamlit as st
import pandas as pd
import numpy as np

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


def _mat_labels(value):
    arr = np.asarray(value).squeeze()
    if arr.ndim != 1 or arr.size == 0:
        return None

    labels = []
    for item in arr:
        item_arr = np.asarray(item).squeeze()
        if isinstance(item, str):
            label = item
        elif item_arr.size == 1 and isinstance(item_arr.item(), str):
            label = item_arr.item()
        else:
            label = str(item)
        labels.append(label.strip())

    if not all(labels):
        return None
    return labels


def _unique_columns(columns):
    seen = {}
    unique = []
    for column in columns:
        base = str(column).strip() or "variavel"
        count = seen.get(base, 0)
        unique.append(base if count == 0 else f"{base}_{count + 1}")
        seen[base] = count + 1
    return unique


def load_mat_dataframe(file):
    from scipy.io import loadmat

    try:
        mat_data = loadmat(BytesIO(file.read()), squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise ValueError(
            "Este parece ser um arquivo MATLAB v7.3. Esse formato usa HDF5 e precisa "
            "do pacote h5py para leitura."
        ) from exc

    variables = {
        name: value
        for name, value in mat_data.items()
        if not name.startswith("__")
    }
    if not variables:
        raise ValueError("Nenhuma vari\u00e1vel de dados foi encontrada no arquivo .mat.")

    numeric_2d = []
    numeric_1d = []
    label_vectors = []

    for name, value in variables.items():
        arr = np.asarray(value).squeeze()

        if arr.dtype.kind in "biufc" and arr.size:
            if arr.ndim == 2:
                numeric_2d.append((name, arr))
            elif arr.ndim == 1:
                numeric_1d.append((name, arr))
            elif arr.ndim == 0:
                numeric_1d.append((name, arr.reshape(1)))
        else:
            labels = _mat_labels(value)
            if labels:
                label_vectors.append((name, labels))

    if numeric_2d:
        main_name, main_arr = max(numeric_2d, key=lambda item: item[1].size)
        n_rows, n_cols = main_arr.shape

        columns = None
        for label_name, labels in sorted(
            label_vectors,
            key=lambda item: ("label" not in item[0].lower() and "ch" not in item[0].lower(), item[0]),
        ):
            if len(labels) == n_cols:
                columns = labels
                break

        if columns is None:
            columns = [f"{main_name}_{idx + 1}" for idx in range(n_cols)]

        df = pd.DataFrame(main_arr, columns=_unique_columns(columns))

        leading_columns = []
        for aux_name, aux_arr in numeric_1d:
            if aux_arr.shape[0] == n_rows:
                leading_columns.append((aux_name, aux_arr))

        for aux_name, aux_arr in reversed(leading_columns):
            df.insert(0, aux_name, aux_arr)

        return df

    if numeric_1d:
        max_len = max(arr.shape[0] for _, arr in numeric_1d)
        same_length_vectors = {
            name: arr
            for name, arr in numeric_1d
            if arr.shape[0] == max_len
        }
        return pd.DataFrame(same_length_vectors)

    raise ValueError(
        "O arquivo .mat foi lido, mas n\u00e3o encontrei matriz ou vetor num\u00e9rico "
        "que possa virar dataframe."
    )


if "dataframes" not in st.session_state:
    st.session_state["dataframes"] = {}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

uploaded_files = st.file_uploader(
    "Carregue um ou mais dataframes:",
    type=["csv", "xls", "xlsx", "sav", "zsav", "mat"],
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

            elif suffix == ".mat":
                df = load_mat_dataframe(file)

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
