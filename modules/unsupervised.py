import streamlit as st
import pandas as pd

from modules.k_means import render_kmeans
from modules.dbscan import render_dbscan


def render_unsupervised(df: pd.DataFrame):
    """
    Orquestrador de aprendizado não supervisionado.
    Separa o fluxo do K-Means de outros métodos (ex.: DBSCAN).
    """
    method = st.radio(
        "Escolha o método de clusterização:",
        ["K-Means", "DBSCAN"],
        horizontal=True,
        key="ml_unsup_method_choice",
    )

    if method == "K-Means":
        render_kmeans(df)
    else:
        render_dbscan(df)
