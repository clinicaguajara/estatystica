import pandas as pd
import streamlit as st

from modules.dbscan import render_dbscan
from modules.fuzzy import render_fuzzy
from modules.gaussian_mixture import render_gaussian_mixture
from modules.hierarchical import render_hierarchical
from modules.k_means import render_kmeans
from modules.som import render_som
from modules.spherical_k_means import render_spherical_kmeans


def render_unsupervised(df: pd.DataFrame):
    """
    Orquestrador de aprendizado nao supervisionado.
    """
    method = st.radio(
        "Escolha o metodo de clusterizacao:",
        ["K-Means", "Spherical K-Means", "Gaussian Mixture", "Hierarchical", "Fuzzy C-Means", "DBSCAN", "SOM"],
        horizontal=True,
        key="ml_unsup_method_choice",
    )

    if method == "K-Means":
        render_kmeans(df)
    elif method == "Spherical K-Means":
        render_spherical_kmeans(df)
    elif method == "Gaussian Mixture":
        render_gaussian_mixture(df)
    elif method == "Hierarchical":
        render_hierarchical(df)
    elif method == "Fuzzy C-Means":
        render_fuzzy(df)
    elif method == "SOM":
        render_som(df)
    else:
        render_dbscan(df)
