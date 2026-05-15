import pandas as pd
import streamlit as st

from modules.unsupervised_common import get_numeric_columns, prepare_feature_matrix


def render_hierarchical(df: pd.DataFrame):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Selecione um DataFrame com pelo menos 2 variaveis numericas.")
        return

    st.markdown("#### Configuracoes do Hierarchical Clustering")

    cols = st.multiselect(
        "Variaveis numericas (features) para clusterizacao:",
        numeric_cols,
        key="ml_unsup_hc_cols",
    )
    if not cols or len(cols) < 2:
        st.info("Selecione ao menos 2 variaveis para continuar.")
        return

    c1, c2 = st.columns(2)
    with c1:
        missing_strategy = st.selectbox(
            "Missing values:",
            ["Excluir linhas com NA", "Imputar media"],
            key="ml_unsup_hc_missing",
        )
    with c2:
        scaler_choice = st.selectbox(
            "Escalonamento:",
            ["Nenhum", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            key="ml_unsup_hc_scaler",
        )

    try:
        X_df, X_model, _ = prepare_feature_matrix(
            df=df,
            cols=cols,
            missing_strategy=missing_strategy,
            scaler_choice=scaler_choice,
        )
    except ValueError as e:
        st.error(str(e))
        return

    max_k = int(min(20, max(2, X_model.shape[0] - 1)))
    n_clusters = st.number_input(
        "n_clusters",
        min_value=2,
        max_value=max_k,
        value=min(3, max_k),
        step=1,
        key="ml_unsup_hc_n_clusters",
    )

    linkage = st.selectbox(
        "linkage",
        ["ward", "complete", "average", "single"],
        key="ml_unsup_hc_linkage",
    )

    metric_options = ["euclidean"] if linkage == "ward" else ["euclidean", "manhattan", "cosine"]
    metric = st.selectbox("metric", metric_options, key="ml_unsup_hc_metric")

    if st.button("Rodar Hierarchical Clustering", use_container_width=True, key="ml_unsup_hc_run"):
        model = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            linkage=str(linkage),
            metric=str(metric),
        )
        labels = model.fit_predict(X_model)

        unique_labels = set(labels.tolist())
        can_score = len(unique_labels) >= 2 and X_model.shape[0] > len(unique_labels)
        if can_score:
            sil = float(silhouette_score(X_model, labels, metric=metric))
            ch_score = float(calinski_harabasz_score(X_model, labels))
            db_score = float(davies_bouldin_score(X_model, labels))
        else:
            sil = float("nan")
            ch_score = float("nan")
            db_score = float("nan")

        st.session_state["ml_unsup_hc_cache"] = {
            "labels": labels.tolist(),
            "used_index": X_df.index.tolist(),
            "metrics": {
                "silhouette": (None if pd.isna(sil) else sil),
                "CH": (None if pd.isna(ch_score) else ch_score),
                "DB": (None if pd.isna(db_score) else db_score),
            },
            "counts": pd.Series(labels).value_counts().sort_index().to_dict(),
            "X_model": X_model.tolist(),
            "params": {
                "n_clusters": int(n_clusters),
                "linkage": str(linkage),
                "metric": str(metric),
            },
        }
        st.success("Hierarchical Clustering executado.")

    cache = st.session_state.get("ml_unsup_hc_cache")
    if not cache:
        return

    m = cache["metrics"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Silhouette", "n/a" if m["silhouette"] is None else f"{m['silhouette']:.4f}")
    with c2:
        st.metric("Calinski-Harabasz", "n/a" if m["CH"] is None else f"{m['CH']:,.1f}")
    with c3:
        st.metric("Davies-Bouldin", "n/a" if m["DB"] is None else f"{m['DB']:.3f}")

    counts = pd.Series(cache["counts"]).sort_index()
    counts.index = [f"cluster_{i}" for i in counts.index]
    st.write("#### Tamanho dos clusters")
    st.dataframe(counts.rename("n").to_frame(), use_container_width=True)

    st.write("#### Dendrograma")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

        X_arr = np.array(cache["X_model"], dtype=float)
        n_obs = int(X_arr.shape[0])
        if n_obs < 2:
            st.info("Nao ha observacoes suficientes para gerar o dendrograma.")
        else:
            params = cache.get("params", {})
            linkage_used = str(params.get("linkage", "ward"))
            metric_used = str(params.get("metric", "euclidean"))
            metric_for_linkage = "euclidean" if linkage_used == "ward" else metric_used

            z_matrix = scipy_linkage(X_arr, method=linkage_used, metric=metric_for_linkage)

            max_leaves = min(60, n_obs)
            is_truncated = n_obs > max_leaves
            if is_truncated:
                st.caption(f"Dendrograma truncado para {max_leaves} folhas para melhorar a leitura.")

            fig, ax = plt.subplots(figsize=(10, 5))
            dendro_kwargs = {
                "ax": ax,
                "leaf_rotation": 90.0,
                "leaf_font_size": 8.0,
            }
            if is_truncated:
                dendro_kwargs.update(
                    {
                        "truncate_mode": "lastp",
                        "p": max_leaves,
                        "show_leaf_counts": True,
                    }
                )
            else:
                dendro_kwargs.update({"no_labels": True})

            dendrogram(z_matrix, **dendro_kwargs)
            ax.set_title("Dendrograma Hierarquico")
            ax.set_xlabel("Clusters agregados" if is_truncated else "Amostras")
            ax.set_ylabel("Distancia")
            plt.tight_layout()
            st.pyplot(fig)
    except Exception:
        st.info("Nao foi possivel gerar o dendrograma.")

    try:
        import numpy as np
        import plotly.express as px
        from sklearn.decomposition import PCA

        X_arr = np.array(cache["X_model"], dtype=float)
        labels = pd.Series(cache["labels"])
        proj = PCA(n_components=2, random_state=42).fit_transform(X_arr)
        plot_df = pd.DataFrame(
            {"PC1": proj[:, 0], "PC2": proj[:, 1], "cluster": [f"cluster_{int(x)}" for x in labels]}
        )
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", title="Hierarchical: Projecao PCA 2D")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Nao foi possivel gerar a projecao 2D.")

    used_index = pd.Index(cache["used_index"])
    labels = pd.Series(cache["labels"], index=used_index)
    labeled = df.loc[used_index].assign(cluster_hierarchical=labels.values)
    st.write("#### Amostra rotulada")
    st.dataframe(labeled.head(20), use_container_width=True)

    st.download_button(
        "Download (CSV Hierarchical)",
        data=labeled.to_csv(index=True).encode("utf-8"),
        file_name="hierarchical_clusters.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_unsup_hc_download",
    )
