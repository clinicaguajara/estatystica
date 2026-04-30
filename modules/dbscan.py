import streamlit as st
import pandas as pd

from modules.unsupervised_common import get_numeric_columns, prepare_feature_matrix


DBSCAN_FINAL_KEY = "ml_unsup_dbscan_final"


def _clear_dbscan_cache():
    st.session_state.pop(DBSCAN_FINAL_KEY, None)


def _save_dbscan_to_state(*, labels, used_index, cols, missing_strategy, scaler_choice, eps, min_samples, metric, algorithm, leaf_size, p_value):
    st.session_state[DBSCAN_FINAL_KEY] = {
        "labels": list(labels),
        "used_index": list(used_index),
        "cols": list(cols),
        "params": {
            "missing_strategy": str(missing_strategy),
            "scaler_choice": str(scaler_choice),
            "eps": float(eps),
            "min_samples": int(min_samples),
            "metric": str(metric),
            "algorithm": str(algorithm),
            "leaf_size": int(leaf_size),
            "p_value": (None if p_value is None else float(p_value)),
        },
    }


def _render_dbscan_cache(df: pd.DataFrame):
    cache = st.session_state.get(DBSCAN_FINAL_KEY)
    if not cache:
        return False

    import numpy as np
    from sklearn.metrics import silhouette_score

    labels = np.array(cache["labels"])
    used_index = pd.Index(cache["used_index"])
    params = cache["params"]
    cols = cache["cols"]

    n_total = int(labels.size)
    n_noise = int((labels == -1).sum())
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    n_clusters = int(len(set(non_noise_labels.tolist())))

    silhouette_val = None
    if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
        try:
            X_df, X_model, _ = prepare_feature_matrix(
                df=df,
                cols=cols,
                missing_strategy=params["missing_strategy"],
                scaler_choice=params["scaler_choice"],
            )
            aligned = X_df.loc[used_index]
            X_non_noise = aligned.values if params["scaler_choice"] == "Nenhum" else X_model
            X_non_noise = X_non_noise[non_noise_mask]
            silhouette_val = float(silhouette_score(X_non_noise, non_noise_labels))
        except Exception:
            silhouette_val = None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Clusters encontrados", f"{n_clusters}")
    with c2:
        st.metric("Pontos de ruído (-1)", f"{n_noise} ({(n_noise / n_total * 100) if n_total else 0:.1f}%)")
    with c3:
        st.metric("Silhouette (sem ruído)", "n/a" if silhouette_val is None else f"{silhouette_val:.4f}")

    counts = pd.Series(labels).value_counts().sort_index()
    counts.index = [f"cluster_{i}" if i != -1 else "ruido_-1" for i in counts.index]
    st.write("#### Tamanho dos grupos")
    st.dataframe(counts.rename("n").to_frame(), use_container_width=True)

    labeled = df.loc[used_index].assign(cluster_dbscan=labels)
    st.write("#### Amostra rotulada")
    st.dataframe(labeled.head(20), use_container_width=True)

    csv_bytes = labeled.to_csv(index=True).encode("utf-8")
    st.download_button(
        "📥 Download (CSV DBSCAN)",
        data=csv_bytes,
        file_name="dbscan_clusters.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_unsup_dbscan_download_csv",
    )
    return True


def render_dbscan(df: pd.DataFrame):
    """
    Fluxo de clusterização DBSCAN, separado do K-Means.
    """
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Selecione um DataFrame com pelo menos **2** variáveis numéricas.")
        return

    st.markdown("#### Configurações do DBSCAN")

    cols = st.multiselect(
        "Variáveis numéricas (features) para clusterização:",
        numeric_cols,
        key="ml_unsup_dbscan_cols",
        help="Escolha as colunas que irão compor o espaço de clusterização.",
    )
    if not cols or len(cols) < 2:
        st.info("Selecione ao menos **2** variáveis para continuar.")
        return

    c1, c2 = st.columns(2)
    with c1:
        missing_strategy = st.selectbox(
            "Missing values:",
            ["Excluir linhas com NA", "Imputar média"],
            key="ml_unsup_dbscan_missing",
        )
    with c2:
        scaler_choice = st.selectbox(
            "Escalonamento:",
            ["Nenhum", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            key="ml_unsup_dbscan_scaler",
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        eps = st.number_input(
            "eps",
            min_value=0.01,
            value=0.50,
            step=0.05,
            format="%.2f",
            key="ml_unsup_dbscan_eps",
            help="Distância máxima para considerar vizinhança no DBSCAN.",
        )
    with c4:
        min_samples = st.number_input(
            "min_samples",
            min_value=2,
            value=5,
            step=1,
            key="ml_unsup_dbscan_min_samples",
            help="Número mínimo de pontos para definir região densa.",
        )
    with c5:
        metric = st.selectbox(
            "metric",
            ["euclidean", "manhattan", "cosine", "minkowski"],
            key="ml_unsup_dbscan_metric",
        )

    c6, c7 = st.columns(2)
    with c6:
        algorithm = st.selectbox(
            "algorithm",
            ["auto", "ball_tree", "kd_tree", "brute"],
            key="ml_unsup_dbscan_algorithm",
        )
    with c7:
        leaf_size = st.number_input(
            "leaf_size",
            min_value=5,
            value=30,
            step=1,
            key="ml_unsup_dbscan_leaf_size",
        )

    if metric == "minkowski":
        p_value = st.number_input(
            "p (Minkowski)",
            min_value=1.0,
            value=2.0,
            step=1.0,
            key="ml_unsup_dbscan_p",
        )
    else:
        p_value = None

    run_dbscan = st.button("Rodar DBSCAN", use_container_width=True, key="ml_unsup_dbscan_run")
    if run_dbscan:
        from sklearn.cluster import DBSCAN

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

        _clear_dbscan_cache()

        db_kwargs = {
            "eps": float(eps),
            "min_samples": int(min_samples),
            "metric": metric,
            "algorithm": algorithm,
            "leaf_size": int(leaf_size),
        }
        if metric == "minkowski":
            db_kwargs["p"] = float(p_value if p_value is not None else 2.0)

        model = DBSCAN(**db_kwargs)
        labels = model.fit_predict(X_model)

        _save_dbscan_to_state(
            labels=labels,
            used_index=X_df.index,
            cols=cols,
            missing_strategy=missing_strategy,
            scaler_choice=scaler_choice,
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p_value=p_value,
        )
        st.success("DBSCAN executado e armazenado.")

    _render_dbscan_cache(df)
