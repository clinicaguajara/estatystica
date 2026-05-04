import streamlit as st
import pandas as pd

from modules.unsupervised_common import get_numeric_columns, prepare_feature_matrix


DBSCAN_FINAL_KEY = "ml_unsup_dbscan_final"


def _clear_dbscan_cache():
    st.session_state.pop(DBSCAN_FINAL_KEY, None)


def _knee_index_by_max_distance(x_values, y_values):
    """
    Heuristica simples de cotovelo: maior distancia perpendicular
    ao segmento entre primeiro e ultimo ponto.
    """
    import numpy as np

    x1, y1 = float(x_values[0]), float(y_values[0])
    x2, y2 = float(x_values[-1]), float(y_values[-1])
    dx, dy = (x2 - x1), (y2 - y1)
    denom = float((dy**2 + dx**2) ** 0.5) if (dx != 0 or dy != 0) else 1.0

    dists = []
    for x, y in zip(x_values, y_values):
        num = abs(dy * float(x) - dx * float(y) + (x2 * y1 - y2 * x1))
        dists.append(num / denom)
    return int(np.argmax(dists))


def _build_kdistance_fig(sorted_k_distances, suggested_eps=None, current_eps=None, theme="dark"):
    import matplotlib.pyplot as plt

    dark_bg, white, purple = "#0E1117", "#FFFFFF", "#7159c1"
    if theme == "dark":
        bg, fg, grid = dark_bg, white, "#22262e"
    else:
        bg, fg, grid = "#FFFFFF", "#0E1117", "#e5e7eb"

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    x_axis = list(range(1, len(sorted_k_distances) + 1))
    ax.plot(x_axis, sorted_k_distances, linewidth=2.0, color=purple)
    ax.set_xlabel("Pontos (ordenados por distancia k)", color=fg)
    ax.set_ylabel("Distancia ao k-esimo vizinho", color=fg)
    ax.set_title("k-distance (estimativa de eps)", color=fg, pad=10)

    if suggested_eps is not None:
        ax.axhline(float(suggested_eps), linestyle="--", color=fg, alpha=0.75)
        ax.text(
            x_axis[max(0, int(len(x_axis) * 0.02))],
            float(suggested_eps),
            f"  eps sugerido ~ {float(suggested_eps):.4f}",
            va="bottom",
            color=fg,
        )

    if current_eps is not None:
        ax.axhline(float(current_eps), linestyle=":", color="#2ecc71", alpha=0.9)

    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    ax.grid(True, color=grid, alpha=0.4)
    fig.tight_layout()
    return fig


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

    st.markdown("#### Estimativa de eps (k-distance)")
    n_samples = int(X_model.shape[0])
    default_use_sample = n_samples > 4000

    kcol1, kcol2 = st.columns(2)
    with kcol1:
        use_sample_for_kdist = st.checkbox(
            "Amostrar pontos no k-distance",
            value=default_use_sample,
            key="ml_unsup_dbscan_kdist_use_sample",
            help="Reduz custo de calculo em bases grandes.",
        )
    with kcol2:
        if use_sample_for_kdist:
            max_kdist_sample = max(500, min(n_samples, 20000))
            default_kdist_sample = min(n_samples, 3000)
            kdist_sample_size = st.number_input(
                "Tamanho da amostra (k-distance)",
                min_value=500,
                max_value=max_kdist_sample,
                value=max(500, default_kdist_sample),
                step=250,
                key="ml_unsup_dbscan_kdist_sample_size",
            )
        else:
            kdist_sample_size = n_samples

    # Amostra opcional para estimativa de eps
    X_for_kdist = X_model
    if use_sample_for_kdist and n_samples > int(kdist_sample_size):
        import numpy as np

        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=int(kdist_sample_size), replace=False)
        X_for_kdist = X_model[idx]

    n_neighbors_kdist = min(max(2, int(min_samples)), int(X_for_kdist.shape[0]))
    if int(min_samples) > int(X_for_kdist.shape[0]):
        st.caption(
            f"min_samples ajustado para {n_neighbors_kdist} no grafico k-distance "
            f"(n da amostra = {int(X_for_kdist.shape[0])})."
        )

    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import numpy as np

    nn_algorithm = algorithm
    # Em alguns ambientes, cosine exige busca brute.
    if metric == "cosine" and nn_algorithm in ("auto", "ball_tree", "kd_tree"):
        nn_algorithm = "brute"

    nn_kwargs = {
        "n_neighbors": int(n_neighbors_kdist),
        "algorithm": nn_algorithm,
        "metric": metric,
        "leaf_size": int(leaf_size),
    }
    if metric == "minkowski":
        nn_kwargs["p"] = float(p_value if p_value is not None else 2.0)

    try:
        nbrs = NearestNeighbors(**nn_kwargs)
        nbrs.fit(X_for_kdist)
        distances, _ = nbrs.kneighbors(X_for_kdist)
        kdist_values = np.sort(distances[:, -1].astype(float))

        x_axis = np.arange(1, len(kdist_values) + 1, dtype=float)
        knee_idx = _knee_index_by_max_distance(x_axis, kdist_values)
        eps_sugerido = float(kdist_values[knee_idx]) if len(kdist_values) else None

        fig_kdist = _build_kdistance_fig(
            sorted_k_distances=kdist_values,
            suggested_eps=eps_sugerido,
            current_eps=float(eps),
            theme="dark",
        )
        st.pyplot(fig_kdist, clear_figure=True)
        plt.close(fig_kdist)

        if eps_sugerido is not None:
            st.caption(
                f"eps sugerido pelo cotovelo: **{eps_sugerido:.4f}** "
                f"(k = {n_neighbors_kdist}, amostra = {int(X_for_kdist.shape[0])})"
            )
            st.caption("Use esse valor como ponto de partida e ajuste conforme ruído/fragmentação.")
    except Exception as e:
        st.warning(
            "Nao foi possivel calcular o k-distance com esta combinacao de parametros. "
            f"Detalhe tecnico: {e}"
        )

    run_dbscan = st.button("Rodar DBSCAN", use_container_width=True, key="ml_unsup_dbscan_run")
    if run_dbscan:
        from sklearn.cluster import DBSCAN

        _clear_dbscan_cache()

        db_kwargs = {
            "eps": float(eps),
            "min_samples": int(min_samples),
            "metric": metric,
            "algorithm": algorithm,
            "leaf_size": int(leaf_size),
        }
        if metric == "cosine" and db_kwargs["algorithm"] in ("auto", "ball_tree", "kd_tree"):
            db_kwargs["algorithm"] = "brute"
            st.caption("Ajuste automatico: metric='cosine' executado com algorithm='brute'.")
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
