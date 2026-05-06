import pandas as pd
import streamlit as st

from modules.unsupervised_common import get_numeric_columns, prepare_feature_matrix


SPHERICAL_FINAL_KEY = "ml_unsup_spherical_final"
SPHERICAL_EXPLORE_KEY = "ml_unsup_spherical_explore"
SPHERICAL_ARI_KEY = "ml_unsup_spherical_ari"
SPHERICAL_COLORWAY = ["#7159c1", "#2ecc71", "#3498db", "#f1c40f", "#e74c3c", "#1abc9c", "#e67e22"]


def _clear_spherical_cache():
    st.session_state.pop(SPHERICAL_FINAL_KEY, None)


def _evaluate_spherical_stability_ari(
    *,
    X_spherical,
    k: int,
    init_method: str,
    n_init: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    base_seed: int,
    n_runs: int = 10,
):
    from itertools import combinations

    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    labels_list = []
    for i in range(int(n_runs)):
        model = _build_spherical_model(
            k=int(k),
            init_method=init_method,
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            algorithm=algorithm,
            random_state=int(base_seed + i) if int(base_seed) >= 0 else -1,
        )
        labels_list.append(model.fit_predict(X_spherical))

    aris = []
    for i, j in combinations(range(len(labels_list)), 2):
        aris.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))

    if len(aris) == 0:
        return float("nan"), float("nan")
    return float(np.mean(aris)), float(np.std(aris))


def _apply_plotly_theme(fig, theme_choice: str):
    template = "plotly_dark" if theme_choice == "Escura" else "plotly_white"
    if theme_choice == "Escura":
        paper_bg = "#0E1117"
        plot_bg = "#0E1117"
        font_color = "#FFFFFF"
        grid_color = "#22262e"
    else:
        paper_bg = "#FFFFFF"
        plot_bg = "#FFFFFF"
        font_color = "#0E1117"
        grid_color = "#e5e7eb"

    fig.update_layout(
        template=template,
        colorway=SPHERICAL_COLORWAY,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font={"color": font_color},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)


def _figure_html_bytes(fig, theme_choice: str) -> bytes:
    import plotly.graph_objects as go

    fig_export = go.Figure(fig.to_dict())
    _apply_plotly_theme(fig_export, theme_choice)
    return fig_export.to_html(include_plotlyjs="cdn").encode("utf-8")


def _render_html_downloads(fig, *, base_file_name: str, key_prefix: str):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Download HTML (Escuro)",
            data=_figure_html_bytes(fig, "Escura"),
            file_name=f"{base_file_name}_dark.html",
            mime="text/html",
            use_container_width=True,
            key=f"{key_prefix}_dark",
        )
    with c2:
        st.download_button(
            "📥 Download HTML (Claro)",
            data=_figure_html_bytes(fig, "Clara"),
            file_name=f"{base_file_name}_light.html",
            mime="text/html",
            use_container_width=True,
            key=f"{key_prefix}_light",
        )


def _build_metric_fig(
    ks,
    values,
    *,
    title: str,
    y_label: str,
    marker_k=None,
    marker_label: str = "best",
    theme: str = "dark",
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    dark_bg, white, purple = "#0E1117", "#FFFFFF", "#7159c1"
    if theme == "dark":
        bg, fg, grid = dark_bg, white, "#22262e"
    else:
        bg, fg, grid = "#FFFFFF", "#0E1117", "#e5e7eb"

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.plot(ks, values, marker="o", linewidth=2.0, color=purple)
    ax.set_xlabel("k", color=fg)
    ax.set_ylabel(y_label, color=fg)
    ax.set_title(title, color=fg, pad=10)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    ax.grid(True, color=grid, alpha=0.4)

    if marker_k is not None:
        try:
            yk = float(pd.Series(values, index=ks).loc[int(marker_k)])
            if np.isfinite(yk):
                ax.axvline(int(marker_k), linestyle="--", color=fg, alpha=0.7)
                ax.text(int(marker_k), yk, f"  {marker_label}~{int(marker_k)}", va="bottom", color=fg)
        except Exception:
            pass

    fig.tight_layout()
    return fig


def _render_png_downloads(fig_dark, fig_light, *, base_file_name: str, key_prefix: str):
    import io

    buf_dark, buf_light = io.BytesIO(), io.BytesIO()
    fig_dark.savefig(
        buf_dark,
        format="png",
        dpi=200,
        facecolor=fig_dark.get_facecolor(),
        edgecolor="none",
    )
    fig_light.savefig(
        buf_light,
        format="png",
        dpi=200,
        facecolor=fig_light.get_facecolor(),
        edgecolor="none",
    )
    buf_dark.seek(0)
    buf_light.seek(0)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download PNG (Escuro)",
            data=buf_dark.getvalue(),
            file_name=f"{base_file_name}_dark.png",
            mime="image/png",
            use_container_width=True,
            key=f"{key_prefix}_dark_png",
        )
    with c2:
        st.download_button(
            "Download PNG (Claro)",
            data=buf_light.getvalue(),
            file_name=f"{base_file_name}_light.png",
            mime="image/png",
            use_container_width=True,
            key=f"{key_prefix}_light_png",
        )


def _normalize_rows_l2(X):
    import numpy as np
    from sklearn.preprocessing import normalize

    row_norms = np.linalg.norm(X, axis=1)
    zero_rows = int((row_norms == 0).sum())
    X_unit = normalize(X, norm="l2", axis=1, copy=True)
    return X_unit, zero_rows


def _elbow_by_max_distance(ks, values):
    import numpy as np

    x1, y1 = float(ks[0]), float(values[0])
    x2, y2 = float(ks[-1]), float(values[-1])
    dx, dy = (x2 - x1), (y2 - y1)
    denom = float((dy**2 + dx**2) ** 0.5) if (dx != 0 or dy != 0) else 1.0

    dists = []
    for k, y in zip(ks, values):
        num = abs(dy * float(k) - dx * float(y) + (x2 * y1 - y2 * x1))
        dists.append(num / denom)
    return int(ks[int(np.argmax(dists))])


def _build_spherical_model(*, k, init_method, n_init, max_iter, tol, algorithm, random_state):
    from sklearn.cluster import KMeans

    seed = None if int(random_state) < 0 else int(random_state)
    return KMeans(
        n_clusters=int(k),
        init=str(init_method),
        n_init=int(n_init),
        max_iter=int(max_iter),
        tol=float(tol),
        algorithm=str(algorithm),
        random_state=seed,
    )


def _save_spherical_to_state(
    *,
    labels,
    used_index,
    cols,
    centers_unit,
    X_spherical,
    inertia,
    silhouette,
    ch_score,
    db_score,
    cluster_counts,
    zero_rows,
):
    st.session_state[SPHERICAL_FINAL_KEY] = {
        "labels": list(labels),
        "used_index": list(used_index),
        "cols": list(cols),
        "centers_unit": centers_unit.tolist(),
        "X_spherical": X_spherical.tolist(),
        "metrics": {
            "inertia": float(inertia),
            "silhouette": (None if pd.isna(silhouette) else float(silhouette)),
            "CH": (None if pd.isna(ch_score) else float(ch_score)),
            "DB": (None if pd.isna(db_score) else float(db_score)),
        },
        "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
        "zero_rows": int(zero_rows),
    }


def _render_spherical_cache(df: pd.DataFrame):
    import numpy as np
    import plotly.express as px
    from sklearn.decomposition import PCA

    cache = st.session_state.get(SPHERICAL_FINAL_KEY)
    if not cache:
        return False

    metrics = cache["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Inertia (L2)", f"{metrics['inertia']:,.2f}")
    with c2:
        st.metric("Silhouette (cosine)", "n/a" if metrics["silhouette"] is None else f"{metrics['silhouette']:.4f}")
    with c3:
        st.metric("Calinski-Harabasz", "n/a" if metrics["CH"] is None else f"{metrics['CH']:,.1f}")
    with c4:
        st.metric("Davies-Bouldin", "n/a" if metrics["DB"] is None else f"{metrics['DB']:.3f}")

    counts = pd.Series(cache["cluster_counts"]).sort_index()
    counts.index = [f"cluster_{i}" for i in counts.index]
    st.write("#### Tamanho dos clusters")
    st.dataframe(counts.rename("n").to_frame(), use_container_width=True)

    centers_unit = np.array(cache["centers_unit"])
    centers_df = pd.DataFrame(centers_unit, columns=cache["cols"])
    centers_df.index = [f"cluster_{i}" for i in range(centers_df.shape[0])]
    st.write("#### Centroides unitarios (espaco esferico)")
    st.dataframe(centers_df, use_container_width=True)

    if int(cache.get("zero_rows", 0)) > 0:
        st.caption(
            f"{int(cache['zero_rows'])} linha(s) ficaram com norma zero apos preprocessamento; "
            "elas permanecem vetores nulos na normalizacao L2."
        )

    labels = np.array(cache["labels"])
    used_index = pd.Index(cache["used_index"])
    labeled = df.loc[used_index].assign(cluster_spherical=labels)

    pca_2d = PCA(n_components=2, random_state=42)
    emb2 = pca_2d.fit_transform(np.array(cache["X_spherical"]))
    emb2_df = pd.DataFrame(
        {
            "PC1": emb2[:, 0],
            "PC2": emb2[:, 1],
            "cluster": [f"cluster_{int(x)}" for x in labels],
        },
        index=used_index,
    )
    fig2d = px.scatter(
        emb2_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Spherical K-Means: Projecao PCA 2D",
        opacity=0.85,
        color_discrete_sequence=SPHERICAL_COLORWAY,
    )
    fig2d.update_layout(legend_title_text="Cluster")
    _apply_plotly_theme(fig2d, "Escura")
    st.plotly_chart(fig2d, use_container_width=True)
    _render_html_downloads(fig2d, base_file_name="spherical_pca_2d", key_prefix="ml_unsup_spherical_download_pca2d")

    X_plot = np.array(cache["X_spherical"])
    max_3d_components = min(X_plot.shape[0], X_plot.shape[1])
    if max_3d_components >= 3:
        pca_3d = PCA(n_components=3, random_state=42)
        emb3 = pca_3d.fit_transform(X_plot)
        emb3_df = pd.DataFrame(
            {
                "PC1": emb3[:, 0],
                "PC2": emb3[:, 1],
                "PC3": emb3[:, 2],
                "cluster": [f"cluster_{int(x)}" for x in labels],
            },
            index=used_index,
        )
        fig3d = px.scatter_3d(
            emb3_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="cluster",
            title="Spherical K-Means: Projecao PCA 3D",
            opacity=0.8,
            color_discrete_sequence=SPHERICAL_COLORWAY,
        )
        fig3d.update_layout(legend_title_text="Cluster")
        _apply_plotly_theme(fig3d, "Escura")
        st.plotly_chart(fig3d, use_container_width=True)
        _render_html_downloads(fig3d, base_file_name="spherical_pca_3d", key_prefix="ml_unsup_spherical_download_pca3d")
    else:
        st.caption("Projecao 3D indisponivel: sao necessarios pelo menos 3 componentes (amostras/features).")

    st.write("#### Amostra rotulada")
    st.dataframe(labeled.head(20), use_container_width=True)

    csv_bytes = labeled.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download (CSV Spherical K-Means)",
        data=csv_bytes,
        file_name="spherical_kmeans_clusters.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_unsup_spherical_download_csv",
    )
    return True


def render_spherical_kmeans(df: pd.DataFrame):
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Selecione um DataFrame com pelo menos 2 variaveis numericas.")
        return

    st.markdown("#### Configuracoes do Spherical K-Means")

    cols = st.multiselect(
        "Variaveis numericas (features) para clusterizacao:",
        numeric_cols,
        key="ml_unsup_spherical_cols",
        help="Escolha as colunas que irao compor o espaco de clusterizacao.",
    )
    if not cols or len(cols) < 2:
        st.info("Selecione ao menos 2 variaveis para continuar.")
        return

    c1, c2 = st.columns(2)
    with c1:
        missing_strategy = st.selectbox(
            "Missing values:",
            ["Excluir linhas com NA", "Imputar media"],
            key="ml_unsup_spherical_missing",
        )
    with c2:
        scaler_choice = st.selectbox(
            "Escalonamento:",
            ["Nenhum", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            key="ml_unsup_spherical_scaler",
            help="A normalizacao L2 sera aplicada depois do escalonamento escolhido.",
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

    X_spherical, zero_rows = _normalize_rows_l2(X_model)

    init_method = st.session_state.get("ml_unsup_spherical_init", "k-means++")
    n_init = int(st.session_state.get("ml_unsup_spherical_n_init", 10))
    max_iter = int(st.session_state.get("ml_unsup_spherical_max_iter", 300))
    tol = float(st.session_state.get("ml_unsup_spherical_tol", 1e-4))
    algorithm = st.session_state.get("ml_unsup_spherical_algorithm", "lloyd")
    random_state = int(st.session_state.get("ml_unsup_spherical_random_state", 42))
    n_clusters = int(st.session_state.get("ml_unsup_spherical_k_final", 3))
    max_k = int(min(20, X_spherical.shape[0]))

    p1, p2, p3 = st.columns(3)
    with p1:
        n_clusters = st.number_input(
            "K (modelo final)",
            min_value=1,
            max_value=max_k,
            value=min(max(1, int(n_clusters)), max_k),
            step=1,
            key="ml_unsup_spherical_k_final",
        )
        init_method = st.selectbox(
            "init",
            ["k-means++", "random"],
            index=(0 if init_method == "k-means++" else 1),
            key="ml_unsup_spherical_init",
        )
    with p2:
        n_init = st.number_input(
            "n_init",
            min_value=1,
            value=int(n_init),
            step=1,
            key="ml_unsup_spherical_n_init",
        )
        max_iter = st.number_input(
            "max_iter",
            min_value=10,
            value=int(max_iter),
            step=10,
            key="ml_unsup_spherical_max_iter",
        )
    with p3:
        tol = st.number_input(
            "tol",
            min_value=1e-10,
            value=float(tol),
            format="%.1e",
            key="ml_unsup_spherical_tol",
        )
        algorithm = st.selectbox(
            "algorithm",
            ["lloyd", "elkan"],
            index=(0 if algorithm == "lloyd" else 1),
            key="ml_unsup_spherical_algorithm",
        )

    random_state = st.number_input(
        "random_state (opcional, -1=desligado)",
        value=int(random_state),
        step=1,
        key="ml_unsup_spherical_random_state",
    )

    if max_k >= 2:
        default_k_range = (2, min(10, max_k))
    else:
        default_k_range = (1, 1)

    k_min, k_max = st.slider(
        "Intervalo para explorar K (spherical):",
        min_value=1,
        max_value=max_k,
        value=default_k_range,
        key="ml_unsup_spherical_k_range",
    )

    run_explore = st.button(
        "Explorar K (Spherical)",
        use_container_width=True,
        key="ml_unsup_spherical_explore_btn",
    )
    if run_explore:
        import numpy as np

        ks = list(range(int(k_min), int(k_max) + 1))
        inertias = []
        silhouettes = []

        for k in ks:
            model = _build_spherical_model(
                k=k,
                init_method=st.session_state["ml_unsup_spherical_init"],
                n_init=st.session_state["ml_unsup_spherical_n_init"],
                max_iter=st.session_state["ml_unsup_spherical_max_iter"],
                tol=st.session_state["ml_unsup_spherical_tol"],
                algorithm=st.session_state["ml_unsup_spherical_algorithm"],
                random_state=st.session_state["ml_unsup_spherical_random_state"],
            )
            labels = model.fit_predict(X_spherical)
            inertias.append(float(model.inertia_))

            if k >= 2 and X_spherical.shape[0] > k:
                try:
                    silhouettes.append(float(silhouette_score(X_spherical, labels, metric="cosine")))
                except Exception:
                    silhouettes.append(float("nan"))
            else:
                silhouettes.append(float("nan"))

        silhouette_arr = np.array(silhouettes, dtype=float)
        silhouette_k = None if np.all(np.isnan(silhouette_arr)) else int(ks[int(np.nanargmax(silhouette_arr))])
        elbow_k = _elbow_by_max_distance(ks, inertias)

        st.session_state[SPHERICAL_EXPLORE_KEY] = {
            "ks": ks,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "elbow_k": elbow_k,
            "silhouette_k": silhouette_k,
        }

    explore_cache = st.session_state.get(SPHERICAL_EXPLORE_KEY)
    if explore_cache:
        import matplotlib.pyplot as plt

        res = pd.DataFrame(
            {
                "k": explore_cache["ks"],
                "inertia_l2": explore_cache["inertias"],
                "silhouette_cosine": explore_cache["silhouettes"],
            }
        )
        st.write("#### Resultados da exploracao")
        st.dataframe(res, use_container_width=True)
        st.caption(
            f"Sugestao por elbow: K={explore_cache['elbow_k']} | "
            f"Sugestao por silhouette: {explore_cache['silhouette_k'] if explore_cache['silhouette_k'] is not None else 'n/a'}"
        )

        fig_elbow_dark = _build_metric_fig(
            explore_cache["ks"],
            explore_cache["inertias"],
            title="Elbow (Inertia L2 por K)",
            y_label="Inertia (L2)",
            marker_k=explore_cache["elbow_k"],
            marker_label="elbow",
            theme="dark",
        )
        fig_elbow_light = _build_metric_fig(
            explore_cache["ks"],
            explore_cache["inertias"],
            title="Elbow (Inertia L2 por K)",
            y_label="Inertia (L2)",
            marker_k=explore_cache["elbow_k"],
            marker_label="elbow",
            theme="light",
        )
        st.pyplot(fig_elbow_dark, clear_figure=True)
        _render_png_downloads(
            fig_elbow_dark,
            fig_elbow_light,
            base_file_name="spherical_elbow",
            key_prefix="ml_unsup_spherical_download_elbow",
        )
        plt.close(fig_elbow_dark)
        plt.close(fig_elbow_light)

        sil_df = res.dropna(subset=["silhouette_cosine"])
        if not sil_df.empty:
            ks_sil = sil_df["k"].tolist()
            vals_sil = sil_df["silhouette_cosine"].tolist()
            fig_sil_dark = _build_metric_fig(
                ks_sil,
                vals_sil,
                title="Silhouette Cosine por K",
                y_label="Silhouette (cosine)",
                marker_k=explore_cache["silhouette_k"],
                marker_label="best",
                theme="dark",
            )
            fig_sil_light = _build_metric_fig(
                ks_sil,
                vals_sil,
                title="Silhouette Cosine por K",
                y_label="Silhouette (cosine)",
                marker_k=explore_cache["silhouette_k"],
                marker_label="best",
                theme="light",
            )
            st.pyplot(fig_sil_dark, clear_figure=True)
            _render_png_downloads(
                fig_sil_dark,
                fig_sil_light,
                base_file_name="spherical_silhouette",
                key_prefix="ml_unsup_spherical_download_silhouette",
            )
            plt.close(fig_sil_dark)
            plt.close(fig_sil_light)

    st.markdown("#### Estabilidade (ARI)")
    s1, s2, s3 = st.columns(3)
    with s1:
        k_for_stability = st.number_input(
            "K para avaliar",
            min_value=2,
            max_value=max_k,
            value=min(max(2, int(n_clusters)), max_k),
            step=1,
            key="ml_unsup_spherical_k_stability",
        )
    with s2:
        n_runs = st.number_input(
            "n execucoes (runs)",
            min_value=2,
            value=10,
            step=1,
            key="ml_unsup_spherical_n_runs",
        )
    with s3:
        base_seed = st.number_input(
            "base_seed (usa base_seed+i)",
            value=int(random_state),
            step=1,
            key="ml_unsup_spherical_base_seed",
        )

    run_stability = st.button(
        "Calcular estabilidade (ARI)",
        use_container_width=True,
        key="ml_unsup_spherical_run_ari_btn",
    )
    if run_stability:
        ari_mean, ari_std = _evaluate_spherical_stability_ari(
            X_spherical=X_spherical,
            k=int(k_for_stability),
            init_method=st.session_state["ml_unsup_spherical_init"],
            n_init=int(st.session_state["ml_unsup_spherical_n_init"]),
            max_iter=int(st.session_state["ml_unsup_spherical_max_iter"]),
            tol=float(st.session_state["ml_unsup_spherical_tol"]),
            algorithm=st.session_state["ml_unsup_spherical_algorithm"],
            base_seed=int(base_seed),
            n_runs=int(n_runs),
        )
        st.session_state[SPHERICAL_ARI_KEY] = {
            "k": int(k_for_stability),
            "n_runs": int(n_runs),
            "base_seed": int(base_seed),
            "ari_mean": ari_mean,
            "ari_std": ari_std,
        }

    ari_cache = st.session_state.get(SPHERICAL_ARI_KEY)
    if ari_cache:
        ari_std_txt = "n/a" if pd.isna(ari_cache["ari_std"]) else f"{ari_cache['ari_std']:.3f}"
        st.metric(
            f"ARI medio (K={int(ari_cache['k'])})",
            "n/a" if pd.isna(ari_cache["ari_mean"]) else f"{ari_cache['ari_mean']:.3f}",
        )
        st.caption(
            f"Desvio-padrao do ARI: {ari_std_txt} "
            "valores proximos de 1 indicam particoes mais estaveis."
        )

    run_model = st.button(
        "Rodar Spherical K-Means",
        use_container_width=True,
        key="ml_unsup_spherical_run_btn",
    )
    if run_model:
        import numpy as np
        from sklearn.preprocessing import normalize

        _clear_spherical_cache()

        model = _build_spherical_model(
            k=st.session_state["ml_unsup_spherical_k_final"],
            init_method=st.session_state["ml_unsup_spherical_init"],
            n_init=st.session_state["ml_unsup_spherical_n_init"],
            max_iter=st.session_state["ml_unsup_spherical_max_iter"],
            tol=st.session_state["ml_unsup_spherical_tol"],
            algorithm=st.session_state["ml_unsup_spherical_algorithm"],
            random_state=st.session_state["ml_unsup_spherical_random_state"],
        )
        labels = model.fit_predict(X_spherical)

        unique_labels = set(labels.tolist())
        can_score = len(unique_labels) >= 2 and X_spherical.shape[0] > len(unique_labels)

        if can_score:
            sil = float(silhouette_score(X_spherical, labels, metric="cosine"))
            ch_score = float(calinski_harabasz_score(X_spherical, labels))
            db_score = float(davies_bouldin_score(X_spherical, labels))
        else:
            sil = float("nan")
            ch_score = float("nan")
            db_score = float("nan")

        centers_unit = normalize(model.cluster_centers_, norm="l2", axis=1, copy=True)
        counts = pd.Series(labels).value_counts().sort_index()

        _save_spherical_to_state(
            labels=labels,
            used_index=X_df.index,
            cols=cols,
            centers_unit=centers_unit,
            X_spherical=X_spherical,
            inertia=float(model.inertia_),
            silhouette=sil,
            ch_score=ch_score,
            db_score=db_score,
            cluster_counts=counts.to_dict(),
            zero_rows=zero_rows,
        )
        st.success("Spherical K-Means executado e armazenado.")

    _render_spherical_cache(df)
