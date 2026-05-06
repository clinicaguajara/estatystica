import pandas as pd
import streamlit as st

from modules.unsupervised_common import get_numeric_columns, prepare_feature_matrix


FUZZY_FINAL_KEY = "ml_unsup_fuzzy_final"
FUZZY_COLORWAY = ["#7159c1", "#2ecc71", "#3498db", "#f1c40f", "#e74c3c", "#1abc9c", "#e67e22"]


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
        colorway=FUZZY_COLORWAY,
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
            "📥 Download HTML (🌙 escuro)",
            data=_figure_html_bytes(fig, "Escura"),
            file_name=f"{base_file_name}_dark.html",
            mime="text/html",
            use_container_width=True,
            key=f"{key_prefix}_dark",
        )
    with c2:
        st.download_button(
            "📥 Download HTML (☀️ claro)",
            data=_figure_html_bytes(fig, "Clara"),
            file_name=f"{base_file_name}_light.html",
            mime="text/html",
            use_container_width=True,
            key=f"{key_prefix}_light",
        )


def _run_fcm(X, *, n_clusters: int, m: float, max_iter: int, tol: float, random_state: int):
    import numpy as np

    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    c = int(n_clusters)
    m = float(m)
    eps = 1e-12

    rng = np.random.default_rng(None if int(random_state) < 0 else int(random_state))
    U = rng.random((n_samples, c))
    U = U / U.sum(axis=1, keepdims=True)

    prev_J = None
    converged = False

    for i in range(int(max_iter)):
        Um = U ** m
        denom = Um.sum(axis=0)[:, None]
        V = (Um.T @ X) / np.maximum(denom, eps)

        D = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)
        D = np.maximum(D, eps)

        zero_mask = D <= eps * 10
        if np.any(zero_mask):
            U_new = np.zeros_like(U)
            rows = np.where(zero_mask.any(axis=1))[0]
            for r in rows:
                idx = np.where(zero_mask[r])[0]
                U_new[r, idx] = 1.0 / len(idx)
            non_rows = np.where(~zero_mask.any(axis=1))[0]
            if len(non_rows) > 0:
                D_non = D[non_rows]
                power = 2.0 / (m - 1.0)
                ratio = (D_non[:, :, None] / D_non[:, None, :]) ** power
                U_new[non_rows] = 1.0 / ratio.sum(axis=2)
        else:
            power = 2.0 / (m - 1.0)
            ratio = (D[:, :, None] / D[:, None, :]) ** power
            U_new = 1.0 / ratio.sum(axis=2)

        J = float(np.sum((U_new ** m) * (D ** 2)))
        if prev_J is not None and abs(prev_J - J) <= float(tol):
            converged = True
            U = U_new
            break

        U = U_new
        prev_J = J

    Um = U ** m
    denom = Um.sum(axis=0)[:, None]
    V = (Um.T @ X) / np.maximum(denom, eps)
    D = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)
    D = np.maximum(D, eps)
    J_final = float(np.sum((U ** m) * (D ** 2)))
    hard_labels = U.argmax(axis=1)

    fpc = float(np.sum(U ** 2) / n_samples)
    pe = float(-np.sum(U * np.log(np.maximum(U, eps))) / n_samples)

    return {
        "U": U,
        "V": V,
        "hard_labels": hard_labels,
        "Jm": J_final,
        "fpc": fpc,
        "pe": pe,
        "n_iter": i + 1,
        "converged": converged,
    }


def render_fuzzy(df: pd.DataFrame):
    import numpy as np
    import plotly.express as px
    from sklearn.decomposition import PCA
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Selecione um DataFrame com pelo menos 2 variaveis numericas.")
        return

    st.markdown("#### Configuracoes do Fuzzy C-Means")

    cols = st.multiselect(
        "Variaveis numericas (features) para clusterizacao:",
        numeric_cols,
        key="ml_unsup_fuzzy_cols",
    )
    if not cols or len(cols) < 2:
        st.info("Selecione ao menos 2 variaveis para continuar.")
        return

    c1, c2 = st.columns(2)
    with c1:
        missing_strategy = st.selectbox(
            "Missing values:",
            ["Excluir linhas com NA", "Imputar media"],
            key="ml_unsup_fuzzy_missing",
        )
    with c2:
        scaler_choice = st.selectbox(
            "Escalonamento:",
            ["Nenhum", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            key="ml_unsup_fuzzy_scaler",
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

    max_k = int(min(20, X_model.shape[0]))
    p1, p2, p3 = st.columns(3)
    with p1:
        n_clusters = st.number_input("n_clusters", min_value=2, max_value=max_k, value=min(3, max_k), step=1, key="ml_unsup_fuzzy_k")
        fuzzifier = st.number_input("m (fuzzifier)", min_value=1.01, max_value=5.0, value=2.0, step=0.05, key="ml_unsup_fuzzy_m")
    with p2:
        max_iter = st.number_input("max_iter", min_value=10, max_value=5000, value=300, step=10, key="ml_unsup_fuzzy_max_iter")
        tol = st.number_input("tol", min_value=1e-10, value=1e-4, format="%.1e", key="ml_unsup_fuzzy_tol")
    with p3:
        random_state = st.number_input(
            "random_state (opcional, -1=desligado)",
            value=42,
            step=1,
            key="ml_unsup_fuzzy_random_state",
        )

    run_model = st.button("Rodar Fuzzy C-Means", use_container_width=True, key="ml_unsup_fuzzy_run")
    if run_model:
        try:
            out = _run_fcm(
                X_model,
                n_clusters=int(n_clusters),
                m=float(fuzzifier),
                max_iter=int(max_iter),
                tol=float(tol),
                random_state=int(random_state),
            )
        except Exception as e:
            st.error(f"Falha ao rodar Fuzzy C-Means: {e}")
            return

        labels = out["hard_labels"]
        unique_labels = set(labels.tolist())
        can_score = len(unique_labels) >= 2 and X_model.shape[0] > len(unique_labels)
        if can_score:
            sil = float(silhouette_score(X_model, labels))
            ch_score = float(calinski_harabasz_score(X_model, labels))
            db_score = float(davies_bouldin_score(X_model, labels))
        else:
            sil = float("nan")
            ch_score = float("nan")
            db_score = float("nan")

        st.session_state[FUZZY_FINAL_KEY] = {
            "used_index": X_df.index.tolist(),
            "cols": cols,
            "X_model": X_model.tolist(),
            "U": out["U"].tolist(),
            "V": out["V"].tolist(),
            "labels": labels.tolist(),
            "Jm": out["Jm"],
            "fpc": out["fpc"],
            "pe": out["pe"],
            "n_iter": out["n_iter"],
            "converged": out["converged"],
            "silhouette": (None if pd.isna(sil) else sil),
            "CH": (None if pd.isna(ch_score) else ch_score),
            "DB": (None if pd.isna(db_score) else db_score),
        }
        st.success("Fuzzy C-Means executado e armazenado.")

    cache = st.session_state.get(FUZZY_FINAL_KEY)
    if not cache:
        return

    labels = np.array(cache["labels"], dtype=int)
    U = np.array(cache["U"], dtype=float)
    X_arr = np.array(cache["X_model"], dtype=float)
    used_index = pd.Index(cache["used_index"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Jm (objetivo)", f"{float(cache['Jm']):,.4f}")
    with c2:
        st.metric("FPC (maior melhor)", f"{float(cache['fpc']):.4f}")
    with c3:
        st.metric("Partition Entropy", f"{float(cache['pe']):.4f}")
    with c4:
        st.metric("Convergiu", "sim" if cache.get("converged", False) else "nao")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Iteracoes", f"{int(cache['n_iter'])}")
    with c6:
        st.metric("Silhouette", "n/a" if cache["silhouette"] is None else f"{float(cache['silhouette']):.4f}")
    with c7:
        st.metric("Calinski-Harabasz", "n/a" if cache["CH"] is None else f"{float(cache['CH']):,.1f}")
    with c8:
        st.metric("Davies-Bouldin", "n/a" if cache["DB"] is None else f"{float(cache['DB']):.3f}")

    proj = PCA(n_components=2, random_state=42).fit_transform(X_arr)
    confidence = U.max(axis=1)
    plot_df = pd.DataFrame(
        {
            "PC1": proj[:, 0],
            "PC2": proj[:, 1],
            "cluster": [f"cluster_{int(c)}" for c in labels],
            "confianca": confidence,
            "sujeito": [str(x) for x in used_index.tolist()],
        }
    )
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="cluster",
        size="confianca",
        title="Fuzzy C-Means: Projecao PCA 2D",
        color_discrete_sequence=FUZZY_COLORWAY,
        hover_data={"sujeito": True, "confianca": ":.4f", "PC1": ":.4f", "PC2": ":.4f"},
    )
    fig.update_traces(marker={"opacity": 0.85})
    fig.update_layout(legend_title_text="Cluster")
    _apply_plotly_theme(fig, "Escura")
    st.plotly_chart(fig, use_container_width=True)
    _render_html_downloads(fig, base_file_name="fuzzy_pca_2d", key_prefix="ml_unsup_fuzzy_download_pca2d")

    counts = pd.Series(labels).value_counts().sort_index()
    counts.index = [f"cluster_{i}" for i in counts.index]
    st.write("#### Tamanho dos clusters (hard label)")
    st.dataframe(counts.rename("n").to_frame(), use_container_width=True)

    proba_cols = [f"u_cluster_{i}" for i in range(U.shape[1])]
    labeled = df.loc[used_index].copy()
    labeled["cluster_fuzzy"] = labels
    labeled["confianca"] = confidence
    sorted_u = np.sort(U, axis=1)
    second_best = sorted_u[:, -2] if U.shape[1] >= 2 else np.zeros(U.shape[0], dtype=float)
    labeled["margem_top2"] = confidence - second_best
    for i, col_name in enumerate(proba_cols):
        labeled[col_name] = U[:, i]

    st.write("#### Casos mais ambiguos (top 20)")
    ambiguous = labeled.sort_values(by=["confianca", "margem_top2"], ascending=[True, True])
    st.dataframe(ambiguous.head(20), use_container_width=True)

    st.write("#### Amostra rotulada")
    st.dataframe(labeled.head(20), use_container_width=True)

    st.download_button(
        "Download (CSV Fuzzy C-Means)",
        data=labeled.to_csv(index=True).encode("utf-8"),
        file_name="fuzzy_cmeans_clusters.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_unsup_fuzzy_download_csv",
    )
