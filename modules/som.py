import pandas as pd
import streamlit as st

from modules.unsupervised_common import get_numeric_columns, prepare_feature_matrix


SOM_FINAL_KEY = "ml_unsup_som_final"


def _clear_som_cache():
    st.session_state.pop(SOM_FINAL_KEY, None)


def _train_som(
    X_model,
    *,
    map_rows: int,
    map_cols: int,
    sigma: float,
    learning_rate: float,
    n_iter: int,
    random_state: int,
):
    import numpy as np

    X = np.asarray(X_model, dtype=float)
    n_samples, n_features = X.shape
    n_nodes = int(map_rows) * int(map_cols)

    rng = np.random.default_rng(None if int(random_state) < 0 else int(random_state))
    replace = bool(n_samples < n_nodes)
    init_idx = rng.choice(n_samples, size=n_nodes, replace=replace)
    weights = X[init_idx].copy()

    coords = np.indices((int(map_rows), int(map_cols))).reshape(2, -1).T.astype(float)
    coord_sq_dists = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)

    n_iter = max(1, int(n_iter))
    for t in range(n_iter):
        x = X[int(rng.integers(0, n_samples))]
        sq_dists = ((weights - x) ** 2).sum(axis=1)
        bmu = int(sq_dists.argmin())

        frac = 1.0 - (float(t) / float(max(1, n_iter - 1)))
        lr_t = float(learning_rate) * frac
        sigma_t = max(1e-6, float(sigma) * frac)

        neigh = np.exp(-coord_sq_dists[bmu] / (2.0 * sigma_t * sigma_t))
        weights += lr_t * neigh[:, None] * (x - weights)

    return weights, coords


def _assign_bmu(X_model, weights_flat, map_cols: int):
    from sklearn.metrics import pairwise_distances_argmin_min
    import numpy as np

    bmu_flat, min_dist = pairwise_distances_argmin_min(
        X_model, weights_flat, axis=1, metric="euclidean"
    )
    bmu_flat = np.asarray(bmu_flat, dtype=int)
    min_dist = np.asarray(min_dist, dtype=float)
    bmu_row = bmu_flat // int(map_cols)
    bmu_col = bmu_flat % int(map_cols)
    return bmu_flat, bmu_row, bmu_col, min_dist


def _compute_topographic_error(X_model, weights_flat, map_rows: int, map_cols: int):
    import numpy as np

    X = np.asarray(X_model, dtype=float)
    W = np.asarray(weights_flat, dtype=float)
    if X.size == 0 or W.shape[0] < 2:
        return float("nan")

    d2 = ((X[:, None, :] - W[None, :, :]) ** 2).sum(axis=2)
    nearest2 = np.argpartition(d2, kth=1, axis=1)[:, :2]

    bmu1 = nearest2[:, 0]
    bmu2 = nearest2[:, 1]
    swap = d2[np.arange(d2.shape[0]), bmu2] < d2[np.arange(d2.shape[0]), bmu1]
    if np.any(swap):
        bmu1_s = bmu1.copy()
        bmu1[swap] = bmu2[swap]
        bmu2[swap] = bmu1_s[swap]

    r1, c1 = bmu1 // int(map_cols), bmu1 % int(map_cols)
    r2, c2 = bmu2 // int(map_cols), bmu2 % int(map_cols)
    manhattan = np.abs(r1 - r2) + np.abs(c1 - c2)

    return float(np.mean(manhattan > 1))


def _compute_u_matrix(weights_flat, map_rows: int, map_cols: int):
    import numpy as np

    W = weights_flat.reshape(int(map_rows), int(map_cols), -1)
    u = np.zeros((int(map_rows), int(map_cols)), dtype=float)

    for r in range(int(map_rows)):
        for c in range(int(map_cols)):
            neigh = []
            if r > 0:
                neigh.append(W[r - 1, c])
            if r < int(map_rows) - 1:
                neigh.append(W[r + 1, c])
            if c > 0:
                neigh.append(W[r, c - 1])
            if c < int(map_cols) - 1:
                neigh.append(W[r, c + 1])

            if neigh:
                d = [float(np.linalg.norm(W[r, c] - n)) for n in neigh]
                u[r, c] = float(np.mean(d))
            else:
                u[r, c] = 0.0
    return u


def _to_original_space(weights_flat, scaler):
    if scaler is None:
        return weights_flat

    try:
        return scaler.inverse_transform(weights_flat)
    except Exception:
        return weights_flat


def _save_som_to_state(
    *,
    labels_node,
    labels_row,
    labels_col,
    quantization_dist,
    used_index,
    cols,
    map_rows,
    map_cols,
    weights_flat,
    weights_original,
    u_matrix,
    node_counts,
    qe,
    topographic_error,
    empty_nodes,
    scaler_used,
):
    st.session_state[SOM_FINAL_KEY] = {
        "labels_node": list(labels_node),
        "labels_row": list(labels_row),
        "labels_col": list(labels_col),
        "quantization_dist": list(quantization_dist),
        "used_index": list(used_index),
        "cols": list(cols),
        "map_rows": int(map_rows),
        "map_cols": int(map_cols),
        "weights_flat": weights_flat.tolist(),
        "weights_original": weights_original.tolist(),
        "u_matrix": u_matrix.tolist(),
        "node_counts": {int(k): int(v) for k, v in node_counts.items()},
        "qe": float(qe),
        "topographic_error": (None if pd.isna(topographic_error) else float(topographic_error)),
        "empty_nodes": int(empty_nodes),
        "scaler_used": bool(scaler_used),
    }


def _plot_heatmap(matrix, *, title: str, color_label: str, text_values=None):
    import plotly.express as px
    import numpy as np

    arr = np.asarray(matrix)
    fig = px.imshow(
        arr,
        origin="lower",
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": "col", "y": "row", "color": color_label},
        title=title,
    )
    if text_values is not None:
        txt = np.asarray(text_values)
        fig.update_traces(text=txt, texttemplate="%{text}")
    fig.update_layout(margin={"l": 30, "r": 20, "t": 50, "b": 30})
    st.plotly_chart(fig, use_container_width=True)


def _build_codebook_table(weights, feature_names, map_cols: int):
    import numpy as np

    weights_arr = np.asarray(weights, dtype=float)
    n_nodes = int(weights_arr.shape[0])
    codebook = pd.DataFrame(weights_arr, columns=feature_names)
    codebook.insert(0, "col", [i % int(map_cols) for i in range(n_nodes)])
    codebook.insert(0, "row", [i // int(map_cols) for i in range(n_nodes)])
    codebook.insert(0, "node", range(n_nodes))
    return codebook


def _plot_selected_node_profile(weights, feature_names, map_rows: int, map_cols: int):
    import plotly.graph_objects as go
    import numpy as np

    c1, c2, c3 = st.columns(3)
    with c1:
        row = st.number_input(
            "Linha do neuronio",
            min_value=0,
            max_value=int(map_rows) - 1,
            value=0,
            step=1,
            key="ml_unsup_som_profile_row",
        )
    with c2:
        col = st.number_input(
            "Coluna do neuronio",
            min_value=0,
            max_value=int(map_cols) - 1,
            value=0,
            step=1,
            key="ml_unsup_som_profile_col",
        )
    with c3:
        chart_type = st.radio(
            "Grafico",
            ["Barplot", "Radar"],
            horizontal=True,
            key="ml_unsup_som_profile_chart",
        )

    node = int(row) * int(map_cols) + int(col)
    values = np.asarray(weights[node], dtype=float)
    title = f"Codebook vector do neuronio {node} (row={int(row)}, col={int(col)})"

    if chart_type == "Radar":
        theta = list(feature_names) + [feature_names[0]]
        r = values.tolist() + [float(values[0])]
        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    fill="toself",
                    name=f"node_{node}",
                    hovertemplate="%{theta}: %{r:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(title=title, showlegend=False, margin={"l": 30, "r": 30, "t": 55, "b": 30})
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(feature_names),
                    y=values,
                    marker_color="#7159c1",
                    hovertemplate="%{x}: %{y:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=title,
            xaxis_title="Variavel",
            yaxis_title="Peso",
            margin={"l": 40, "r": 20, "t": 55, "b": 80},
        )
        fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)


def _plot_codebook_bar_grid(weights, feature_names, map_rows: int, map_cols: int):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    weights_arr = np.asarray(weights, dtype=float)
    n_nodes = int(weights_arr.shape[0])
    max_nodes_for_grid = 100

    if n_nodes > max_nodes_for_grid:
        st.info(
            "Mini barplots por neuronio ficam pesados para mapas com mais de "
            f"{max_nodes_for_grid} nos. Use os component planes ou inspecione um neuronio."
        )
        return

    h_spacing = min(0.015, 0.8 / max(1, int(map_cols) - 1))
    v_spacing = min(0.045, 0.8 / max(1, int(map_rows) - 1))
    subplot_titles = [f"N{i}" for i in range(n_nodes)]
    fig = make_subplots(
        rows=int(map_rows),
        cols=int(map_cols),
        subplot_titles=subplot_titles,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
    )

    finite_vals = weights_arr[np.isfinite(weights_arr)]
    if finite_vals.size:
        y_min = float(finite_vals.min())
        y_max = float(finite_vals.max())
        pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
        y_range = [y_min - pad, y_max + pad]
    else:
        y_range = None

    for node, values in enumerate(weights_arr):
        row = node // int(map_cols) + 1
        col = node % int(map_cols) + 1
        fig.add_trace(
            go.Bar(
                x=list(feature_names),
                y=values,
                marker_color="#7159c1",
                showlegend=False,
                hovertemplate=f"node {node}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Mini barplots dos codebook vectors",
        height=max(420, int(map_rows) * 150),
        margin={"l": 30, "r": 20, "t": 80, "b": 30},
    )
    fig.update_xaxes(showticklabels=False)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    st.plotly_chart(fig, use_container_width=True)


def _render_codebook_views(cache, map_rows: int, map_cols: int):
    import numpy as np

    feature_names = list(cache["cols"])
    weights_model = np.asarray(cache["weights_flat"], dtype=float)
    weights_original = np.asarray(cache.get("weights_original", cache["weights_flat"]), dtype=float)

    space = st.radio(
        "Espaco dos pesos:",
        ["Original das variaveis", "Modelo treinado"],
        horizontal=True,
        key="ml_unsup_som_codebook_space",
        help="O espaco do modelo e util quando as variaveis foram escalonadas.",
    )
    weights = weights_original if space == "Original das variaveis" else weights_model

    if space == "Original das variaveis" and not cache.get("scaler_used", False):
        st.caption("Sem escalonamento, o espaco original e o espaco do modelo sao iguais.")

    tabs = st.tabs(["Component planes", "Neuronio", "Mini barplots", "Tabela"])

    with tabs[0]:
        default_features = feature_names[: min(4, len(feature_names))]
        selected_features = st.multiselect(
            "Variaveis para visualizar no mapa:",
            feature_names,
            default=default_features,
            key="ml_unsup_som_component_features",
        )
        for feature in selected_features:
            idx = feature_names.index(feature)
            matrix = weights[:, idx].reshape(int(map_rows), int(map_cols))
            _plot_heatmap(
                matrix,
                title=f"Component plane: {feature}",
                color_label="peso",
            )

    with tabs[1]:
        _plot_selected_node_profile(weights, feature_names, map_rows, map_cols)

    with tabs[2]:
        _plot_codebook_bar_grid(weights, feature_names, map_rows, map_cols)

    with tabs[3]:
        codebook = _build_codebook_table(weights, feature_names, map_cols)
        st.dataframe(codebook, use_container_width=True)
        csv_bytes = codebook.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download (CSV codebook SOM)",
            data=csv_bytes,
            file_name="som_codebook_vectors.csv",
            mime="text/csv",
            use_container_width=True,
            key="ml_unsup_som_codebook_download_csv",
        )


def _render_som_cache(df: pd.DataFrame):
    import numpy as np

    cache = st.session_state.get(SOM_FINAL_KEY)
    if not cache:
        return False

    map_rows = int(cache["map_rows"])
    map_cols = int(cache["map_cols"])
    n_nodes = map_rows * map_cols

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Quantization error", f"{float(cache['qe']):.4f}")
    with c2:
        st.metric(
            "Topographic error",
            "n/a" if cache.get("topographic_error") is None else f"{float(cache['topographic_error']):.4f}",
        )
    with c3:
        st.metric("Nos vazios", f"{int(cache['empty_nodes'])}/{n_nodes}")
    with c4:
        st.metric("Total de nos", f"{n_nodes}")

    labels_node = np.array(cache["labels_node"], dtype=int)
    node_counts = pd.Series(cache["node_counts"]).sort_index()
    node_counts = node_counts.reindex(range(n_nodes), fill_value=0)
    occupancy = node_counts.values.reshape(map_rows, map_cols)

    st.write("#### Ocupacao dos nos (amostras por no)")
    _plot_heatmap(
        occupancy,
        title="SOM node occupancy",
        color_label="n amostras",
        text_values=occupancy,
    )

    st.write("#### U-Matrix (distancia media aos vizinhos)")
    _plot_heatmap(
        cache["u_matrix"],
        title="SOM U-Matrix",
        color_label="distancia",
    )

    st.write("#### Codebook vectors (perfil prototipico dos neuronios)")
    _render_codebook_views(cache, map_rows, map_cols)

    labels_row = np.array(cache["labels_row"], dtype=int)
    labels_col = np.array(cache["labels_col"], dtype=int)
    quantization_dist = np.array(cache["quantization_dist"], dtype=float)
    used_index = pd.Index(cache["used_index"])

    labeled = df.loc[used_index].copy()
    labeled["som_node"] = labels_node
    labeled["som_row"] = labels_row
    labeled["som_col"] = labels_col
    labeled["som_quant_error"] = quantization_dist

    st.write("#### Amostra rotulada")
    st.dataframe(labeled.head(20), use_container_width=True)

    st.write("#### Tamanho dos nos")
    node_table = pd.DataFrame(
        {
            "node": range(n_nodes),
            "row": [i // map_cols for i in range(n_nodes)],
            "col": [i % map_cols for i in range(n_nodes)],
            "n": node_counts.values,
        }
    )
    st.dataframe(node_table, use_container_width=True)

    csv_bytes = labeled.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download (CSV SOM)",
        data=csv_bytes,
        file_name="som_clusters.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_unsup_som_download_csv",
    )
    return True


def render_som(df: pd.DataFrame):
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Selecione um DataFrame com pelo menos 2 variaveis numericas.")
        return

    st.markdown("#### Configuracoes do SOM")

    cols = st.multiselect(
        "Variaveis numericas (features) para clusterizacao:",
        numeric_cols,
        key="ml_unsup_som_cols",
        help="Escolha as colunas que irao compor o espaco de treino do SOM.",
    )
    if not cols or len(cols) < 2:
        st.info("Selecione ao menos 2 variaveis para continuar.")
        return

    c1, c2 = st.columns(2)
    with c1:
        missing_strategy = st.selectbox(
            "Missing values:",
            ["Excluir linhas com NA", "Imputar media"],
            key="ml_unsup_som_missing",
        )
    with c2:
        scaler_choice = st.selectbox(
            "Escalonamento:",
            ["Nenhum", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            key="ml_unsup_som_scaler",
            help="SOM usa distancia euclidiana; escalar costuma ajudar.",
        )

    try:
        X_df, X_model, scaler = prepare_feature_matrix(
            df=df,
            cols=cols,
            missing_strategy=missing_strategy,
            scaler_choice=scaler_choice,
        )
    except ValueError as e:
        st.error(str(e))
        return

    n_samples = int(X_model.shape[0])
    map_rows_default = int(st.session_state.get("ml_unsup_som_map_rows", 8))
    map_cols_default = int(st.session_state.get("ml_unsup_som_map_cols", 8))
    sigma_default = float(st.session_state.get("ml_unsup_som_sigma", max(1.0, min(map_rows_default, map_cols_default) / 2.0)))

    p1, p2, p3 = st.columns(3)
    with p1:
        map_rows = st.number_input(
            "map_rows",
            min_value=2,
            max_value=50,
            value=max(2, map_rows_default),
            step=1,
            key="ml_unsup_som_map_rows",
        )
        map_cols = st.number_input(
            "map_cols",
            min_value=2,
            max_value=50,
            value=max(2, map_cols_default),
            step=1,
            key="ml_unsup_som_map_cols",
        )
    with p2:
        n_iter = st.number_input(
            "n_iter",
            min_value=100,
            max_value=100000,
            value=int(st.session_state.get("ml_unsup_som_n_iter", 3000)),
            step=100,
            key="ml_unsup_som_n_iter",
        )
        learning_rate = st.number_input(
            "learning_rate",
            min_value=0.001,
            max_value=1.0,
            value=float(st.session_state.get("ml_unsup_som_learning_rate", 0.5)),
            step=0.01,
            format="%.3f",
            key="ml_unsup_som_learning_rate",
        )
    with p3:
        sigma = st.number_input(
            "sigma",
            min_value=0.1,
            max_value=50.0,
            value=max(0.1, sigma_default),
            step=0.1,
            format="%.2f",
            key="ml_unsup_som_sigma",
        )
        random_state = st.number_input(
            "random_state (opcional, -1=desligado)",
            value=int(st.session_state.get("ml_unsup_som_random_state", 42)),
            step=1,
            key="ml_unsup_som_random_state",
        )

    st.caption(
        f"Amostras usadas no treino: {n_samples}. "
        "SOM cria um codebook 2D; depois cada sujeito e associado ao no BMU."
    )

    run_model = st.button(
        "Rodar SOM",
        use_container_width=True,
        key="ml_unsup_som_run_btn",
    )
    if run_model:
        import numpy as np

        _clear_som_cache()

        weights_flat, _ = _train_som(
            X_model=X_model,
            map_rows=int(map_rows),
            map_cols=int(map_cols),
            sigma=float(sigma),
            learning_rate=float(learning_rate),
            n_iter=int(n_iter),
            random_state=int(random_state),
        )

        bmu_node, bmu_row, bmu_col, min_dist = _assign_bmu(
            X_model=X_model,
            weights_flat=weights_flat,
            map_cols=int(map_cols),
        )

        n_nodes = int(map_rows) * int(map_cols)
        node_counts = pd.Series(bmu_node).value_counts().sort_index()
        node_counts = node_counts.reindex(range(n_nodes), fill_value=0)
        qe = float(np.mean(min_dist))
        topographic_error = _compute_topographic_error(
            X_model=X_model,
            weights_flat=weights_flat,
            map_rows=int(map_rows),
            map_cols=int(map_cols),
        )
        empty_nodes = int((node_counts.values == 0).sum())
        u_matrix = _compute_u_matrix(weights_flat, map_rows=int(map_rows), map_cols=int(map_cols))
        weights_original = _to_original_space(weights_flat, scaler)

        _save_som_to_state(
            labels_node=bmu_node,
            labels_row=bmu_row,
            labels_col=bmu_col,
            quantization_dist=min_dist,
            used_index=X_df.index,
            cols=cols,
            map_rows=map_rows,
            map_cols=map_cols,
            weights_flat=weights_flat,
            weights_original=weights_original,
            u_matrix=u_matrix,
            node_counts=node_counts.to_dict(),
            qe=qe,
            topographic_error=topographic_error,
            empty_nodes=empty_nodes,
            scaler_used=(scaler is not None),
        )
        st.success("SOM executado e armazenado.")

    _render_som_cache(df)
