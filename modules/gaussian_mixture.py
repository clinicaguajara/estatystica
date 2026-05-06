import pandas as pd
import streamlit as st

from modules.unsupervised_common import get_numeric_columns, prepare_feature_matrix


GMM_FINAL_KEY = "ml_unsup_gmm_final"
GMM_EXPLORE_KEY = "ml_unsup_gmm_explore"


def _clear_gmm_cache():
    st.session_state.pop(GMM_FINAL_KEY, None)


def _build_gmm_model(
    *,
    n_components,
    covariance_type,
    tol,
    reg_covar,
    max_iter,
    n_init,
    init_params,
    random_state,
):
    from sklearn.mixture import GaussianMixture

    seed = None if int(random_state) < 0 else int(random_state)
    return GaussianMixture(
        n_components=int(n_components),
        covariance_type=str(covariance_type),
        tol=float(tol),
        reg_covar=float(reg_covar),
        max_iter=int(max_iter),
        n_init=int(n_init),
        init_params=str(init_params),
        random_state=seed,
    )


def _save_gmm_to_state(
    *,
    labels,
    confidence,
    probabilities,
    used_index,
    cols,
    means_scaled,
    means_original,
    weights,
    scaler_used,
    projection_2d,
    projection_label,
    X_model,
    cluster_counts,
    metrics,
    converged,
    n_iter,
):
    st.session_state[GMM_FINAL_KEY] = {
        "labels": list(labels),
        "confidence": list(confidence),
        "probabilities": probabilities.tolist(),
        "used_index": list(used_index),
        "cols": list(cols),
        "means_scaled": means_scaled.tolist(),
        "means_original": means_original.tolist(),
        "weights": list(weights),
        "scaler_used": bool(scaler_used),
        "projection_2d": projection_2d.tolist(),
        "projection_label": str(projection_label),
        "X_model": X_model.tolist(),
        "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
        "metrics": metrics,
        "converged": bool(converged),
        "n_iter": int(n_iter),
    }


def _build_projection_2d(X_model, feature_names):
    import numpy as np

    X_arr = np.asarray(X_model, dtype=float)
    if X_arr.shape[1] <= 2:
        if X_arr.shape[1] == 1:
            x = X_arr[:, 0]
            y = np.zeros_like(x)
            return np.column_stack([x, y]), f"{feature_names[0]} (eixo unico)"
        return X_arr[:, :2], f"{feature_names[0]} x {feature_names[1]} (espaco do modelo)"

    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=0)
        proj = pca.fit_transform(X_arr)
        explained = float(pca.explained_variance_ratio_.sum() * 100.0)
        label = f"PCA 2D no espaco do modelo (var. explicada: {explained:.1f}%)"
        return proj, label
    except Exception:
        return X_arr[:, :2], f"{feature_names[0]} x {feature_names[1]} (espaco do modelo)"


def _probability_columns(n_components: int) -> list[str]:
    return [f"prob_cluster_{i}" for i in range(int(n_components))]


def _compute_probability_diagnostics(probabilities):
    import numpy as np

    proba = np.asarray(probabilities, dtype=float)
    max_prob = proba.max(axis=1)

    if proba.shape[1] >= 2:
        sorted_probs = np.sort(proba, axis=1)
        second_prob = sorted_probs[:, -2]
        margin_top2 = max_prob - second_prob
    else:
        second_prob = np.full(proba.shape[0], np.nan, dtype=float)
        margin_top2 = np.full(proba.shape[0], np.nan, dtype=float)

    eps = float(np.finfo(float).eps)
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    if proba.shape[1] > 1:
        entropy_norm = entropy / float(np.log(proba.shape[1]))
    else:
        entropy_norm = np.zeros(proba.shape[0], dtype=float)

    return max_prob, second_prob, margin_top2, entropy_norm


def _render_gmm_cache(df: pd.DataFrame):
    import numpy as np
    import plotly.express as px
    from sklearn.decomposition import PCA

    cache = st.session_state.get(GMM_FINAL_KEY)
    if not cache:
        return False

    metrics = cache["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("AIC (menor melhor)", f"{metrics['AIC']:,.2f}")
    with c2:
        st.metric("BIC (menor melhor)", f"{metrics['BIC']:,.2f}")
    with c3:
        st.metric("Silhouette", "n/a" if metrics["silhouette"] is None else f"{metrics['silhouette']:.4f}")
    with c4:
        st.metric("Convergiu", "sim" if cache["converged"] else "nao")

    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("Calinski-Harabasz", "n/a" if metrics["CH"] is None else f"{metrics['CH']:,.1f}")
    with c6:
        st.metric("Davies-Bouldin", "n/a" if metrics["DB"] is None else f"{metrics['DB']:.3f}")
    with c7:
        st.metric("Iteracoes (EM)", f"{cache['n_iter']}")

    st.caption(f"Lower bound (log-likelihood medio): {metrics['lower_bound']:.6f}")

    counts = pd.Series(cache["cluster_counts"]).sort_index()
    counts.index = [f"cluster_{i}" for i in counts.index]
    st.write("#### Tamanho dos clusters")
    st.dataframe(counts.rename("n").to_frame(), use_container_width=True)

    weights_df = pd.DataFrame(
        {
            "cluster": [f"cluster_{i}" for i in range(len(cache["weights"]))],
            "peso": cache["weights"],
        }
    )
    st.write("#### Pesos dos componentes")
    st.dataframe(weights_df, use_container_width=True)

    means_original = np.array(cache["means_original"])
    means_df = pd.DataFrame(means_original, columns=cache["cols"])
    means_df.index = [f"cluster_{i}" for i in range(means_df.shape[0])]
    st.write("#### Medias por componente (espaco original, quando possivel)")
    st.dataframe(means_df, use_container_width=True)

    labels = np.array(cache["labels"])
    probabilities = np.array(cache.get("probabilities", []), dtype=float)
    if probabilities.size == 0:
        confidence = np.array(cache["confidence"], dtype=float)
        probabilities = confidence.reshape(-1, 1)
    else:
        confidence = np.array(cache["confidence"], dtype=float)
    used_index = pd.Index(cache["used_index"])

    max_prob, second_prob, margin_top2, entropy_norm = _compute_probability_diagnostics(probabilities)
    proba_cols = _probability_columns(probabilities.shape[1])

    labeled = df.loc[used_index].copy()
    labeled["cluster_gmm"] = labels
    labeled["prob_cluster_max"] = max_prob
    labeled["prob_segundo_cluster"] = second_prob
    labeled["margem_top2"] = margin_top2
    labeled["entropia_normalizada"] = entropy_norm
    for i, col_name in enumerate(proba_cols):
        labeled[col_name] = probabilities[:, i]

    st.write("#### Visualizacao 2D dos sujeitos")
    projection = np.array(cache.get("projection_2d", []), dtype=float)
    if projection.shape[0] == len(labels) and projection.shape[1] == 2:
        scatter_df = pd.DataFrame(
            {
                "x": projection[:, 0],
                "y": projection[:, 1],
                "cluster": [f"cluster_{int(c)}" for c in labels.tolist()],
                "prob_cluster_max": max_prob,
                "sujeito": [str(idx) for idx in used_index.tolist()],
            }
        )
        fig = px.scatter(
            scatter_df,
            x="x",
            y="y",
            color="cluster",
            size="prob_cluster_max",
            hover_data={
                "sujeito": True,
                "prob_cluster_max": ":.4f",
                "x": ":.4f",
                "y": ":.4f",
            },
            title="Distribuicao dos sujeitos no espaco do modelo",
        )
        fig.update_traces(marker={"opacity": 0.8})
        fig.update_layout(
            xaxis_title=cache.get("projection_label", "Componente 1"),
            yaxis_title="Componente 2",
            legend_title="Cluster",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bolhas maiores indicam maior confianca no cluster dominante.")
    else:
        st.info("Nao foi possivel gerar a projecao 2D para visualizacao.")

    st.write("#### Visualizacao 3D dos sujeitos")
    X_subjects = np.array(cache.get("X_model", []), dtype=float)
    if X_subjects is not None and X_subjects.ndim == 2:
        max_3d_components = min(X_subjects.shape[0], X_subjects.shape[1])
        if max_3d_components >= 3:
            try:
                proj3 = PCA(n_components=3, random_state=42).fit_transform(X_subjects)
                scatter3d_df = pd.DataFrame(
                    {
                        "PC1": proj3[:, 0],
                        "PC2": proj3[:, 1],
                        "PC3": proj3[:, 2],
                        "cluster": [f"cluster_{int(c)}" for c in labels.tolist()],
                        "prob_cluster_max": max_prob,
                        "sujeito": [str(idx) for idx in used_index.tolist()],
                    }
                )
                fig3d = px.scatter_3d(
                    scatter3d_df,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color="cluster",
                    size="prob_cluster_max",
                    hover_data={
                        "sujeito": True,
                        "prob_cluster_max": ":.4f",
                        "PC1": ":.4f",
                        "PC2": ":.4f",
                        "PC3": ":.4f",
                    },
                    title="Distribuicao 3D dos sujeitos no espaco do modelo (PCA)",
                )
                fig3d.update_traces(marker={"opacity": 0.8})
                fig3d.update_layout(legend_title="Cluster")
                st.plotly_chart(fig3d, use_container_width=True)
            except Exception:
                st.info("Nao foi possivel gerar a projecao 3D para visualizacao.")
        else:
            st.caption("Projecao 3D indisponivel: sao necessarios pelo menos 3 componentes (amostras/features).")
    else:
        st.info("Nao foi possivel gerar a projecao 3D para visualizacao.")

    st.write("#### Probabilidades por sujeito")
    st.caption("Cada sujeito recebe probabilidades para todos os clusters (soma = 1).")

    n_rows = int(len(labeled))
    if n_rows > 0:
        row_pos = st.number_input(
            "Linha para inspecionar (posicao no dataset filtrado):",
            min_value=0,
            max_value=max(0, n_rows - 1),
            value=0,
            step=1,
            key="ml_unsup_gmm_subject_position",
        )
        row_pos = int(row_pos)
        sujeito_id = used_index[row_pos]
        sujeito_probs = pd.DataFrame(
            {"cluster": proba_cols, "probabilidade": probabilities[row_pos].tolist()}
        )
        prob_fig = px.bar(
            sujeito_probs,
            x="cluster",
            y="probabilidade",
            title=f"Pertencimento do sujeito {sujeito_id}",
            range_y=[0.0, 1.0],
        )
        st.plotly_chart(prob_fig, use_container_width=True)

    uncertainty_df = labeled[
        ["cluster_gmm", "prob_cluster_max", "prob_segundo_cluster", "margem_top2", "entropia_normalizada"]
    ].sort_values(by=["prob_cluster_max", "margem_top2"], ascending=[True, True])
    st.write("#### Casos mais ambiguos (top 20)")
    st.dataframe(uncertainty_df.head(20), use_container_width=True)

    st.write("#### Amostra rotulada")
    st.dataframe(labeled.head(20), use_container_width=True)

    csv_bytes = labeled.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download (CSV Gaussian Mixture)",
        data=csv_bytes,
        file_name="gaussian_mixture_clusters.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_unsup_gmm_download_csv",
    )
    return True


def render_gaussian_mixture(df: pd.DataFrame):
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Selecione um DataFrame com pelo menos 2 variaveis numericas.")
        return

    st.markdown("#### Configuracoes do Gaussian Mixture")

    cols = st.multiselect(
        "Variaveis numericas (features) para clusterizacao:",
        numeric_cols,
        key="ml_unsup_gmm_cols",
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
            key="ml_unsup_gmm_missing",
        )
    with c2:
        scaler_choice = st.selectbox(
            "Escalonamento:",
            ["Nenhum", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            key="ml_unsup_gmm_scaler",
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
    max_components = int(min(20, max(1, n_samples - 1)))

    n_components = int(st.session_state.get("ml_unsup_gmm_n_components", 3))
    covariance_type = st.session_state.get("ml_unsup_gmm_covariance_type", "full")
    init_params = st.session_state.get("ml_unsup_gmm_init_params", "kmeans")
    n_init = int(st.session_state.get("ml_unsup_gmm_n_init", 5))
    max_iter = int(st.session_state.get("ml_unsup_gmm_max_iter", 300))
    tol = float(st.session_state.get("ml_unsup_gmm_tol", 1e-3))
    reg_covar = float(st.session_state.get("ml_unsup_gmm_reg_covar", 1e-6))
    random_state = int(st.session_state.get("ml_unsup_gmm_random_state", 42))

    p1, p2, p3 = st.columns(3)
    with p1:
        n_components = st.number_input(
            "n_components (modelo final)",
            min_value=1,
            max_value=max_components,
            value=min(max(1, n_components), max_components),
            step=1,
            key="ml_unsup_gmm_n_components",
        )
        covariance_type = st.selectbox(
            "covariance_type",
            ["full", "tied", "diag", "spherical"],
            index=["full", "tied", "diag", "spherical"].index(covariance_type if covariance_type in ["full", "tied", "diag", "spherical"] else "full"),
            key="ml_unsup_gmm_covariance_type",
        )
        init_params = st.selectbox(
            "init_params",
            ["kmeans", "random"],
            index=(0 if init_params == "kmeans" else 1),
            key="ml_unsup_gmm_init_params",
        )
    with p2:
        n_init = st.number_input(
            "n_init",
            min_value=1,
            value=int(n_init),
            step=1,
            key="ml_unsup_gmm_n_init",
        )
        max_iter = st.number_input(
            "max_iter",
            min_value=10,
            value=int(max_iter),
            step=10,
            key="ml_unsup_gmm_max_iter",
        )
        random_state = st.number_input(
            "random_state (opcional, -1=desligado)",
            value=int(random_state),
            step=1,
            key="ml_unsup_gmm_random_state",
        )
    with p3:
        tol = st.number_input(
            "tol",
            min_value=1e-8,
            value=float(tol),
            format="%.1e",
            key="ml_unsup_gmm_tol",
        )
        reg_covar = st.number_input(
            "reg_covar",
            min_value=0.0,
            value=float(reg_covar),
            format="%.1e",
            key="ml_unsup_gmm_reg_covar",
        )

    upper_default = min(10, max_components)
    range_default = (1, upper_default if upper_default >= 1 else 1)
    cmin, cmax = st.slider(
        "Intervalo para explorar componentes (AIC/BIC):",
        min_value=1,
        max_value=max_components,
        value=range_default,
        key="ml_unsup_gmm_component_range",
    )

    run_explore = st.button(
        "Explorar Componentes (AIC/BIC)",
        use_container_width=True,
        key="ml_unsup_gmm_explore_btn",
    )
    if run_explore:
        import numpy as np

        ks = list(range(int(cmin), int(cmax) + 1))
        aics = []
        bics = []
        silhouettes = []

        for k in ks:
            try:
                model = _build_gmm_model(
                    n_components=k,
                    covariance_type=st.session_state["ml_unsup_gmm_covariance_type"],
                    tol=st.session_state["ml_unsup_gmm_tol"],
                    reg_covar=st.session_state["ml_unsup_gmm_reg_covar"],
                    max_iter=st.session_state["ml_unsup_gmm_max_iter"],
                    n_init=st.session_state["ml_unsup_gmm_n_init"],
                    init_params=st.session_state["ml_unsup_gmm_init_params"],
                    random_state=st.session_state["ml_unsup_gmm_random_state"],
                )
                model.fit(X_model)
                labels = model.predict(X_model)

                aics.append(float(model.aic(X_model)))
                bics.append(float(model.bic(X_model)))

                unique_labels = set(labels.tolist())
                can_score = len(unique_labels) >= 2 and X_model.shape[0] > len(unique_labels)
                if can_score:
                    silhouettes.append(float(silhouette_score(X_model, labels)))
                else:
                    silhouettes.append(float("nan"))
            except Exception:
                aics.append(float("nan"))
                bics.append(float("nan"))
                silhouettes.append(float("nan"))

        aic_arr = np.array(aics, dtype=float)
        bic_arr = np.array(bics, dtype=float)
        best_aic_k = None if np.all(np.isnan(aic_arr)) else int(ks[int(np.nanargmin(aic_arr))])
        best_bic_k = None if np.all(np.isnan(bic_arr)) else int(ks[int(np.nanargmin(bic_arr))])

        st.session_state[GMM_EXPLORE_KEY] = {
            "k": ks,
            "aic": aics,
            "bic": bics,
            "silhouette": silhouettes,
            "best_aic_k": best_aic_k,
            "best_bic_k": best_bic_k,
        }

    explore_cache = st.session_state.get(GMM_EXPLORE_KEY)
    if explore_cache:
        import plotly.express as px

        res = pd.DataFrame(
            {
                "n_components": explore_cache["k"],
                "AIC": explore_cache["aic"],
                "BIC": explore_cache["bic"],
                "Silhouette": explore_cache["silhouette"],
            }
        )
        st.write("#### Resultados da exploracao")
        fig_explore = px.line(
            res,
            x="n_components",
            y=["AIC", "BIC"],
            markers=True,
            title="Curvas AIC e BIC por numero de componentes",
        )
        fig_explore.update_layout(legend_title="Metrica", xaxis_title="n_components", yaxis_title="Valor")
        st.plotly_chart(fig_explore, use_container_width=True)
        st.dataframe(res, use_container_width=True)
        st.caption(
            f"Melhor AIC: {explore_cache['best_aic_k'] if explore_cache['best_aic_k'] is not None else 'n/a'} | "
            f"Melhor BIC: {explore_cache['best_bic_k'] if explore_cache['best_bic_k'] is not None else 'n/a'}"
        )

    run_model = st.button(
        "Rodar Gaussian Mixture",
        use_container_width=True,
        key="ml_unsup_gmm_run_btn",
    )
    if run_model:
        import numpy as np

        _clear_gmm_cache()

        try:
            model = _build_gmm_model(
                n_components=st.session_state["ml_unsup_gmm_n_components"],
                covariance_type=st.session_state["ml_unsup_gmm_covariance_type"],
                tol=st.session_state["ml_unsup_gmm_tol"],
                reg_covar=st.session_state["ml_unsup_gmm_reg_covar"],
                max_iter=st.session_state["ml_unsup_gmm_max_iter"],
                n_init=st.session_state["ml_unsup_gmm_n_init"],
                init_params=st.session_state["ml_unsup_gmm_init_params"],
                random_state=st.session_state["ml_unsup_gmm_random_state"],
            )
            model.fit(X_model)
            labels = model.predict(X_model)
            probabilities = model.predict_proba(X_model)
            confidence = probabilities.max(axis=1)
        except Exception as e:
            st.error(f"Nao foi possivel ajustar o Gaussian Mixture com os parametros atuais. Detalhe tecnico: {e}")
            return

        unique_labels = set(labels.tolist())
        can_score = len(unique_labels) >= 2 and X_model.shape[0] > len(unique_labels)
        if can_score:
            silhouette = float(silhouette_score(X_model, labels))
            ch_score = float(calinski_harabasz_score(X_model, labels))
            db_score = float(davies_bouldin_score(X_model, labels))
        else:
            silhouette = float("nan")
            ch_score = float("nan")
            db_score = float("nan")

        means_scaled = model.means_
        if scaler is not None:
            try:
                means_original = scaler.inverse_transform(means_scaled)
            except Exception:
                means_original = means_scaled
        else:
            means_original = means_scaled

        counts = pd.Series(labels).value_counts().sort_index()
        metrics = {
            "AIC": float(model.aic(X_model)),
            "BIC": float(model.bic(X_model)),
            "lower_bound": float(model.lower_bound_),
            "silhouette": (None if pd.isna(silhouette) else float(silhouette)),
            "CH": (None if pd.isna(ch_score) else float(ch_score)),
            "DB": (None if pd.isna(db_score) else float(db_score)),
        }

        projection_2d, projection_label = _build_projection_2d(X_model=X_model, feature_names=cols)

        _save_gmm_to_state(
            labels=labels,
            confidence=confidence,
            probabilities=probabilities,
            used_index=X_df.index,
            cols=cols,
            means_scaled=means_scaled,
            means_original=means_original,
            weights=model.weights_,
            scaler_used=(scaler is not None),
            projection_2d=projection_2d,
            projection_label=projection_label,
            X_model=X_model,
            cluster_counts=counts.to_dict(),
            metrics=metrics,
            converged=model.converged_,
            n_iter=model.n_iter_,
        )
        st.success("Gaussian Mixture executado e armazenado.")

    _render_gmm_cache(df)
