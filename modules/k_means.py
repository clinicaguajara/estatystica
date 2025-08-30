# modules/ml_unsupervised.py
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd

# ── Funções utilitárias (gráficos e métricas) ────────────────────────────────
def _build_elbow_fig(ks, inertias, elbow_k=None, theme="dark"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Paleta Estatystica
    dark_bg, white, purple = "#0E1117", "#FFFFFF", "#7159c1"

    # Cores por tema
    if theme == "dark":
        bg, fg, grid = dark_bg, white, "#22262e"
    else:
        bg, fg, grid = "#FFFFFF", "#0E1117", "#e5e7eb"  # claro com linha roxa

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(ks, inertias, marker="o", linewidth=2.0, color=purple)
    ax.set_xlabel("k", color=fg)
    ax.set_ylabel("Inertia (SSE)", color=fg)
    ax.set_title("Elbow (Inertia)", color=fg, pad=10)

    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    ax.grid(True, color=grid, alpha=0.4)

    if elbow_k is not None:
        try:
            yk = np.interp(elbow_k, ks, inertias)
            ax.axvline(elbow_k, linestyle="--", color=fg, alpha=0.7)
            ax.text(elbow_k, yk, f"  elbow≈{elbow_k}", va="bottom", color=fg)
        except Exception:
            pass

    fig.tight_layout()
    return fig


def _build_silhouette_fig(ks, silhouettes, silhouette_k=None, theme="dark"):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Paleta Estatystica
    dark_bg, white, purple = "#0E1117", "#FFFFFF", "#7159c1"

    # Cores por tema
    if theme == "dark":
        bg, fg, grid = dark_bg, white, "#22262e"
    else:
        bg, fg, grid = "#FFFFFF", "#0E1117", "#e5e7eb"

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(ks, silhouettes, marker="o", linewidth=2.0, color=purple)
    ax.set_xlabel("k", color=fg)
    ax.set_ylabel("Silhouette", color=fg)
    ax.set_title("Silhouette vs k", color=fg, pad=10)

    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    ax.grid(True, color=grid, alpha=0.4)

    if silhouette_k is not None:
        try:
            yk = float(pd.Series(silhouettes, index=ks).loc[silhouette_k])
            ax.axvline(silhouette_k, linestyle="--", color=fg, alpha=0.7)
            ax.text(silhouette_k, yk, f"  best≈{silhouette_k}", va="bottom", color=fg)
        except Exception:
            pass

    fig.tight_layout()
    return fig


def _render_downloads(fig_dark, fig_light, base_name: str):
    import io
    import streamlit as st

    # Salva em buffers PNG (respeitando facecolor configurada do fig)
    buf_dark, buf_light = io.BytesIO(), io.BytesIO()
    fig_dark.savefig(buf_dark, format="png", dpi=200,
                     facecolor=fig_dark.get_facecolor(), edgecolor="none")
    fig_light.savefig(buf_light, format="png", dpi=200,
                      facecolor=fig_light.get_facecolor(), edgecolor="none")
    buf_dark.seek(0); buf_light.seek(0)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"📥 Download (tema escuro)",
            data=buf_dark.getvalue(),
            file_name=f"{base_name.lower().replace(' ', '_')}_dark.png",
            mime="image/png",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            f"📥 Download (tema claro)",
            data=buf_light.getvalue(),
            file_name=f"{base_name.lower().replace(' ', '_')}_light.png",
            mime="image/png",
            use_container_width=True,
        )


def plot_elbow(ks, inertias, elbow_k=None):
    """
    <docstrings>
    Desenha o Elbow no tema escuro do Estatystica e oferece downloads
    em tema escuro e claro (linha roxa nos dois).
    """
    import matplotlib.pyplot as plt
    import streamlit as st

    # constrói figs
    fig_dark = _build_elbow_fig(ks, inertias, elbow_k=elbow_k, theme="dark")
    fig_light = _build_elbow_fig(ks, inertias, elbow_k=elbow_k, theme="light")

    # renderiza no app (escuro) e oferece downloads
    st.pyplot(fig_dark, clear_figure=True)
    _render_downloads(fig_dark, fig_light, base_name="Elbow (Inertia)")

    # boa prática: fechar figuras
    plt.close(fig_dark); plt.close(fig_light)


def plot_silhouette(ks, silhouettes, silhouette_k=None):
    """
    <docstrings>
    Desenha o Silhouette no tema escuro do Estatystica e oferece downloads
    em tema escuro e claro (linha roxa nos dois).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    if not any(pd.notna(silhouettes)):
        st.caption("Silhouette não calculado para este intervalo/dados.")
        return

    fig_dark = _build_silhouette_fig(ks, silhouettes, silhouette_k=silhouette_k, theme="dark")
    fig_light = _build_silhouette_fig(ks, silhouettes, silhouette_k=silhouette_k, theme="light")

    st.pyplot(fig_dark, clear_figure=True)
    _render_downloads(fig_dark, fig_light, base_name="Silhouette vs k")

    plt.close(fig_dark); plt.close(fig_light)


def elbow_by_max_distance(ks, inertias):
    """
    <docstrings>
    Heurística do cotovelo: maior distância do ponto à reta entre os extremos.
    Args:
        ks (list[int])
        inertias (list[float])
    Returns:
        int: k sugerido
    """
    import numpy as np

    x1, y1 = ks[0], inertias[0]
    x2, y2 = ks[-1], inertias[-1]
    dx, dy = (x2 - x1), (y2 - y1)
    denom = float((dy**2 + dx**2) ** 0.5) if (dx != 0 or dy != 0) else 1.0

    dists = []
    for k, sse in zip(ks, inertias):
        num = abs(dy * k - dx * sse + (x2 * y1 - y2 * x1))
        dists.append(num / denom)

    import numpy as np
    return int(ks[int(np.argmax(dists))])


def evaluate_stability_ari(
    X_scaled,
    k: int,
    init_method: str,
    n_init: int,
    max_iter: int,
    tol: float,
    algorithm: str,
    base_seed: int,
    n_runs: int = 10,
):
    """
    <docstrings>
    Mede a estabilidade da partição para um K fixo via ARI médio entre múltiplas execuções.
    Estratégia:
        • roda K-Means n_runs vezes, variando random_state = base_seed + i
        • calcula ARI para todos os pares de rotulações
        • retorna média e desvio-padrão dos ARIs
    Returns:
        (ari_mean: float, ari_std: float)
    Calls:
        sklearn.cluster.KMeans, sklearn.metrics.adjusted_rand_score
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    from itertools import combinations

    labels_list = []
    for i in range(int(n_runs)):
        seed = None if base_seed < 0 else int(base_seed + i)
        km = KMeans(
            n_clusters=int(k),
            init=init_method,
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            algorithm=algorithm,
            random_state=seed,
        ).fit(X_scaled)
        labels_list.append(km.labels_)

    if len(labels_list) < 2:
        return float("nan"), float("nan")

    aris = []
    for i, j in combinations(range(len(labels_list)), 2):
        aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    import numpy as np
    return float(np.mean(aris)), float(np.std(aris))


def internal_indices_ch_db(X_scaled, labels):
    """
    <docstrings>
    Calcula índices internos CH (↑ melhor) e DB (↓ melhor) para uma solução final.
    Returns:
        (CH: float, DB: float)
    Calls:
        sklearn.metrics.calinski_harabasz_score, davies_bouldin_score
    """
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    CH = calinski_harabasz_score(X_scaled, labels)
    DB = davies_bouldin_score(X_scaled, labels)
    return float(CH), float(DB)


# ── GRAFICO: Perfis por cluster (linhas) ─────────────────────────────────────
def _build_cluster_profile_fig(
    centers, feature_names, theme="dark",
    cluster_labels=None, y_label=None,
    xrotation=60, xfontsize=9, wrap_width=14
):
    import matplotlib.pyplot as plt
    import numpy as np
    import textwrap

    # Paleta/tema
    dark_bg, white, purple = "#0E1117", "#FFFFFF", "#7159c1"
    if theme == "dark":
        bg, fg, grid = dark_bg, white, "#22262e"
    else:
        bg, fg, grid = "#FFFFFF", "#0E1117", "#e5e7eb"

    n_clusters, n_features = centers.shape
    x = np.arange(n_features)

    # Quebra opcional dos labels (None/0 desliga)
    if wrap_width and wrap_width > 0:
        xticklabels = [textwrap.fill(str(n), width=int(wrap_width)) for n in feature_names]
    else:
        xticklabels = list(map(str, feature_names))

    colors = ["#e74c3c","#2ecc71","#3498db","#f1c40f",
              "#9b59b6","#e67e22","#1abc9c","#95a5a6",
              "#34495e","#7f8c8d"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for i in range(n_clusters):
        color = colors[i % len(colors)]
        label = cluster_labels[i] if cluster_labels else f"cluster_{i}"
        ax.plot(x, centers[i, :], marker="o", linewidth=2.2, color=color, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=int(xrotation), ha="right",
                       color=fg, fontsize=int(xfontsize))
    ax.set_ylabel(y_label or "Valores", color=fg)
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.grid(True, color=grid, alpha=0.35)
    leg = ax.legend(frameon=True)
    leg.get_frame().set_facecolor(bg)
    leg.get_frame().set_edgecolor(fg)
    for tx in leg.get_texts():
        tx.set_color(fg)

    # dá mais espaço para rótulos inclinados
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    return fig


def plot_cluster_profiles(centers_scaled, centers_original, feature_names,
                          default_show_scaled: bool, cluster_labels=None):
    import streamlit as st
    import matplotlib.pyplot as plt

    st.markdown("**Exibição dos perfis**")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        show_scaled = st.checkbox(
            "Usar espaço do modelo", value=default_show_scaled,
            help="Se ligado, plota os centróides no espaço escalonado."
        )
    with colB:
        xrotation = st.slider("Rotação dos labels (°)", 0, 90, 60, step=5)
    with colC:
        wrap_width = st.number_input("Quebra a cada N chars (0=off)", 0, 40, 14, step=1)
    with colD:
        xfontsize = st.number_input("Tamanho da fonte", 6, 16, 9, step=1)

    centers = centers_scaled if show_scaled else centers_original
    y_label = "Z-scores" if show_scaled else "Valores (espaço original)"

    fig_dark = _build_cluster_profile_fig(
        centers=centers, feature_names=feature_names, theme="dark",
        cluster_labels=cluster_labels, y_label=y_label,
        xrotation=xrotation, xfontsize=xfontsize, wrap_width=wrap_width
    )
    fig_light = _build_cluster_profile_fig(
        centers=centers, feature_names=feature_names, theme="light",
        cluster_labels=cluster_labels, y_label=y_label,
        xrotation=xrotation, xfontsize=xfontsize, wrap_width=wrap_width
    )

    st.pyplot(fig_dark, clear_figure=True)
    _render_downloads(fig_dark, fig_light, base_name="Cluster profiles")

    plt.close(fig_dark); plt.close(fig_light)


# ── UI principal (sem form/expander), reuso dos hiperparâmetros ─────────────
def _get_param(key, default):
    return st.session_state.get(key, default)

# ====== CACHE DO RESULTADO FINAL ======
FINAL_KEY = "ml_unsup_final"

def clear_final_cache():
    st.session_state.pop(FINAL_KEY, None)

def _final_signature(cols, missing_strategy, scaler_choice, *, k, init, n_init, max_iter, tol, algorithm, random_state):
    # Usada só para dizer se o cache está desatualizado (stale)
    return (tuple(cols), str(missing_strategy), str(scaler_choice),
            int(k), str(init), int(n_init), int(max_iter), float(tol),
            str(algorithm), int(random_state))

def _save_final_to_state(*, sig, labels, used_index, centers_scaled, centers_original,
                         feature_names, inertia, sil, CH, DB, scaler_used, counts):
    st.session_state[FINAL_KEY] = {
        "sig": sig,
        "labels": list(labels),                 # não converte para int; preserva
        "used_index": list(used_index),         # idem (pode ser string/Datetime)
        "centers_scaled": centers_scaled.tolist(),
        "centers_original": centers_original.tolist(),
        "feature_names": list(feature_names),
        "metrics": {
            "inertia": float(inertia),
            "silhouette": (None if pd.isna(sil) else float(sil)),
            "CH": float(CH), "DB": float(DB),
        },
        "scaler_used": bool(scaler_used),
        "cluster_counts": {int(k): int(v) for k, v in counts.items()},
    }

def _render_final_from_cache(df):
    """Renderiza SEM refazer o modelo; não cria duplicatas."""
    import numpy as np

    cache = st.session_state.get(FINAL_KEY)
    if not cache:
        return False

    # Métricas
    m = cache["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Inertia (SSE)", f"{m['inertia']:,.2f}")
    with c2: st.metric("Silhouette", "n/a" if m['silhouette'] is None else f"{m['silhouette']:.4f}")
    with c3: st.metric("Calinski–Harabasz (↑)", f"{m['CH']:,.1f}")
    with c4: st.metric("Davies–Bouldin (↓)", f"{m['DB']:.3f}")

    # Tamanhos
    counts = pd.Series(cache["cluster_counts"]).sort_index()
    counts.index = [f"cluster_{i}" for i in counts.index]
    st.write("#### Tamanho dos clusters")
    st.dataframe(counts.rename("n").to_frame(), use_container_width=True)

    # Centroides + Perfis
    centers_original = np.array(cache["centers_original"])
    centers_scaled   = np.array(cache["centers_scaled"])
    feature_names    = cache["feature_names"]

    centers_df = pd.DataFrame(centers_original, columns=feature_names)
    centers_df.index = [f"cluster_{i}" for i in range(centers_df.shape[0])]
    st.write("#### Centroides (no espaço original, quando possível)")
    st.dataframe(centers_df, use_container_width=True)

    st.markdown("#### Perfis por cluster (linhas × features)")
    # widgets de exibição: uma única instância → não precisamos de múltiplas keys
    plot_cluster_profiles(
        centers_scaled=centers_scaled,
        centers_original=centers_original,
        feature_names=feature_names,
        default_show_scaled=cache["scaler_used"],
        cluster_labels=[f"cluster_{i}" for i in range(centers_scaled.shape[0])],
    )

    # CSV
    labels = np.array(cache["labels"])
    used_index = pd.Index(cache["used_index"])
    labeled = df.loc[used_index].assign(cluster=labels)
    csv_bytes = labeled.to_csv(index=True).encode("utf-8")
    st.download_button("📥 Download (CSV clusterizado)",
                       data=csv_bytes, file_name="kmeans_clusters.csv",
                       mime="text/csv", use_container_width=True)
    return True


def render_unsupervised(df: pd.DataFrame):
    """
    <docstrings>
    K-Means versátil e leve, sem st.form e com gráficos/métricas modularizados:
      • Seleção de features numéricas, missing (drop/média), escalonamento (Nenhum/Standard/MinMax/Robust)
      • Exploração de K (Elbow + Silhouette) usando os mesmos hiperparâmetros do modelo final
      • Heurística do cotovelo (maior distância à reta) → elbow_k
      • Estabilidade (ARI) para um K escolhido (n execuções reprodutíveis)
      • Modelo final com CH (↑) e DB (↓), centroides e download do CSV
    """

    # ── 0) Guardas e features numéricas ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Selecione um DataFrame com pelo menos **2** variáveis numéricas.")
        return
    st.markdown("#### Configurações do modelo")

    cols = st.multiselect(
        "Variáveis numéricas (features) para clusterização:",
        numeric_cols,
        help="Escolha as colunas que irão compor o espaço de clusterização."
    )
    if not cols or len(cols) < 2:
        st.info("Selecione ao menos **2** variáveis para continuar.")
        return

    col1, col2 = st.columns(2)
    with col1:
        missing_strategy = st.selectbox(
            "Missing values:",
            ["Excluir linhas com NA", "Imputar média"],
            help="Defina como tratar valores ausentes nas features selecionadas."
        )
    with col2:
        scaler_choice = st.selectbox(
            "Escalonamento:",
            ["Nenhum", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            help="K-Means usa distância Euclidiana; escalonar costuma ajudar."
        )
    
    sample_for_metrics = st.checkbox(
        "Amostrar para métricas (silhouette)",
        value=False,
        help="Para bases grandes, usa amostra ao calcular silhouette."
    )

    k_min, k_max = st.slider(
        "Intervalo para explorar soluções do valor de K:",
        min_value=1, max_value=20, value=(2, 10)
    )
    
    if sample_for_metrics:
        max_points_for_sil = st.number_input(
            "Tamanho máx. da amostra para silhouette:",
            min_value=100, max_value=10000, value=2000, step=100
        )
    else:
        max_points_for_sil = 2000

    # ── 1) Preparar X (missing + scaling) ────────────────────────────────────
    X = df[cols].copy()
    if missing_strategy == "Excluir linhas com NA":
        X = X.dropna(axis=0)
    else:
        X = X.fillna(X.mean(numeric_only=True))

    scaler = None
    if scaler_choice != "Nenhum":
        if scaler_choice == "StandardScaler":
            from sklearn.preprocessing import StandardScaler as _Scaler
        elif scaler_choice == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler as _Scaler
        else:
            from sklearn.preprocessing import RobustScaler as _Scaler
        scaler = _Scaler()
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = X.values

    n_samples = X_scaled.shape[0]
    if n_samples < 2:
        st.error("Dados insuficientes após tratamento de missing.")
        return

    # ── 2) Hiperparâmetros do modelo final (reusados também na exploração) ──

    init_method  = _get_param("ml_init_method", "k-means++")
    n_init       = _get_param("ml_n_init", 10)
    max_iter     = _get_param("ml_max_iter", 300)
    tol          = _get_param("ml_tol", 1e-4)
    algorithm    = _get_param("ml_algorithm", "lloyd")
    random_state = _get_param("ml_random_state", 42)
    n_clusters_final = _get_param("ml_n_clusters_final", 3)

    colA, colB, colC = st.columns(3)
    with colA:
        n_clusters_final = st.number_input("K (modelo final)", min_value=1, value=int(n_clusters_final), step=1, key="ml_n_clusters_final")
        init_method = st.selectbox("init", ["k-means++", "random"], index=(0 if init_method == "k-means++" else 1), key="ml_init_method")
    with colB:
        n_init = st.number_input("n_init", min_value=1, value=int(n_init), step=1, help="Reinicializações.", key="ml_n_init")
        max_iter = st.number_input("max_iter", min_value=10, value=int(max_iter), step=10, key="ml_max_iter")
    with colC:
        tol = st.number_input("tol", min_value=1e-10, value=float(tol), format="%.1e", key="ml_tol")
        algorithm = st.selectbox(
            "algorithm", ["lloyd", "elkan"],
            index=(0 if algorithm == "lloyd" else 1),
            help="‘elkan’ pode ser mais rápido; ‘lloyd’ espelha o método batch clássico.",
            key="ml_algorithm"
        )

    random_state = st.number_input("random_state (opcional, -1=desligado)", value=int(random_state), step=1, key="ml_random_state")

    # Controles extras do modelo final
    colF1, colF2 = st.columns(2)
    with colF1:
        show_centers = st.checkbox("Mostrar centroides", value=True)
    with colF2:
        show_preview = st.checkbox("Mostrar amostra rotulada", value=True)

    # ── 3) Explorar múltiplos K (usa hiperparâmetros acima) ─────────────────
    run_explore = st.button("Explorar K", use_container_width=True)
    state_key = "ml_unsup_explore"

    if run_explore:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import numpy as np

        ks = list(range(k_min, k_max + 1))
        inertias = []
        silhouettes = []

        # amostra reprodutível para silhouette
        rng = np.random.default_rng(None if st.session_state["ml_random_state"] < 0 else int(st.session_state["ml_random_state"]))
        if sample_for_metrics and n_samples > max_points_for_sil:
            idx = rng.choice(n_samples, size=int(max_points_for_sil), replace=False)
            X_for_sil = X_scaled[idx]
        else:
            X_for_sil = X_scaled

        for k in ks:
            km = KMeans(
                n_clusters=int(k),
                init=st.session_state["ml_init_method"],
                n_init=int(st.session_state["ml_n_init"]),
                max_iter=int(st.session_state["ml_max_iter"]),
                tol=float(st.session_state["ml_tol"]),
                algorithm=st.session_state["ml_algorithm"],
                random_state=None if st.session_state["ml_random_state"] < 0 else int(st.session_state["ml_random_state"]),
            ).fit(X_scaled)

            inertias.append(km.inertia_)

            if k >= 2 and X_for_sil.shape[0] > k:
                try:
                    s = silhouette_score(X_for_sil, km.predict(X_for_sil))
                except Exception:
                    s = float("nan")
            else:
                s = float("nan")
            silhouettes.append(s)

        elbow_k = elbow_by_max_distance(ks, inertias)

        import numpy as np
        silhouettes_arr = np.array(silhouettes, dtype=float)
        silhouette_k = None if np.all(np.isnan(silhouettes_arr)) else int(ks[int(np.nanargmax(silhouettes_arr))])

        st.session_state[state_key] = {
            "params": {
                "cols": tuple(cols),
                "missing_strategy": missing_strategy,
                "scaler_choice": scaler_choice,
                "sample_for_metrics": sample_for_metrics,
                "max_points_for_sil": int(max_points_for_sil),
                "k_min": int(k_min),
                "k_max": int(k_max),
                "init_method": st.session_state["ml_init_method"],
                "n_init": int(st.session_state["ml_n_init"]),
                "max_iter": int(st.session_state["ml_max_iter"]),
                "tol": float(st.session_state["ml_tol"]),
                "algorithm": st.session_state["ml_algorithm"],
                "random_state": int(st.session_state["ml_random_state"]),
            },
            "ks": ks,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "elbow_k": elbow_k,
            "silhouette_k": silhouette_k,
        }

    # Exibe resultados da exploração com funções de gráfico separadas
    if state_key in st.session_state:
        res_state = st.session_state[state_key]
        ks = res_state["ks"]
        inertias = res_state["inertias"]
        silhouettes = res_state["silhouettes"]
        elbow_k = res_state["elbow_k"]
        silhouette_k = res_state["silhouette_k"]

        res = pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": silhouettes})
        st.write("#### Resultados da exploração")

        plot_elbow(ks, inertias, elbow_k=elbow_k)
        plot_silhouette(ks, silhouettes, silhouette_k=silhouette_k)

        st.dataframe(res, use_container_width=True)

        if silhouette_k is not None:
            st.success(f"*Elbow* ↓ K = {elbow_k} | *Elbow* olha a inércia (SSE), que cai sempre ao aumentar K")
            st.success(f"*Silhouette* ↑ K = {silhouette_k} | *Silhouette* olha compacidade vs separação e pode preferir K maior quando há subestruturas claras.")
        else:
            st.success(f"*Elbow* ↓ K = {elbow_k} | *Elbow* olha a inércia (SSE), que cai sempre ao aumentar K")
            st.error("*Silhouette* indisponível para as variáveis (features) escolhidas.")


    # ── 4) Estabilidade (ARI) para um K escolhido ───────────────────────────
    st.markdown("#### Estabilidade (ARI)")
    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        k_for_stability = st.number_input("K para avaliar", min_value=2, value=int(n_clusters_final), step=1)
    with colS2:
        n_runs = st.number_input("n execuções (runs)", min_value=2, value=10, step=1, help="Mais execuções → estimativa de estabilidade mais robusta.")
    with colS3:
        base_seed = st.number_input("base_seed (usa base_seed+i)", value=int(random_state), step=1)

    run_stab = st.button("Calcular estabilidade (ARI)", use_container_width=True)
    if run_stab:
        ari_mean, ari_std = evaluate_stability_ari(
            X_scaled=X_scaled,
            k=int(k_for_stability),
            init_method=st.session_state["ml_init_method"],
            n_init=int(st.session_state["ml_n_init"]),
            max_iter=int(st.session_state["ml_max_iter"]),
            tol=float(st.session_state["ml_tol"]),
            algorithm=st.session_state["ml_algorithm"],
            base_seed=int(base_seed),
            n_runs=int(n_runs),
        )
        st.metric(f"ARI médio (K={int(k_for_stability)})", "n/a" if pd.isna(ari_mean) else f"{ari_mean:.3f}")
        st.caption(f"Desvio-padrão do ARI: {'n/a' if pd.isna(ari_std) else f'{ari_std:.3f}'} — valores próximos de 1 indicam partições muito estáveis.")

    # Assinatura dos parâmetros/dados atuais (serve para saber se o cache está fresco)
    current_sig = _final_signature(
        cols=cols, missing_strategy=missing_strategy, scaler_choice=scaler_choice,
        k=st.session_state["ml_n_clusters_final"], init=st.session_state["ml_init_method"],
        n_init=st.session_state["ml_n_init"], max_iter=st.session_state["ml_max_iter"],
        tol=st.session_state["ml_tol"], algorithm=st.session_state["ml_algorithm"],
        random_state=st.session_state["ml_random_state"],
    )


    # Se já existe resultado no cache, reexibir SEM reexecutar
    cached = st.session_state.get(FINAL_KEY)
     

    # ── 5) Rodar o modelo final ──────────────────────────────────────────────
    run_final = st.button("Rodar K-Means", use_container_width=True)
    if run_final:
        clear_final_cache()  # apaga o resultado antigo (limpa a tela no rerun)

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import numpy as np

        km_final = KMeans(
            n_clusters=int(st.session_state["ml_n_clusters_final"]),
            init=st.session_state["ml_init_method"],
            n_init=int(st.session_state["ml_n_init"]),
            max_iter=int(st.session_state["ml_max_iter"]),
            tol=float(st.session_state["ml_tol"]),
            algorithm=st.session_state["ml_algorithm"],
            random_state=None if st.session_state["ml_random_state"] < 0 else int(st.session_state["ml_random_state"]),
        )
        labels = km_final.fit_predict(X_scaled)

        inertia = km_final.inertia_
        sil_ok = (int(st.session_state["ml_n_clusters_final"]) >= 2) and (X_scaled.shape[0] > int(st.session_state["ml_n_clusters_final"]))
        sil = silhouette_score(X_scaled, labels) if sil_ok else float("nan")
        CH, DB = internal_indices_ch_db(X_scaled, labels)

        centers_scaled = km_final.cluster_centers_
        if scaler is not None:
            try:
                centers_original = scaler.inverse_transform(centers_scaled)
            except Exception:
                centers_original = centers_scaled
        else:
            centers_original = centers_scaled

        counts = pd.Series(labels).value_counts().sort_index()

        _save_final_to_state(
            sig=current_sig,
            labels=labels,
            used_index=X.index,
            centers_scaled=centers_scaled,
            centers_original=centers_original,
            feature_names=cols,
            inertia=inertia, sil=sil, CH=CH, DB=DB,
            scaler_used=(scaler is not None),
            counts=counts.to_dict(),
        )

        st.success("K-Means executado e armazenado. Você pode ajustar gráficos/baixar CSV sem perder a análise.")

    # Render único: se há cache → renderiza; se não há → nada a mostrar
    cached = st.session_state.get(FINAL_KEY)
    
    if cached:
        # Aviso se os parâmetros atuais diferem do que gerou o cache
        if tuple(cached["sig"]) != tuple(current_sig):
            st.warning("Mostrando o **resultado em cache** (parâmetros atuais diferem). "
                    "Clique em **Rodar K-Means** para atualizar.")
        _render_final_from_cache(df)