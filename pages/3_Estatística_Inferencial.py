# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from utils.design import load_css
import plotly.express as px
import plotly.graph_objects as go

# CUSTOM FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def sem_mediation_analysis(df: pd.DataFrame):
    """
    Executa análise de mediação via SEM (X → M → Y), com:
      • Estimativas simultâneas de trajetórias diretas, indiretas e totais
      • Índices de ajuste global exibidos em tabela
      • Gráficos de barras e alluvial (Sankey) em tema escuro + downloads em claro/escuro
    """
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import io
    import numpy as np
    import plotly.graph_objects as go
    from semopy import Model
    from semopy.stats import calc_stats

    # 1) Seleção de variáveis
    numeric = df.select_dtypes(include="number").columns.tolist()
    if len(numeric) < 3:
        st.warning("É necessário pelo menos 3 variáveis numéricas.")
        return
    X = st.selectbox("X (independente):", numeric, key="sem_x")
    M = st.selectbox("M (mediadora):", [c for c in numeric if c != X], key="sem_m")
    Y = st.selectbox("Y (dependente):", [c for c in numeric if c not in (X, M)], key="sem_y")

    # 2) Nomes sanitizados e preparo do dataframe
    orig = [X, M, Y]
    san = {orig[i]: f"V{i}" for i in range(3)}  # ex: {'PID-5-BF': 'V0', 'LSM-21': 'V1', 'AQ-50': 'V2'}
    df_sem = df[orig].dropna().rename(columns=san)

    # 2.1) Normalização opcional (mantém nomes sanitizados)
    normalize = st.checkbox(
        "Normalizar variáveis (z-score) para interpretação padronizada", value=False
    )
    if normalize:
        df_sem = (df_sem - df_sem.mean()) / df_sem.std()

    X_s, M_s, Y_s = san[X], san[M], san[Y]

    # 3) Ajuste do SEM
    model_desc = f"""
    {M_s} ~ a*{X_s}
    {Y_s} ~ b*{M_s} + c*{X_s}
    """
    sem = Model(model_desc)
    sem.fit(df_sem)

    # 4) Estimativas principais
    params = sem.inspect(std_est=True)
    a = float(params.loc[(params["lval"] == M_s) & (params["op"] == "~") & (params["rval"] == X_s), "Estimate"].iloc[0])
    b = float(params.loc[(params["lval"] == Y_s) & (params["op"] == "~") & (params["rval"] == M_s), "Estimate"].iloc[0])
    c_prime = float(params.loc[(params["lval"] == Y_s) & (params["op"] == "~") & (params["rval"] == X_s), "Estimate"].iloc[0])

    indirect = a * b          # efeito indireto
    direct = c_prime          # efeito direto (c′)
    total = direct + indirect # efeito total (c = c′ + a×b)

    # 5) Índices de ajuste global
    stats_df = calc_stats(sem)

    # 6) Exibição textual
    st.write("### Modelagem")
    st.markdown(f"""
    - Caminho a ({X} ➝ {M}): `{a:.3f}`  
    - Caminho b ({M} ➝ {Y}): `{b:.3f}`  
    - **Efeito total (c) {X} ➝ {Y}**: `{total:.3f}`  
    - **Efeito direto (c′, controlando {M})**: `{direct:.3f}`  
    - **Efeito indireto (a×b)**: `{indirect:.3f}`  
    """)


    # 7) Tabela de coeficientes SEM
    path_params = params[params["op"] == "~"].copy()
    inv_san = {v: k for k, v in san.items()}
    path_params["lval"] = path_params["lval"].map(inv_san)
    path_params["rval"] = path_params["rval"].map(inv_san)

    columns_map = {
        "Estimate": "Coeficiente",
        "SE": "Erro padrão",
        "z-value": "z-valor",
        "p-value": "p-valor"
    }
    index_labels = [f"{row['lval']} ➝ {row['rval']}" for _, row in path_params.iterrows()]
    sem_summary_df = pd.DataFrame(index=index_labels)
    for raw_col, display_col in columns_map.items():
        if raw_col in path_params.columns:
            sem_summary_df[display_col] = path_params[raw_col].values
    st.dataframe(sem_summary_df.style.format(precision=4))
    
    if normalize:
        st.caption("⚠️ Dados padronizados por z-score (média 0, desvio-padrão 1)")

    # 8) Bootstrap de intervalos de confiança
    st.write("### Bootstrap")
    n_boot = st.slider("Réplicas do bootstrap:", 100, 50000, 1000, 100)
    if st.button("Calcular intervalos de confiança", key="btn_sem_boot", use_container_width=True):
        indirect_boot, direct_boot = [], []
        with st.spinner("Executando bootstrap…"):
            for _ in range(n_boot):
                sample_df = df_sem.sample(n=len(df_sem), replace=True)
                sem_b = Model(model_desc)
                sem_b.fit(sample_df)
                p = sem_b.inspect(std_est=True)
                a_b = float(p.loc[(p["lval"] == M_s) & (p["op"] == "~") & (p["rval"] == X_s), "Estimate"])
                b_b = float(p.loc[(p["lval"] == Y_s) & (p["op"] == "~") & (p["rval"] == M_s), "Estimate"])
                c_b = float(p.loc[(p["lval"] == Y_s) & (p["op"] == "~") & (p["rval"] == X_s), "Estimate"])
                indirect_boot.append(a_b * b_b)
                direct_boot.append(c_b)
        ci_ind_low, ci_ind_high = np.percentile(indirect_boot, [2.5, 97.5])
        ci_dir_low, ci_dir_high = np.percentile(direct_boot, [2.5, 97.5])
        st.write("### Intervalos de Confiança (95%)")
        st.markdown(f"""
        - **Indireto (a×b)**: `{ci_ind_low:.3f}` a `{ci_ind_high:.3f}`  
        - **Direto (c′)**: `{ci_dir_low:.3f}` a `{ci_dir_high:.3f}`  
        """)

    # 9) Índices de ajuste do modelo
    st.write("### Índices de ajuste do modelo")
    st.dataframe(stats_df)

    # 10) Gráficos de efeito com matplotlib
    st.write("### Tamanho de efeito")
    effects = pd.DataFrame({"Efeito": ["Direto (c′)", "Indireto", "Total"], "Valor": [c_prime, indirect, total]})
    dark_bg, white, purple = "#0E1117", "#FFFFFF", "#7159c1"
    # tema escuro
    fig_bar_d, ax_bar_d = plt.subplots(figsize=(6, 3.5), facecolor=dark_bg)
    ax_bar_d.set_facecolor(dark_bg)
    ax_bar_d.bar(effects["Efeito"], effects["Valor"], color=purple, edgecolor=white, linewidth=1.5, alpha=0.8)
    ax_bar_d.axhline(0, color=white, linewidth=1.5)
    ax_bar_d.tick_params(colors=white)
    for spine in ax_bar_d.spines.values(): spine.set_edgecolor(white)
    plt.tight_layout()
    st.pyplot(fig_bar_d)

    # Downloads barras
    buf_bar_d = io.BytesIO(); fig_bar_d.savefig(buf_bar_d, format="png", facecolor=dark_bg); buf_bar_d.seek(0)
    buf_bar_l = io.BytesIO();
    fig_bar_l, ax_bar_l = plt.subplots(figsize=(6, 3.5), facecolor="white")
    ax_bar_l.bar(effects["Efeito"], effects["Valor"], color=purple, edgecolor="black", linewidth=1.5, alpha=0.8)
    ax_bar_l.axhline(0, color="black", linewidth=1.5)
    for spine in ax_bar_l.spines.values(): spine.set_edgecolor("black")
    plt.tight_layout(); fig_bar_l.savefig(buf_bar_l, format="png", facecolor="white"); buf_bar_l.seek(0)
    c1, c2 = st.columns(2)
    with c1: st.download_button("📥 Download (escuro)", data=buf_bar_d, file_name="sem_bar_dark.png", mime="image/png", use_container_width=True)
    with c2: st.download_button("📥 Download (claro)", data=buf_bar_l, file_name="sem_bar_light.png", mime="image/png", use_container_width=True)

    # 11) Sankey interativo
    st.write("### Diagrama de Sankey")
    st.caption("Visualize o fluxo de efeito direto, indireto e total entre X, M e Y.")
    fig_snk = go.Figure(go.Sankey(arrangement="snap",
        node=dict(label=[X, M, Y], pad=15, thickness=20, color=dark_bg, line=dict(color=white, width=1)),
        link=dict(
            source=[0, 1, 0], target=[1, 2, 2], value=[abs(a), abs(indirect), abs(c_prime)],
            color=["rgba(113,89,193,0.8)" if v>=0 else "rgba(193,89,113,0.8)" for v in [a, indirect, c_prime]]
        )
    ))
    fig_snk.update_layout(paper_bgcolor=dark_bg, font=dict(color=white, size=14), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_snk, use_container_width=True)

    # 12) Downloads Sankey
    html_dark = fig_snk.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    fig_snk.update_layout(paper_bgcolor="white", font=dict(color="black", size=14))
    html_light = fig_snk.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    col1, col2 = st.columns(2)
    with col1: st.download_button("📥 Download (escuro)", data=html_dark, file_name="sem_sankey_dark.html", mime="text/html", use_container_width=True)
    with col2: st.download_button("📥 Download (claro)", data=html_light, file_name="sem_sankey_light.html", mime="text/html", use_container_width=True)

def mediation_analysis(df: pd.DataFrame):
    """
    Executa análise de mediação simples (X → M → Y), com:
      • Gráfico de barras (efeitos direto, indireto e total) em tema escuro + download em claro/escuro
      • Diagrama alluvial (Sankey) com Plotly em tema escuro + download em claro/escuro
    """
    import copy

    from statsmodels.formula.api import ols
    
    # Seleção de variáveis
    st.write("### Análise de Mediação")
    st.caption("Selecione X (independente), M (mediadora) e Y (dependente).")

    numeric = df.select_dtypes(include="number").columns.tolist()
    if len(numeric) < 3:
        st.warning("É necessário pelo menos 3 variáveis numéricas.")
        return
    
    X = st.selectbox("X:", numeric, key="med_x")
    M = st.selectbox("M:", [c for c in numeric if c != X], key="med_m")
    Y = st.selectbox("Y:", [c for c in numeric if c not in (X, M)], key="med_y")

    # Ajuste dos modelos
    m_mod     = ols(f"Q('{M}') ~ Q('{X}')",              data=df).fit()
    y_mod     = ols(f"Q('{Y}') ~ Q('{X}') + Q('{M}')",   data=df).fit()
    total_mod = ols(f"Q('{Y}') ~ Q('{X}')",              data=df).fit()

    # 1) Modelo M ~ X
    df_mx = df[[X, M]].dropna()
    X_m = sm.add_constant(df_mx[X])
    m_mod = sm.OLS(df_mx[M], X_m).fit()
    a = m_mod.params[X]

    # 2) Modelo Y ~ X + M
    df_ymx = df[[X, M, Y]].dropna()
    X_ym = sm.add_constant(df_ymx[[X, M]])
    y_mod = sm.OLS(df_ymx[Y], X_ym).fit()
    b = y_mod.params[M]
    c_prime = y_mod.params[X]

    # 3) Modelo total Y ~ X
    df_yx = df[[X, Y]].dropna()
    X_y = sm.add_constant(df_yx[X])
    total_mod = sm.OLS(df_yx[Y], X_y).fit()
    c_total = total_mod.params[X]

    # 4) Efeito indireto
    indirect = a * b

    st.write("### Modelagem simples")
    # Exibe coeficientes
    st.markdown(f"""
    - Caminho a ({X} ➝ {M}): `{a:.3f}`  
    - Caminho b ({M} ➝ {Y}): `{b:.3f}`  
    - Total c ({X} ➝ {Y}): `{c_total:.3f}`  
    - Efeito direto (controlando M): `{c_prime:.3f}`  
    - Efeito indireto (a×b): `{indirect:.3f}`  
    """)

    # Mostra a tabela dos coeficientes
    summary_df = pd.DataFrame({
        "Coeficiente": y_mod.params,
        "Erro padrão": y_mod.bse,
        "t-valor": y_mod.tvalues,
        "p-valor": y_mod.pvalues
    })
    st.dataframe(summary_df.style.format(precision=4))

    # Mostra as métricas globais do modelo
    st.markdown("### Índices de ajuste do modelo")
    ols_metrics = pd.DataFrame([{
        "R²": y_mod.rsquared,
        "R² ajustado": y_mod.rsquared_adj,
        "F": y_mod.fvalue,
        "p(F)": y_mod.f_pvalue,
        "AIC": y_mod.aic,
        "BIC": y_mod.bic,
        "Log-likelihood": y_mod.llf,
        "N": int(y_mod.nobs),
        "DF Modelo": int(y_mod.df_model),
        "DF Resíduo": int(y_mod.df_resid)
    }])
    st.dataframe(ols_metrics.style.format(precision=4))



    # ─── BARRAS MATPLOTLIB ─────────────────────────────────────

    st.write("### Tamanho de efeito")

    effects = pd.DataFrame({
        "Efeito": ["Direto (c′)", "Indireto (a×b)", "Total (c)"],
        "Valor": [c_prime, indirect, c_total]
    })
    dark_bg, white, purple = "#0E1117", "#FFFFFF", "#7159c1"

    # Barras tema escuro
    fig_bar_d, ax_bar_d = plt.subplots(figsize=(6,3.5), facecolor=dark_bg)
    ax_bar_d.set_facecolor(dark_bg)
    bars = ax_bar_d.bar(effects["Efeito"], effects["Valor"],
                        color=purple, edgecolor=white, linewidth=1.5, alpha=0.8)
    ax_bar_d.axhline(0, color=white, linewidth=1.5)
    ax_bar_d.tick_params(colors=white)
    for spine in ax_bar_d.spines.values():
        spine.set_edgecolor(white)

    plt.tight_layout()
    st.pyplot(fig_bar_d)

    # Prepara buffers de download para barras
    buf_bar_d = io.BytesIO()
    fig_bar_d.savefig(buf_bar_d, format="png", facecolor=dark_bg)
    buf_bar_d.seek(0)

    # ─── BARRAS CLARO PARA DOWNLOAD ─────────────────────────────────────────────
    fig_bar_l, ax_bar_l = plt.subplots(figsize=(6,3.5), facecolor="white")
    ax_bar_l.set_facecolor("white")
    # redesenha as barras exatamente como no tema escuro
    bars_l = ax_bar_l.bar(
        effects["Efeito"],
        effects["Valor"],
        color=purple,            # ou defina outra cor, se preferir
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8
    )
    # linha de base
    ax_bar_l.axhline(0, color="black", linewidth=1.5)
    # estilização de eixos se precisar
    ax_bar_l.tick_params(colors="black")
    for spine in ax_bar_l.spines.values():
        spine.set_edgecolor("black")

    plt.tight_layout()
    # salva no buffer
    buf_bar_l = io.BytesIO()
    fig_bar_l.savefig(buf_bar_l, format="png", facecolor="white")
    buf_bar_l.seek(0)

    # ─── BOTÕES DE DOWNLOAD BARRAS ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download (escuro)",
                           data=buf_bar_d, file_name="mediacao_bar_dark.png",
                           mime="image/png", use_container_width=True)
    with c2:
        st.download_button("📥 Download (claro)",
                           data=buf_bar_l, file_name="mediacao_bar_light.png",
                           mime="image/png", use_container_width=True)

    st.caption("Cálculo inferencial [Statsmodels](https://www.statsmodels.org/stable/index.html) v0.14.4 | Plotagem [Matplotlib](https://matplotlib.org/stable/index.html) v3.10.5")

    # ─── SANKEY INTERATIVO EM TEMA ESCURO ─────────────────────────────────────────
    st.write("### Diagrama de Sankey")
    st.caption("O gráfico alluvial  —ou diagrama de Sankey — é um tipo de visualização que mostra como quantidades fluem entre diferentes categorias ou etapas de um processo. Ele é particularmente útil para representar relações causais, transições, partições e redistribuições de valores entre grupos.")
    
    fig_snk_d = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=[X, M, Y],
            pad=15,
            thickness=20,
            color="#0E1117",
            line=dict(color="white", width=1)
        ),
        link=dict(
            source=[0, 1, 0],
            target=[1, 2, 2],
            value=[abs(a), abs(indirect), abs(c_prime)],
            color=[
                "rgba(113,89,193,0.8)" if a >= 0 else "rgba(193,89,113,0.8)",
                "rgba(113,89,193,0.8)" if indirect >= 0 else "rgba(193,89,113,0.8)",
                "rgba(113,89,193,0.8)" if c_prime >= 0 else "rgba(193,89,113,0.8)"
            ]
        )
    ))
    
    fig_snk_d.update_layout(
        paper_bgcolor="#0E1117",
        font=dict(color="black", size=14),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Exibe interativamente no tema escuro
    st.plotly_chart(fig_snk_d, use_container_width=True)

    # ─── PREPARA DOIS HTMLs INTERATIVOS PARA DOWNLOAD ─────────────────────────────
    
    # Cria versão clara do gráfico para download
    fig_snk_g = copy.deepcopy(fig_snk_d)
    fig_snk_g.update_layout(
        paper_bgcolor="gray",
        font=dict(color="black", size=14)
    )
    html_dark = fig_snk_g.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    # Cria versão clara do gráfico para download
    fig_snk_l = copy.deepcopy(fig_snk_d)
    fig_snk_l.update_layout(
        paper_bgcolor="white",
        font=dict(color="black", size=14)
    )
    html_light = fig_snk_l.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    # ─── BOTÕES DE DOWNLOAD ALLUVIAL ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download (tema escuro)",
            data=html_dark,
            file_name="mediacao_sankey_dark.html",
            mime="text/html",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📥 Download (tema claro)",
            data=html_light,
            file_name="mediacao_sankey_light.html",
            mime="text/html",
            use_container_width=True
        )

def correlation_analysis(df: pd.DataFrame):
    """
    Calcula e exibe uma matriz de correlação com coeficientes e valores-p combinados,
    além de um heatmap visual com download em tema escuro e claro.
    """
    
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr, kendalltau

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("É necessário pelo menos duas variáveis numéricas.")
        return

    selected_cols = st.multiselect("Variáveis para matriz de correlação:", numeric_cols, default=None)

    if selected_cols:
        method = st.radio("Método:", ["Pearson", "Spearman", "Kendall"], horizontal=True)

        matrix_display = pd.DataFrame(index=selected_cols, columns=selected_cols)
        matrix_r       = pd.DataFrame(index=selected_cols, columns=selected_cols, dtype=float)

        for col1 in selected_cols:
            for col2 in selected_cols:
                x = df[col1].dropna()
                y = df[col2].dropna()
                x, y = x.align(y, join="inner")

                # ← checagem de tamanho mínimo
                if len(x) < 2:
                    matrix_r.loc[col1, col2]       = np.nan
                    matrix_display.loc[col1, col2] = "N/A"
                    continue

                # cálculo normal
                if method == "Pearson":
                    r, p = pearsonr(x, y)
                elif method == "Spearman":
                    r, p = spearmanr(x, y)
                else:  # Kendall
                    r, p = kendalltau(x, y)

                matrix_r.loc[col1, col2]       = r
                matrix_display.loc[col1, col2] = f"{r:.2f} (p={p:.3g})"

        # ─────────────────────────────────────────────────────
        # Tabela formatada
        st.dataframe(matrix_display, use_container_width=True)

        # ─────────────────────────────────────────────────────
        # Heatmap com tema escuro
    
        st.markdown("<br>", unsafe_allow_html=True)

        dark_bg = "#0E1117"
        fig_dark, ax_dark = plt.subplots(figsize=(6, 4), facecolor=dark_bg)
        ax_dark.set_facecolor(dark_bg)
            
        heatmap = sns.heatmap(
            matrix_r.astype(float),
            annot=True, fmt=".2f", cmap="magma",
            cbar=True, square=True, linewidths=0.5,
            annot_kws={"color": "white", "size": 9},
            ax=ax_dark
        )

        # Título e eixos escuros
        ax_dark.set_title("Matriz de Correlação", color="white")
        ax_dark.tick_params(colors="white")
        for spine in ax_dark.spines.values():
            spine.set_edgecolor("white")

        # Corrigir colorbar (barra lateral)
        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(colorbar.ax.yaxis.get_majorticklabels(), color='white')


        plt.tight_layout()
        st.pyplot(fig_dark)

        # Buffer para download (tema escuro)
        dark_buf = io.BytesIO()
        fig_dark.savefig(dark_buf, format="png", facecolor=dark_bg)
        dark_buf.seek(0)

        # Heatmap tema claro
        fig_light, ax_light = plt.subplots(figsize=(6, 4), facecolor="white")
        ax_light.set_facecolor("white")

        sns.heatmap(
            matrix_r.astype(float),
            annot=True, fmt=".2f", cmap="coolwarm",
            cbar=True, square=True, linewidths=0.5,
            annot_kws={"color": "white", "size": 9},
            ax=ax_light
        )
        ax_light.set_title("Matriz de Correlação")
        plt.tight_layout()

        light_buf = io.BytesIO()
        fig_light.savefig(light_buf, format="png", facecolor="white")
        light_buf.seek(0)

        # Botões de download
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download (tema escuro)",
                data=dark_buf,
                file_name="heatmap_correlacao_dark.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="📥 Download (tema claro)",
                data=light_buf,
                file_name="heatmap_correlacao_light.png",
                mime="image/png",
                use_container_width=True
            )

def scatter_interactive(df):
    """
    Torna interativo um scatter plot:
    - Escolha de eixos X e Y (somente numéricos)
    - Tema escuro
    - Botões de download HTML para dark e light
    """
    # 1) Verifica colunas numéricas
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.warning("Seu DataFrame precisa ter ao menos 2 colunas numéricas.")
        return

    # 2) Seletor de eixos
    x = st.selectbox("Eixo X:", num_cols, key="sc_x")
    y = st.selectbox("Eixo Y:", [c for c in num_cols if c != x], key="sc_y")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3) Monta o gráfico com tema escuro
    custom_color = "rgba(98, 75, 175, 0.808)"
    fig_dark = px.scatter(
        df,
        x=x,
        y=y,
        opacity=0.75,
        template="plotly_dark",
        color_discrete_sequence=[custom_color]
    )
    fig_dark.update_traces(marker=dict(size=6, line=dict(width=0)))

    st.plotly_chart(fig_dark, use_container_width=True)

    # 4) Prepara HTML para download
    html_dark = fig_dark.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    fig_light = fig_dark.to_html()  # workaround: clonamos a saída
    # para gerar versão clara, recarregamos como objeto
    import plotly.io as pio
    fig_light = pio.from_json(fig_dark.to_json())
    fig_light.update_layout(template="plotly_white")
    html_light = fig_light.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    # 5) Botões de download
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Download (tema escuro)",
            data=html_dark,
            file_name=f"scatter_{x}_{y}_dark.html",
            mime="text/html",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "📥 Download (tema claro)",
            data=html_light,
            file_name=f"scatter_{x}_{y}_light.html",
            mime="text/html",
            use_container_width=True
        )

def linear_regression_analysis(df: pd.DataFrame):
    """
    Realiza regressão linear simples entre duas variáveis numéricas, com visualização gráfica e download.
    """

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("É necessário pelo menos duas variáveis numéricas.")
        return

    col_x = st.selectbox("Variável Independente (X):", numeric_cols, key="reg_x")
    col_y = st.selectbox("Variável Dependente (Y):", [c for c in numeric_cols if c != col_x], key="reg_y")

    x = df[col_x].dropna()
    y = df[col_y].dropna()
    x, y = x.align(y, join="inner")

    X_const = sm.add_constant(x)  # Adiciona intercepto
    model = sm.OLS(y, X_const).fit()

    intercept = model.params['const']
    slope = model.params[col_x]
    r_squared = model.rsquared
    p_value = model.pvalues[col_x]

    st.markdown(f"**Equação:** `{col_y} = {intercept:.3f} + {slope:.3f} × {col_x}`")
    st.markdown(f"**R²:** `{r_squared:.3f}`")
    st.markdown(f"**Valor-p (coef. X):** `{p_value:.4g}`")
    
    st.subheader("Linha de Regressão")

    # ───────────────────────────────────────────────────────
    # Gráfico com tema escuro
    dark_bg = "#0E1117"
    white = "#FFFFFF"
    purple = "#7159c1"

    fig_dark, ax_dark = plt.subplots(facecolor=dark_bg)
    ax_dark.set_facecolor(dark_bg)

    # Pontos
    ax_dark.scatter(x, y, color=purple, edgecolors=white, alpha=0.7)

    # Linha de regressão
    x_line = pd.Series(sorted(x))
    y_line = intercept + slope * x_line
    ax_dark.plot(x_line, y_line, color=white, linewidth=2, label="Regressão Linear")

    ax_dark.set_xlabel(col_x, color=white)
    ax_dark.set_ylabel(col_y, color=white)
    ax_dark.legend(facecolor=dark_bg, edgecolor=white, labelcolor=white)

    ax_dark.tick_params(colors=white)
    for spine in ax_dark.spines.values():
        spine.set_edgecolor(white)

    plt.tight_layout()
    st.pyplot(fig_dark)

    # Buffer para download (dark)
    dark_buf = io.BytesIO()
    fig_dark.savefig(dark_buf, format="png", facecolor=dark_bg)
    dark_buf.seek(0)

    # ───────────────────────────────────────────────────────
    # Gráfico tema claro
    plt.style.use("default")
    fig_light, ax_light = plt.subplots()
    ax_light.scatter(x, y, color=purple, edgecolors="black", alpha=0.7)
    ax_light.plot(x_line, y_line, color="black", linewidth=2, label="Regressão Linear")
    ax_light.set_title(f"{col_y} em função de {col_x}")
    ax_light.set_xlabel(col_x)
    ax_light.set_ylabel(col_y)
    ax_light.legend()
    plt.tight_layout()

    light_buf = io.BytesIO()
    fig_light.savefig(light_buf, format="png", facecolor="white")
    light_buf.seek(0)

    # ───────────────────────────────────────────────────────
    # Botões de download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download (tema escuro)",
            data=dark_buf,
            file_name=f"regressao_{col_y}_vs_{col_x}_dark.png",
            mime="image/png",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📥 Download (tema claro)",
            data=light_buf,
            file_name=f"regressao_{col_y}_vs_{col_x}_light.png",
            mime="image/png",
            use_container_width=True
        )


def density_by_category(df: pd.DataFrame):
    """
    Exibe KDE/Histograma de uma variável numérica por níveis de uma variável categórica,
    com paleta Plotly, mapeamento cor→cluster personalizável e downloads dark/light.
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    import itertools
    import plotly.figure_factory as ff
    import plotly.io as pio
    from plotly.express import colors
    from pandas.api.types import is_numeric_dtype

    # 1) Tipos
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    low_card_numeric = [c for c in df.columns if is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 20]
    cat_cols = list(dict.fromkeys(cat_cols + low_card_numeric))

    if not num_cols or not cat_cols:
        st.warning("Preciso de pelo menos 1 coluna numérica e 1 categórica.")
        return

    # 2) Seleção de colunas
    x_cat = st.selectbox("Variável categórica (ex.: sexo/cluster):", cat_cols, key="kde_cat")
    y_num = st.selectbox("Variável numérica (ex.: escore):", num_cols, key="kde_num")

    st.caption("KDE supõe variável contínua. Para Likert discreta, use histograma/ECDF.")

    # 3) Parâmetros visuais
    c1, c2, c3 = st.columns(3)
    with c1:
        show_rug = st.toggle("Mostrar rug", value=False)
    with c2:
        show_hist = st.toggle("Mostrar histograma", value=False)
    with c3:
        common_norm = st.toggle("Normalizar", value=True)

    # 4) Paleta
    palettes = {k: v for k, v in colors.qualitative.__dict__.items() if isinstance(v, list)}
    palette_name = st.selectbox("Paleta de cores:", options=sorted(palettes.keys()))
    base_palette = palettes[palette_name]

    # 5) Dados por grupo
    grouped = df[[x_cat, y_num]].dropna()
    series, labels = [], []
    invalid_labels = []
    for level, sub in grouped.groupby(x_cat, dropna=False):
        vals = pd.to_numeric(sub[y_num], errors="coerce").dropna().values
        if len(vals) >= 2 and np.unique(vals).size >= 2:
            series.append(vals)
            labels.append(str(level))
        elif len(vals) > 0:
            invalid_labels.append(str(level))

    if not series:
        if invalid_labels:
            st.warning(
                "Sem dados suficientes para estimar densidades. "
                "Algumas categorias têm menos de 2 valores distintos e não podem ser usadas em KDE: "
                + ", ".join(invalid_labels)
            )
        else:
            st.info("Sem dados suficientes para estimar densidades.")
        return

    # 6) Mapeamento cor→cluster (persistente)
    use_fixed_map = st.toggle("Fixar mapeamento de cores", value=False)
    map_key = f"color_map::{x_cat}::{palette_name}"
    if "color_maps" not in st.session_state:
        st.session_state["color_maps"] = {}

    # cria default se não existir (ordem estável: labels ordenados mapeados para paleta cíclica)
    if map_key not in st.session_state["color_maps"]:
        default_map = {}
        for i, lab in enumerate(sorted(labels)):
            default_map[lab] = base_palette[i % len(base_palette)]
        st.session_state["color_maps"][map_key] = default_map

    # UI para editar mapeamento
    color_map = st.session_state["color_maps"][map_key].copy()
    cols_per_row = 3
    if use_fixed_map:

        rows = [sorted(labels)[i:i+cols_per_row] for i in range(0, len(sorted(labels)), cols_per_row)]
        for row in rows:
            rcols = st.columns(len(row))
            for j, lab in enumerate(row):
                # opções: somente cores da paleta atual
                with rcols[j]:
                    sel = st.selectbox(
                        f"{x_cat} = {lab}",
                        options=base_palette,
                        index=base_palette.index(color_map.get(lab, base_palette[0])) if color_map.get(lab) in base_palette else 0,
                        key=f"map_{map_key}_{lab}"
                    )
                    color_map[lab] = sel

        if st.button("Reiniciar mapeamento", use_container_width=True):
            st.session_state["color_maps"].pop(map_key, None)
            st.rerun()

        # salva alterações
        st.session_state["color_maps"][map_key] = color_map

    # 7) Lista de cores final na ordem de `labels` (não ordenadas!)
    if use_fixed_map:
        # se aparecer um cluster novo que não estava no mapa, atribui próxima cor da paleta
        for lab in labels:
            if lab not in color_map:
                nxt = base_palette[len(color_map) % len(base_palette)]
                color_map[lab] = nxt
        color_list = [color_map[lab] for lab in labels]
    else:
        # sem mapeamento fixo: paleta cíclica na ordem dos labels atuais
        color_list = list(itertools.islice(itertools.cycle(base_palette), len(labels)))

    bin_size = None
    if show_hist:
        bin_size = [None] * len(series)  # cria lista explícita

    # 8) Figura (tema escuro)
    try:
        fig_dark = ff.create_distplot(
            hist_data=series,
            group_labels=labels,
            bin_size=bin_size,
            show_hist=show_hist,
            show_rug=show_rug,
            colors=color_list
        )
    except np.linalg.LinAlgError:
        st.error(
            "Não foi possível estimar KDE. "
            "Os dados de uma ou mais categorias parecem ter variância zero ou estão em subespaço de baixa dimensão. "
            "Tente outra variável numérica ou reduza a cardinalidade da variável categórica."
        )
        return


    fig_dark.update_layout(
            template="plotly_dark",
            xaxis_title=y_num,
            yaxis_title="Density" if not show_hist else "Frequência",
            legend_title=x_cat,
            hovermode="x unified"
        )

    # 9) Normalização conjunta (somente KDE)
    if common_norm and not show_hist and len(fig_dark.data) > 0:
        areas = []
        for tr in fig_dark.data:
            if tr.type == "scatter" and tr.mode == "lines":
                x = np.array(tr.x, dtype=float)
                y = np.array(tr.y, dtype=float)
                areas.append(np.trapz(y, x))
            else:
                areas.append(0.0)
        total_area = sum(a for a in areas if a > 0)
        if total_area > 0:
            k = 1.0 / total_area
            for tr in fig_dark.data:
                if tr.type == "scatter" and tr.mode == "lines":
                    tr.update(y=(np.array(tr.y, dtype=float) * k))

    # 10) Exibir + downloads
    st.plotly_chart(fig_dark, use_container_width=True)

    html_dark = fig_dark.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    fig_light = pio.from_json(fig_dark.to_json())
    html_light = fig_light.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    cdl1, cdl2 = st.columns(2)
    with cdl1:
        st.download_button("📥 Download (tema escuro)", data=html_dark,
                           file_name=f"kde_{y_num}_by_{x_cat}_dark.html",
                           mime="text/html", use_container_width=True)
    with cdl2:
        st.download_button("📥 Download (tema claro)", data=html_light,
                           file_name=f"kde_{y_num}_by_{x_cat}_light.html",
                           mime="text/html", use_container_width=True)



def render_graph_selector() -> str:
    """
    <docstrings>
    Exibe o seletor da grande área de ML.
    """
    return st.radio(
        "Escolha o tipo de visualização de variáveis:",
        (
            "Gráfico de dispersão",
            "Gráfico de densidade por categoria"
        ),
        horizontal=True,
        key="graph_choice",
    )

# PAGE 3 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

st.title("Estatística Inferencial")

st.caption("""
A seção **Estatística Inferencial** permite investigar relações entre variáveis por meio de testes de correlação, regressão linear e visualizações analíticas. 
Inclui cálculo dos coeficientes de **Pearson**, **Spearman** e **Kendall**, com valores-p para teste de significância, além de gráficos de dispersão e **regressão linear simples** com equações ajustadas e métricas como **R²** e valor-p do coeficiente. 
""")

# Verify dataframe
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Este dataframe não possui colunas numéricas.")
    st.stop()

df_names = list(st.session_state.dataframes.keys())

selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names)
df = st.session_state.dataframes[selected_df_name]

num_cols = df.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.warning("Este dataframe não possui colunas numéricas.")
    st.stop()

st.divider()

# BODY ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# Executa os módulos de inferência
st.write("### Exploração visual")

graph = render_graph_selector()

if graph == "Gráfico de dispersão":
    scatter_interactive(df)
if graph == "Gráfico de densidade por categoria":
    density_by_category(df)

st.divider()

st.write("### Testagem de associação entre variáveis")

option = st.radio(
    "Selecione o tipo de análise:",
    [
        "Correlação",
        "Regressão linear simples",
        "Análise de mediação",
        "Modelagem de equações estruturais"
    ]
)

st.markdown("<br>", unsafe_allow_html=True)

if option == "Correlação":
    correlation_analysis(df)
    st.caption(
        "Cálculo inferencial [SciPy](https://docs.scipy.org/doc/scipy/) v.1.16.1 | "
        "Heatmap [Seaborn](https://seaborn.pydata.org/) 0.13.2"
    )

elif option == "Regressão linear simples":
    linear_regression_analysis(df)
    st.caption(
        "Cálculo inferencial [Statsmodels](https://www.statsmodels.org/stable/index.html) v0.14.4 | "
        "Plotagem [Matplotlib](https://matplotlib.org/stable/index.html) v3.10.52"
    )

elif option == "Análise de mediação":
    mediation_analysis(df)
    st.caption(
        "Cálculo inferencial [Statsmodels](https://www.statsmodels.org/stable/index.html) v0.14.4 | "
        "Diagrama [Plotly](https://plotly.com/) v2.30"
    )

elif option == "Modelagem de equações estruturais":
    sem_mediation_analysis(df)
    st.caption(
        "Cálculo inferencial [Statsmodels](https://www.statsmodels.org/stable/index.html) v0.14.4 | "
        "Diagrama [Plotly](https://plotly.com/) v2.30"
    )

