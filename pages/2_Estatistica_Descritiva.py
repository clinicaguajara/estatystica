# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit   as st
import pandas      as pd
import plotly.express as px
import plotly.io as pio
import difflib

from utils.design import load_css

# CUSTOM FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           
def describe_numeric_column(df: pd.DataFrame, df_name="selected_df_name"):
    """
    <docstrings>
    Exibe estatísticas descritivas e visualizações para uma coluna numérica selecionada de um DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a ser descrito.

    Calls:
        df.select_dtypes(): Seleciona colunas numéricas | método do DataFrame.
        st.selectbox(): Seleciona coluna a ser descrita | instanciado por st.
        st.table(): Exibe estatísticas como tabela | instanciado por st.
        plt.subplots(): Criação de gráficos | método do matplotlib.
        pd.Series.mode(): Retorna o(s) valor(es) mais frequentes | método de Series.
        pd.Series.quantile(): Calcula percentis da distribuição | método de Series.
        pd.Series.skew(): Calcula a assimetria | método de Series.
        pd.Series.kurtosis(): Calcula a curtose | método de Series.

    Raises:
        st.warning: Caso o dataframe não tenha colunas numéricas.
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import io

    # Verificação de integridade antes da renderização
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] == 0:
        st.warning(f"O dataframe '{df_name}' está vazio ou inválido.")
        st.stop()

    # ───────────────────────────────────────────────────────
    # Verifica e seleciona coluna numérica
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("Este dataframe não possui colunas numéricas.")
        return

    selected_col = st.selectbox(f"Selecione uma coluna numérica para descrever em **{df_name}**:", numeric_cols)
    col_data = df[selected_col].dropna()

    # ───────────────────────────────────────────────────────
    # Estatísticas descritivas
    mode_value = col_data.mode()
    moda = mode_value.iloc[0] if not mode_value.empty else None

    q1 = col_data.quantile(0.25)
    q2 = col_data.quantile(0.50)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1

    # Tendência Central
    tendencia_central = {
        "Média": col_data.mean(),
        "Mediana": col_data.median(),
        "Moda": moda,
        "Q1 (25%)": q1,
        "Q2 (50%)": q2,
        "Q3 (75%)": q3,
    }

    # Dispersão e Forma
    dispersao = {
        "Desvio Padrão": col_data.std(),
        "Variância": col_data.var(),
        "IQR (Q3 - Q1)": iqr,
        "Amplitude": col_data.max() - col_data.min(),
        "Mínimo": col_data.min(),
        "Máximo": col_data.max(),
        "Assimetria (Skewness)": col_data.skew(),
        "Curtose": col_data.kurtosis(),
        "Valores Ausentes": df[selected_col].isna().sum(),
        "Valores Únicos": col_data.nunique()
    }

    # ───────────────────────────────────────────────────────
    # Visualização gráfica
    plot_type = st.radio("Escolha o tipo de gráfico:", [ "Curva de Densidade", "Histograma", "Boxplot"], horizontal=True)

    col_data_clean = col_data.dropna()
    dark_bg = "#0E1117"
    white = "#FFFFFF"
    purple = "#7159c1"

    # Gráfico modo escuro
    fig, ax = plt.subplots(facecolor=dark_bg)
    ax.set_facecolor(dark_bg)
    if plot_type == "Histograma":
            # Frequência por valor único (exato, sem bins)
            valores_unicos = sorted(col_data_clean.unique())
            counts = col_data_clean.value_counts().sort_index()

            bars = ax.bar(valores_unicos, counts, color=purple, edgecolor=white, width=0.6)

            ax.set_title(f"Histograma de {selected_col}", color=white)
            ax.set_xlabel(selected_col, color=white)
            ax.set_ylabel("Frequência", color=white)
            ax.tick_params(colors=white)
            # Anota frequências em cada barra
            for rect in bars:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, height, int(height),
                        ha='center', va='bottom', color=white)


    elif plot_type == "Boxplot":
        ax.boxplot(
            col_data_clean, vert=False,
            boxprops=dict(color=white),
            capprops=dict(color=white),
            whiskerprops=dict(color=white),
            flierprops=dict(markeredgecolor=purple),
            medianprops=dict(color=white)
        )
        ax.plot(col_data_clean.mean(), 1, 'o', color=purple, label='Média')
        ax.set_xlabel(selected_col, color=white)

    elif plot_type == "Curva de Densidade":
        col_data_clean.plot(kind='density', ax=ax, color=purple)
        ax.set_xlabel(selected_col, color=white)
        ax.set_ylabel("Densidade", color=white)

    ax.tick_params(colors=white)
    for spine in ax.spines.values():
        spine.set_edgecolor(white)

    plt.tight_layout()
    st.pyplot(fig)

    # ───────────────────────────────────────────────────────
    # Download dos gráficos

    dark_buf = io.BytesIO()
    fig.savefig(dark_buf, format="png")
    dark_buf.seek(0)

    plt.style.use("default")
    light_fig, light_ax = plt.subplots(facecolor="white")
    light_ax.set_facecolor("white")

    if plot_type == "Histograma":
        light_ax.hist(col_data_clean, bins=20, color=purple, edgecolor="black")
        light_ax.set_title(f"Histograma de {selected_col}")
        light_ax.set_xlabel(selected_col)
        light_ax.set_ylabel("Frequência")

    elif plot_type == "Boxplot":
        light_ax.boxplot(
            col_data_clean, vert=False,
            boxprops=dict(color="black"),
            capprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            flierprops=dict(markeredgecolor=purple),
            medianprops=dict(color="black")
        )
        light_ax.set_title(f"Boxplot de {selected_col}")
        light_ax.set_xlabel(selected_col)

    elif plot_type == "Curva de Densidade":
        col_data_clean.plot(kind='density', ax=light_ax, color=purple)
        light_ax.set_title(f"Curva de Densidade de {selected_col}")
        light_ax.set_xlabel(selected_col)

    light_fig.tight_layout()
    light_buf = io.BytesIO()
    light_fig.savefig(light_buf, format="png", facecolor="white")
    light_buf.seek(0)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download (tema escuro)",
            data=dark_buf,
            file_name=f"{selected_col}_{plot_type.lower().replace(' ', '_')}_dark.png",
            mime="image/png",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📥 Download (tema claro)",
            data=light_buf,
            file_name=f"{selected_col}_{plot_type.lower().replace(' ', '_')}_light.png",
            mime="image/png",
            use_container_width=True
        )


    # Renderiza as tabelas
    st.write("### Medidas de tendência central")
    st.caption("Métricas que resumem a localização dos dados na distribuição.")
    st.table(pd.DataFrame(tendencia_central.items(), columns=["Estatística", "Valor"]))
    st.caption("Cálculo descritivo [Matplotlib](https://matplotlib.org/stable/index.html) v3.10.5 | Manipulação de dataframes [Pandas](https://pandas.pydata.org/docs/) v.2.3.1")

    st.write("### Medidas de dispersão e forma")
    st.caption("Indicadores de variabilidade, amplitude e o formato da distribuição.")
    st.table(pd.DataFrame(dispersao.items(), columns=["Estatística", "Valor"]))
    st.caption("Cálculo descritivo [Matplotlib](https://matplotlib.org/stable/index.html) v3.10.5 | Manipulação de dataframes [Pandas](https://pandas.pydata.org/docs/) v.2.3.1")


def describe_categorical_column(df: pd.DataFrame, df_name="selected_df_name"):
    """
    Exibe visualizações categóricas por coluna:
    1) Contagem de frequências
    2) Quadrados proporcionais (treemap)
    """
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        st.info("Este dataframe não possui colunas categóricas para visualização.")
        return

    st.write("### Visualização categórica por coluna")
    selected_col = st.selectbox(
        f"Selecione uma coluna categórica para descrever em **{df_name}**:",
        cat_cols,
        key="cat_selected_col",
    )

    include_na = st.toggle("Incluir valores ausentes como categoria", value=False, key="cat_include_na")
    max_categories = st.slider(
        "Número máximo de categorias no gráfico:",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        key="cat_max_categories",
    )

    series = df[selected_col]
    if include_na:
        series = series.fillna("NA")
    else:
        series = series.dropna()

    counts = series.astype(str).value_counts()
    if counts.empty:
        st.warning("A coluna selecionada não possui dados válidos para visualização.")
        return

    top_counts = counts.head(max_categories).copy()
    outros = int(counts.iloc[max_categories:].sum())
    if outros > 0:
        top_counts.loc["Outros"] = outros

    freq_df = top_counts.reset_index()
    freq_df.columns = ["categoria", "frequencia"]
    freq_df["percentual"] = (freq_df["frequencia"] / freq_df["frequencia"].sum() * 100).round(2)

    chart_type = st.radio(
        "Escolha o tipo de gráfico para categorias:",
        ["Contagem de frequências", "Quadrados proporcionais", "Sunburst", "Choropleth (países)"],
        horizontal=True,
        key="cat_plot_type",
    )

    if chart_type == "Contagem de frequências":
        fig = px.bar(
            freq_df,
            x="categoria",
            y="frequencia",
            text="frequencia",
            color="categoria",
            # Define paleta explícita para garantir cores consistentes no HTML exportado.
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title=f"Frequência por categoria: {selected_col}",
        )
        fig.update_layout(showlegend=False, xaxis_title=selected_col, yaxis_title="Frequência")
    elif chart_type in ["Quadrados proporcionais", "Sunburst"]:
        is_sunburst = chart_type == "Sunburst"
        hier_plot_fn = px.sunburst if is_sunburst else px.treemap
        hier_title = f"{chart_type} por categoria: {selected_col}"

        # Inclui o tipo do gráfico na chave para separar as preferências de treemap x sunburst.
        treemap_key_base = f"{df_name}::{selected_col}::{chart_type}"
        treemap_df = freq_df.copy()

        with st.expander(f"Configurar {chart_type.lower()}"):
            density_mode_tree = st.radio(
                "Escala de densidade:",
                [
                    "Contínua (linear)",
                    "Contínua (log10)",
                    "Faixas automáticas (quantis)",
                    "Faixas manuais",
                ],
                horizontal=True,
                key=f"treemap_density_mode::{treemap_key_base}",
            )
            text_mode_tree = st.selectbox(
                "Texto nas áreas:",
                [
                    "Categoria + frequência + %",
                    "Categoria + frequência",
                    "Categoria + %",
                    "Só categoria",
                ],
                index=0,
                key=f"treemap_text_mode::{treemap_key_base}",
            )
            text_size_tree = st.slider(
                "Tamanho da fonte nas áreas:",
                min_value=9,
                max_value=32,
                value=14,
                step=1,
                key=f"treemap_text_size::{treemap_key_base}",
            )
            text_color_tree = st.selectbox(
                "Cor do texto:",
                ["Automática", "Branco", "Preto"],
                index=0,
                key=f"treemap_text_color::{treemap_key_base}",
            )
            border_width_tree = st.slider(
                "Espessura da borda entre áreas:",
                min_value=0.0,
                max_value=4.0,
                value=1.0,
                step=0.5,
                key=f"treemap_border_width::{treemap_key_base}",
            )

            if density_mode_tree == "Faixas automáticas (quantis)":
                n_bins_tree = st.slider(
                    "Número de faixas:",
                    min_value=3,
                    max_value=7,
                    value=5,
                    key=f"treemap_density_bins::{treemap_key_base}",
                )
            else:
                n_bins_tree = None

            if density_mode_tree == "Faixas manuais":
                min_freq_tree = int(treemap_df["frequencia"].min())
                max_freq_tree = int(treemap_df["frequencia"].max())
                default_cut_values_tree = sorted(
                    {
                        int(treemap_df["frequencia"].quantile(0.25)),
                        int(treemap_df["frequencia"].quantile(0.50)),
                        int(treemap_df["frequencia"].quantile(0.75)),
                    }
                )
                default_cut_values_tree = [v for v in default_cut_values_tree if min_freq_tree < v < max_freq_tree]
                default_cut_text_tree = ", ".join(str(v) for v in default_cut_values_tree)

                raw_cut_text_tree = st.text_input(
                    "Cortes (separados por vírgula). Ex.: 100, 300, 800",
                    value=default_cut_text_tree,
                    key=f"treemap_density_manual_cuts::{treemap_key_base}",
                )
            else:
                raw_cut_text_tree = ""

        if density_mode_tree == "Contínua (linear)":
            fig = hier_plot_fn(
                treemap_df,
                path=["categoria"],
                values="frequencia",
                color="frequencia",
                color_continuous_scale="Blues",
                labels={"frequencia": "Frequency", "percentual": "Percentual (%)"},
                title=hier_title,
            )
        elif density_mode_tree == "Contínua (log10)":
            import math

            treemap_df_log = treemap_df.copy()
            treemap_df_log["frequencia_log10"] = treemap_df_log["frequencia"].apply(lambda v: math.log10(v + 1))
            fig = hier_plot_fn(
                treemap_df_log,
                path=["categoria"],
                values="frequencia",
                color="frequencia_log10",
                color_continuous_scale="Blues",
                labels={
                    "frequencia": "Frequency",
                    "frequencia_log10": "log10(Frequency + 1)",
                    "percentual": "Percentual (%)",
                },
                title=hier_title,
            )
            fig.update_traces(marker_colorbar_title="log10(Frequency + 1)")
        else:
            treemap_df_binned = treemap_df.copy()

            if density_mode_tree == "Faixas automáticas (quantis)":
                n_unique_tree = int(treemap_df_binned["frequencia"].nunique())
                n_bins_effective_tree = max(1, min(int(n_bins_tree or 5), n_unique_tree))

                if n_bins_effective_tree == 1:
                    treemap_df_binned["faixa_densidade"] = "Faixa única"
                    category_order_tree = ["Faixa única"]
                else:
                    faixa_cat_tree = pd.qcut(
                        treemap_df_binned["frequencia"],
                        q=n_bins_effective_tree,
                        duplicates="drop",
                    )
                    category_order_tree = [str(c) for c in faixa_cat_tree.cat.categories]
                    treemap_df_binned["faixa_densidade"] = faixa_cat_tree.astype(str)
            else:
                min_freq_tree = int(treemap_df_binned["frequencia"].min())
                max_freq_tree = int(treemap_df_binned["frequencia"].max())
                parsed_values_tree = []
                parse_error_tree = False
                for piece in raw_cut_text_tree.replace(";", ",").split(","):
                    token = piece.strip()
                    if not token:
                        continue
                    try:
                        parsed_values_tree.append(float(token))
                    except ValueError:
                        parse_error_tree = True
                        break

                if parse_error_tree:
                    st.warning("Cortes manuais inválidos na visualização hierárquica. Use apenas números separados por vírgula.")
                    treemap_df_binned["faixa_densidade"] = "Faixa única"
                    category_order_tree = ["Faixa única"]
                else:
                    cut_values_tree = sorted(set(v for v in parsed_values_tree if min_freq_tree < v < max_freq_tree))
                    bins_tree = [float("-inf")] + cut_values_tree + [float("inf")]
                    faixa_cat_tree = pd.cut(
                        treemap_df_binned["frequencia"],
                        bins=bins_tree,
                        include_lowest=True,
                    )
                    category_order_tree = [str(c) for c in faixa_cat_tree.cat.categories]
                    treemap_df_binned["faixa_densidade"] = faixa_cat_tree.astype(str)

            if len(category_order_tree) <= 1:
                density_palette_tree = [px.colors.sequential.Blues[-1]]
            else:
                density_palette_tree = px.colors.sample_colorscale(
                    "Blues",
                    [0.20 + 0.75 * (i / (len(category_order_tree) - 1)) for i in range(len(category_order_tree))],
                )
            density_color_map_tree = dict(zip(category_order_tree, density_palette_tree))
            treemap_df_binned["faixa_densidade"] = pd.Categorical(
                treemap_df_binned["faixa_densidade"],
                categories=category_order_tree,
                ordered=True,
            )

            fig = hier_plot_fn(
                treemap_df_binned,
                path=["categoria"],
                values="frequencia",
                color="faixa_densidade",
                color_discrete_map=density_color_map_tree,
                labels={
                    "frequencia": "Frequency",
                    "faixa_densidade": "Faixa de densidade",
                    "percentual": "Percentual (%)",
                },
                title=hier_title,
            )

        if text_mode_tree == "Categoria + frequência + %":
            text_template_tree = "%{label}<br>%{value}<br>%{percentRoot:.1%}"
        elif text_mode_tree == "Categoria + frequência":
            text_template_tree = "%{label}<br>%{value}"
        elif text_mode_tree == "Categoria + %":
            text_template_tree = "%{label}<br>%{percentRoot:.1%}"
        else:
            text_template_tree = "%{label}"

        fig.update_traces(
            texttemplate=text_template_tree,
            textfont_size=text_size_tree,
            marker_line_width=border_width_tree,
        )
        if text_color_tree == "Branco":
            fig.update_traces(textfont_color="#FFFFFF")
        elif text_color_tree == "Preto":
            fig.update_traces(textfont_color="#000000")
    else:
        # Para mapa, usamos todas as categorias (sem limite top N) para preservar cobertura.
        # Isso evita vieses de visualização causados por truncar apenas as categorias mais frequentes.
        full_counts = counts.reset_index()
        full_counts.columns = ["categoria", "frequencia"]

        # Camada 1 de robustez: normalização manual de aliases comuns (PT/EN/siglas).
        # Objetivo: corrigir casos frequentes antes da tentativa de resolução automática.
        country_alias = {
            "Brasil": "Brazil",
            "BRASIL": "Brazil",
            "EUA": "United States",
            "USA": "United States",
            "UK": "United Kingdom",
            "Inglaterra": "United Kingdom",
            "Coreia do Sul": "South Korea",
            "Rússia": "Russia",
            "República Tcheca": "Czech Republic",
        }
        full_counts["categoria_norm"] = full_counts["categoria"].astype(str).str.strip()
        full_counts["pais_plot"] = full_counts["categoria_norm"].replace(country_alias)

        try:
            import pycountry
            # Camada 2 de robustez: catálogo oficial de países (ISO 3166 via pycountry).
            # Quando disponível, o mapa usa ISO-3 para reduzir ambiguidades de nomes.
            pycountry_names = sorted([c.name for c in pycountry.countries])
            has_pycountry = True
        except Exception:
            pycountry = None
            # Fallback: lista do gapminder (mais limitada que pycountry).
            # Mantém o gráfico funcional, mas com validação menos rigorosa.
            pycountry_names = sorted(set(px.data.gapminder()["country"].unique().tolist()))
            has_pycountry = False

        known_countries_list = pycountry_names
        known_countries = set(known_countries_list)

        def resolve_country_to_iso3(country_name: str):
            # Resolve um texto de país para:
            # - (ISO3, nome canônico) quando pycountry está disponível
            # - (nome, nome) no fallback por nome textual
            # - (None, nome) quando não reconhecido
            raw_name = str(country_name).strip()
            normalized_name = country_alias.get(raw_name, raw_name)

            if has_pycountry:
                try:
                    # lookup cobre nome oficial, alpha-2/alpha-3 e alguns nomes alternativos.
                    c = pycountry.countries.lookup(normalized_name)
                    return c.alpha_3, c.name
                except LookupError:
                    try:
                        # search_fuzzy tenta aproximação para entradas quase corretas.
                        c = pycountry.countries.search_fuzzy(normalized_name)[0]
                        return c.alpha_3, c.name
                    except LookupError:
                        return None, normalized_name

            # Fallback sem pycountry: só aceita países que já existam na lista conhecida.
            # Sem pycountry, o reconhecimento é mais frágil para variações de escrita.
            if normalized_name in known_countries:
                return normalized_name, normalized_name
            return None, normalized_name

        # Resolve para um identificador robusto de mapa (ISO3 quando pycountry estiver disponível).
        resolved_pairs = full_counts["pais_plot"].apply(resolve_country_to_iso3)
        full_counts["map_location"] = resolved_pairs.apply(lambda x: x[0])
        full_counts["pais_resolvido"] = resolved_pairs.apply(lambda x: x[1])

        # Sem pycountry não há validação robusta de correspondência.
        # Aqui priorizamos exibir o mapa por nome textual em vez de bloquear categorias.
        if not has_pycountry:
            full_counts["map_location"] = full_counts["pais_plot"]

        # Chave base para persistir preferências deste mapa (por dataframe + coluna).
        map_key_base = f"{df_name}::{selected_col}"

        # Camada 3 de robustez: intervenção manual guiada para casos não reconhecidos.
        # O mapeamento fica persistido no session_state por dataframe+coluna.
        not_found = full_counts[full_counts["map_location"].isna()].copy() if has_pycountry else pd.DataFrame()
        if not not_found.empty:
            st.warning("Algumas categorias não foram reconhecidas como países.")
            st.caption("Escolha manualmente um país válido para cada categoria não reconhecida.")

            if "choropleth_country_map" not in st.session_state:
                st.session_state["choropleth_country_map"] = {}

            if map_key_base not in st.session_state["choropleth_country_map"]:
                st.session_state["choropleth_country_map"][map_key_base] = {}
            manual_map = st.session_state["choropleth_country_map"][map_key_base]

            options = ["(não mapear)"] + known_countries_list
            unresolved = sorted(not_found["categoria_norm"].unique().tolist())

            with st.expander("Mapear categorias não reconhecidas"):
                for raw_cat in unresolved:
                    # Sugestão automática por similaridade para acelerar curadoria humana.
                    suggestion = (difflib.get_close_matches(raw_cat, known_countries_list, n=1) or ["(não mapear)"])[0]
                    current_value = manual_map.get(raw_cat, suggestion)
                    if current_value not in options:
                        current_value = "(não mapear)"
                    selected_country = st.selectbox(
                        f"{raw_cat} →",
                        options=options,
                        index=options.index(current_value),
                        key=f"map_country::{map_key_base}::{raw_cat}",
                    )
                    if selected_country == "(não mapear)":
                        manual_map.pop(raw_cat, None)
                    else:
                        manual_map[raw_cat] = selected_country

            if manual_map:
                # Reaplica resolução após ajustes manuais para atualizar mapa e cobertura.
                full_counts["pais_plot"] = full_counts.apply(
                    lambda row: manual_map.get(row["categoria_norm"], row["pais_plot"]),
                    axis=1,
                )
                resolved_pairs = full_counts["pais_plot"].apply(resolve_country_to_iso3)
                full_counts["map_location"] = resolved_pairs.apply(lambda x: x[0])
                full_counts["pais_resolvido"] = resolved_pairs.apply(lambda x: x[1])

            # Recalcula o que ainda ficou fora após mapeamento manual.
            not_found_after = full_counts[full_counts["map_location"].isna()].copy()
            if not not_found_after.empty:
                st.caption("Ainda existem categorias sem correspondência. Ajuste no seletor acima se quiser mapear todas.")

        # Indicador explícito de qualidade do mapeamento no dataset atual.
        matched = full_counts["map_location"].notna().sum()
        total = len(full_counts)
        coverage = (matched / total * 100) if total else 0
        if has_pycountry:
            st.caption(f"Cobertura final de países reconhecidos: {matched}/{total} categorias ({coverage:.1f}%).")
        else:
            st.caption("Validação automática de países limitada neste ambiente (pycountry não disponível).")

        # Agrega por localização final para evitar múltiplas linhas no mesmo país
        # (ex.: "Inglaterra" + "United Kingdom") sobrescrevendo a cor no mapa.
        map_df = (
            full_counts.dropna(subset=["map_location"])
            .groupby("map_location", as_index=False)
            .agg(
                frequencia=("frequencia", "sum"),
                pais_resolvido=("pais_resolvido", "first"),
                categorias_origem=("categoria", lambda s: " | ".join(sorted(set(s.astype(str))))),
            )
        )

        # Controle de leitura do mapa para reduzir efeito de outliers (ex.: 2000 vs 100-200).
        density_mode = st.radio(
            "Escala de densidade do mapa:",
            [
                "Contínua (linear)",
                "Contínua (log10)",
                "Faixas automáticas (quantis)",
                "Faixas manuais",
            ],
            horizontal=True,
            key=f"map_density_mode::{map_key_base}",
        )

        map_location_mode = "ISO-3" if has_pycountry else "country names"
        map_labels = {
            "frequencia": "Frequency",
            "pais_resolvido": "Country",
            "categorias_origem": "Categorias de origem",
            "faixa_densidade": "Faixa de densidade",
        }

        if density_mode == "Contínua (linear)":
            fig = px.choropleth(
                map_df,
                locations="map_location",
                locationmode=map_location_mode,
                color="frequencia",
                hover_name="pais_resolvido",
                hover_data={"categorias_origem": True, "map_location": True},
                color_continuous_scale="Blues",
                labels=map_labels,
                title=f"Distribuição por país: {selected_col}",
            )
            fig.update_traces(colorbar_title="Frequency")
        elif density_mode == "Contínua (log10)":
            # Compressão logarítmica para reduzir dominância visual de países com valores muito altos.
            import math
            map_df_log = map_df.copy()
            map_df_log["frequencia_log10"] = map_df_log["frequencia"].apply(lambda v: math.log10(v + 1))

            fig = px.choropleth(
                map_df_log,
                locations="map_location",
                locationmode=map_location_mode,
                color="frequencia_log10",
                hover_name="pais_resolvido",
                hover_data={"frequencia": True, "categorias_origem": True, "map_location": True},
                color_continuous_scale="Blues",
                labels={**map_labels, "frequencia_log10": "log10(Frequency + 1)"},
                title=f"Distribuição por país: {selected_col}",
            )
            fig.update_traces(colorbar_title="log10(Frequency + 1)")
        else:
            # Modos por faixas: gera categorias e aplica cor discreta por intervalo.
            map_df_binned = map_df.copy()

            if density_mode == "Faixas automáticas (quantis)":
                n_bins_requested = st.slider(
                    "Número de faixas:",
                    min_value=3,
                    max_value=7,
                    value=5,
                    key=f"map_density_bins::{map_key_base}",
                )
                n_unique = int(map_df_binned["frequencia"].nunique())
                n_bins_effective = max(1, min(n_bins_requested, n_unique))

                if n_bins_effective == 1:
                    map_df_binned["faixa_densidade"] = "Faixa única"
                    category_order = ["Faixa única"]
                else:
                    faixa_cat = pd.qcut(
                        map_df_binned["frequencia"],
                        q=n_bins_effective,
                        duplicates="drop",
                    )
                    category_order = [str(c) for c in faixa_cat.cat.categories]
                    map_df_binned["faixa_densidade"] = faixa_cat.astype(str)
            else:
                min_freq = int(map_df_binned["frequencia"].min())
                max_freq = int(map_df_binned["frequencia"].max())
                default_cut_values = sorted(
                    {
                        int(map_df_binned["frequencia"].quantile(0.25)),
                        int(map_df_binned["frequencia"].quantile(0.50)),
                        int(map_df_binned["frequencia"].quantile(0.75)),
                    }
                )
                default_cut_values = [v for v in default_cut_values if min_freq < v < max_freq]
                default_cut_text = ", ".join(str(v) for v in default_cut_values)

                raw_cut_text = st.text_input(
                    "Cortes (separados por vírgula). Ex.: 100, 300, 800",
                    value=default_cut_text,
                    key=f"map_density_manual_cuts::{map_key_base}",
                )

                parsed_values = []
                parse_error = False
                for piece in raw_cut_text.replace(";", ",").split(","):
                    token = piece.strip()
                    if not token:
                        continue
                    try:
                        parsed_values.append(float(token))
                    except ValueError:
                        parse_error = True
                        break

                if parse_error:
                    st.warning("Cortes manuais inválidos. Use apenas números separados por vírgula.")
                    map_df_binned["faixa_densidade"] = "Faixa única"
                    category_order = ["Faixa única"]
                else:
                    cut_values = sorted(set(v for v in parsed_values if min_freq < v < max_freq))
                    bins = [float("-inf")] + cut_values + [float("inf")]
                    faixa_cat = pd.cut(
                        map_df_binned["frequencia"],
                        bins=bins,
                        include_lowest=True,
                    )
                    category_order = [str(c) for c in faixa_cat.cat.categories]
                    map_df_binned["faixa_densidade"] = faixa_cat.astype(str)

            if len(category_order) <= 1:
                density_palette = [px.colors.sequential.Blues[-1]]
            else:
                # Evita tons muito claros no início para não "sumir" no fundo escuro.
                density_palette = px.colors.sample_colorscale(
                    "Blues",
                    [0.20 + 0.75 * (i / (len(category_order) - 1)) for i in range(len(category_order))],
                )
            density_color_map = dict(zip(category_order, density_palette))
            map_df_binned["faixa_densidade"] = pd.Categorical(
                map_df_binned["faixa_densidade"],
                categories=category_order,
                ordered=True,
            )

            fig = px.choropleth(
                map_df_binned,
                locations="map_location",
                locationmode=map_location_mode,
                color="faixa_densidade",
                hover_name="pais_resolvido",
                hover_data={"frequencia": True, "categorias_origem": True, "map_location": True},
                color_discrete_map=density_color_map,
                labels=map_labels,
                title=f"Distribuição por país: {selected_col}",
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FFFFFF"),
        )
        fig.update_geos(
            showcoastlines=True,
            coastlinecolor="#AAAAAA",
            showframe=False,
            showland=True,
            landcolor="#1E1E1E",
            bgcolor="#0E1117",
        )
        # Mantém a tabela sincronizada com o que foi usado no mapa
        freq_df = full_counts[["categoria", "frequencia"]].copy()
        freq_df["percentual"] = (freq_df["frequencia"] / freq_df["frequencia"].sum() * 100).round(2)
    # Cópia base para exportação (evita variação de estilo após renderização no Streamlit).
    fig_export_base = pio.from_json(fig.to_json())
    st.plotly_chart(fig, use_container_width=True)

    # Download do gráfico em tema escuro e claro (HTML interativo)
    fig_dark = pio.from_json(fig_export_base.to_json())
    fig_dark.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if "geo" in fig_dark.layout and fig_dark.layout["geo"] is not None:
        fig_dark.update_geos(bgcolor="#0E1117")
    html_dark = fig_dark.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    fig_light = pio.from_json(fig_export_base.to_json())
    fig_light.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        coloraxis_colorbar=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        legend=dict(font=dict(color="black")),
    )
    if "geo" in fig_light.layout and fig_light.layout["geo"] is not None:
        fig_light.update_geos(
            bgcolor="white",
            showland=True,
            landcolor="black",
            showocean=False,
            coastlinecolor="white",
            countrycolor="white",
            showcoastlines=True,
            showframe=False,
        )
    html_light = fig_light.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download (tema escuro)",
            data=html_dark,
            file_name=f"{selected_col}_categorias_dark.html",
            mime="text/html",
            use_container_width=True,
            key=f"download_cat_dark_{selected_col}_{chart_type}",
        )
    with col2:
        st.download_button(
            label="📥 Download (tema claro)",
            data=html_light,
            file_name=f"{selected_col}_categorias_light.html",
            mime="text/html",
            use_container_width=True,
            key=f"download_cat_light_{selected_col}_{chart_type}",
        )

    st.write("#### Tabela de frequências")
    st.dataframe(freq_df, use_container_width=True, hide_index=True)

# PAGE 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

# Título e instruções iniciais
st.title("Estatísticas Descritivas")

st.caption("""
A seção **Estatísticas Descritivas** fornece uma análise detalhada da distribuição de variáveis numéricas, incluindo medidas de **tendência central** —média, moda e mediana—, medidas de **dispersão** e **froma** —desvio padrão, amplitude, assimetria e curtose. 
Também permite gerar gráficos interativos —histograma, boxplot e curvas de densidade. 
""")

# Verify dataframe
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Nenhum dataframe carregado.")
    st.stop()

# Seleção do dataframe para visualização
df_names = list(st.session_state.dataframes.keys())

if not df_names:
    st.warning("Nenhum dataframe disponível.")
    st.stop()

selected_df_name = st.session_state.get("selected_df_name")

if selected_df_name not in df_names:
    selected_df_name = df_names[0]


selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names, index=df_names.index(selected_df_name))
df = st.session_state.dataframes[selected_df_name]

st.divider()

# BODY ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# Controle do número de linhas com incremento nativo
st.write("### Descrição por coluna")
num_rows = st.number_input(
    "Número de linhas para inspeção visual:",
    min_value=5,
    max_value=100,
    value=5,
    step=5,
    format="%d"
)

# Visualização do dataframe selecionado
st.write(f"Visualizando as primeiras {num_rows} linhas de **{selected_df_name}**:")
st.dataframe(df.head(num_rows), use_container_width=True)

describe_numeric_column(df, selected_df_name)

st.divider()
describe_categorical_column(df, selected_df_name)
