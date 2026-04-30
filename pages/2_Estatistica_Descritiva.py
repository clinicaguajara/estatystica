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

    # McKinney, 2010
    st.info(
        """**W. McKinney. Data Structures for Statistical Computing in Python (2010).** [doi](https://doi.org/10.25080/Majora-92bf1922-00a)  
    \nMcKinney (2010) argumenta que a integração de pandas com NumPy, SciPy, Matplotlib e outras bibliotecas científicas torna o Python uma opção cada vez mais atraente para análise de dados estatísticos, especialmente em comparação com R. O artigo aponta a evolução futura da biblioteca e seu papel central em um ecossistema de modelagem estatística em Python.
    """,
        icon="📜"
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
        ["Contagem de frequências", "Quadrados proporcionais", "Choropleth (países)"],
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
            title=f"Frequência por categoria: {selected_col}",
        )
        fig.update_layout(showlegend=False, xaxis_title=selected_col, yaxis_title="Frequência")
    elif chart_type == "Quadrados proporcionais":
        fig = px.treemap(
            freq_df,
            path=["categoria"],
            values="frequencia",
            color="frequencia",
            color_continuous_scale="Blues",
            title=f"Quadrados proporcionais por categoria: {selected_col}",
        )
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

        # Camada 3 de robustez: intervenção manual guiada para casos não reconhecidos.
        # O mapeamento fica persistido no session_state por dataframe+coluna.
        not_found = full_counts[full_counts["map_location"].isna()].copy() if has_pycountry else pd.DataFrame()
        if not not_found.empty:
            st.warning("Algumas categorias não foram reconhecidas como países.")
            st.caption("Escolha manualmente um país válido para cada categoria não reconhecida.")

            if "choropleth_country_map" not in st.session_state:
                st.session_state["choropleth_country_map"] = {}

            map_key_base = f"{df_name}::{selected_col}"
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

        fig = px.choropleth(
            full_counts.dropna(subset=["map_location"]),
            locations="map_location",
            # Com pycountry: ISO-3 (mais estável). Sem pycountry: nomes de países (mais sensível a variações).
            locationmode="ISO-3" if has_pycountry else "country names",
            color="frequencia",
            hover_name="categoria",
            color_continuous_scale="Blues_r",
            labels={"frequencia": "Frequency"},
            title=f"Distribuição por país: {selected_col}",
        )
        fig.update_traces(colorbar_title="Frequency")
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
    st.plotly_chart(fig, use_container_width=True)

    # Download do gráfico em tema escuro e claro (HTML interativo)
    fig_dark = pio.from_json(fig.to_json())
    fig_dark.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if "geo" in fig_dark.layout and fig_dark.layout["geo"] is not None:
        fig_dark.update_geos(bgcolor="#0E1117")
    html_dark = fig_dark.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    fig_light = pio.from_json(fig.to_json())
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
