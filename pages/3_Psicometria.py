# ───────────────────────────────────────────────────────────
# REQUIRED IMPORTS
# ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.decomposition import PCA
from utils.design import load_css
import matplotlib.pyplot as plt




def cronbach_alpha(df):
    """Calcula o alfa de Cronbach para um subconjunto de colunas numéricas."""
    df_corr = df.dropna().astype(float)
    k = df_corr.shape[1]
    variancias_itens = df_corr.var(axis=0, ddof=1)
    variancia_total = df_corr.sum(axis=1).var(ddof=1)
    if variancia_total == 0:
        return 0
    alpha = (k / (k - 1)) * (1 - (variancias_itens.sum() / variancia_total))
    return round(alpha, 4)

def render_psychometric_properties(df: pd.DataFrame, escalas_dict: dict):
    """Renderiza as propriedades psicométricas de uma escala ou fator selecionado."""
    # Cores padrão (tema escuro)
    dark_bg = "#0E1117"
    white = "#FFFFFF"
    purple = "#7159c1"
    st.subheader("Propriedades Psicométricas")

    if not escalas_dict:
        st.info("Crie uma escala primeiro para acessar esta análise.")
        return

    escala_nome = st.selectbox("Selecione a escala:", list(escalas_dict.keys()), key="escala_psico_select")
    escala_data = escalas_dict[escala_nome]

    fatores = escala_data.get("fatores", {})
    opcoes_nivel = ["Escala Total"] + list(fatores.keys())
    alvo = st.radio("Escolha o nível de análise:", opcoes_nivel, key="nivel_analise_radio")

    cols = escala_data["itens"] if alvo == "Escala Total" else fatores[alvo]["itens"]
    df_target = df[cols].dropna()

    if df_target.shape[0] < 10:
        st.warning("Número de observações insuficiente para análise psicométrica robusta.")
        return

    # KMO e Bartlett
    st.subheader("Adequação da Amostra")
    kmo_all, kmo_model = calculate_kmo(df_target)
    chi_sq, p_value = calculate_bartlett_sphericity(df_target)
    st.write(f"KMO: **{round(kmo_model, 4)}**")
    st.write(f"Bartlett: χ² = `{chi_sq:.2f}`, valor-p = `{p_value:.4g}`")

    if kmo_model >= 0.6 and p_value < 0.05:
        st.success("Amostra adequada para extração fatorial.")
    else:
        st.warning("Amostra pode não ser adequada para extração fatorial.")

    # Alfa de Cronbach
    st.subheader("Consistência Interna")
    alpha = cronbach_alpha(df_target)
    st.write(f"Alfa de Cronbach: **`{alpha}`**")

    # Escolha de método
    st.subheader("Extração Fatorial")
    metodo = st.selectbox(
        "Tipo de análise:",
        ["Nenhuma", "Análise Fatorial Exploratória (EFA)", "Componentes Principais (PCA)", "Fatorial Confirmatória (em breve)"],
        key="metodo_analise_select"
    )

    if metodo == "Análise Fatorial Exploratória (EFA)":
        fa = FactorAnalyzer(rotation='varimax')
        fa.fit(df_target)
        # Scree Plot com estilo escuro
        eigenvalues, _ = fa.get_eigenvalues()
        st.caption("""
        A Análise Fatorial Exploratória (EFA) é uma técnica estatística multivariada que busca identificar estruturas latentes (fatores) a partir de padrões de correlação entre variáveis observadas. 
        Ela é amplamente utilizada em psicometria para investigar se um conjunto de itens pode ser agrupado em dimensões subjacentes coerentes, como traços de personalidade ou habilidades cognitivas. 
        A EFA assume que os fatores não são observáveis diretamente, mas influenciam as respostas dos participantes aos itens medidos.
        """)
        st.markdown("##### 🔍 Escolha o número de fatores a extrair")
        st.caption("Visualize os autovalores (eigenvalues) e selecione abaixo a quantidade de fatores a manter.")

        dark_bg = "#0E1117"
        white = "#FFFFFF"
        purple = "#7159c1"

        fig, ax = plt.subplots(facecolor=dark_bg)
        ax.set_facecolor(dark_bg)

        ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o", linestyle="-", color=purple)
        ax.set_title("Scree Plot (Eigenvalues)", color=white)
        ax.set_xlabel("Fator", color=white)
        ax.set_ylabel("Autovalor", color=white)
        ax.axhline(1, color='red', linestyle='--', linewidth=1.5)
        ax.tick_params(colors=white)
        for spine in ax.spines.values():
            spine.set_edgecolor(white)

        plt.tight_layout()
        st.pyplot(fig)

        # Slider e matriz
        n_fatores = st.slider("Número de fatores", min_value=1, max_value=len(eigenvalues), value=1, step=1, key="slider_n_fatores")

        fa_n = FactorAnalyzer(n_factors=n_fatores, rotation='varimax')
        fa_n.fit(df_target)

        st.dataframe(
            pd.DataFrame(
                fa_n.loadings_,
                index=cols,
                columns=[f"Fator {i+1}" for i in range(n_fatores)]
            )
        )

    # === PCA ===
    elif metodo == "Componentes Principais (PCA)":
        pca = PCA()
        explained = pca.fit(df_target).explained_variance_ratio_

        st.caption(
            "A PCA reduz dimensionalidade transformando variáveis correlacionadas em "
            "componentes ortogonais, ordenados por variância explicada."
        )
        st.markdown("##### 🔍 Escolha o número de componentes principais")
        st.caption("Visualize a variância explicada e selecione quantos componentes manter.")

        fig_pca, ax_pca = plt.subplots(facecolor=dark_bg)
        ax_pca.set_facecolor(dark_bg)
        ax_pca.plot(range(1, len(explained) + 1), explained,
                    marker="o", linestyle="-", color=purple)
        ax_pca.set_title("Scree Plot (PCA)", color=white)
        ax_pca.set_xlabel("Componente", color=white)
        ax_pca.set_ylabel("Variância Explicada", color=white)
        ax_pca.tick_params(colors=white)
        for spine in ax_pca.spines.values():
            spine.set_edgecolor(white)
        plt.tight_layout()
        st.pyplot(fig_pca)

        n_components = st.slider("Número de componentes", 1, len(explained), 1, key="slider_n_pcs")
        pca_reduzido = PCA(n_components=n_components)
        pca_reduzido.fit(df_target)
        comps = pd.DataFrame(pca_reduzido.components_.T, index=cols,
                             columns=[f"PC{i+1}" for i in range(n_components)])
        st.dataframe(comps)



    elif metodo == "Fatorial Confirmatória (em breve)":
        st.info("Este módulo será integrado futuramente com modelagem SEM.")

# ───────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ───────────────────────────────────────────────────────────
load_css()
st.title("Análise Fatorial")
st.subheader("Construção e validação de escalas.")
st.divider()
# ───────────────────────────────────────────────────────────
# VERIFICAÇÃO DE DATAFRAMES
# ───────────────────────────────────────────────────────────
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Você precisa carregar um arquivo .csv na Página Inicial.")
    st.stop()

df_names = list(st.session_state.dataframes.keys())
selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names)
df = st.session_state.dataframes[selected_df_name]

num_cols = df.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.warning("O dataframe selecionado não contém colunas numéricas.")
    st.stop()

# ───────────────────────────────────────────────────────────
# INICIALIZAÇÃO DAS ESCALAS POR DATAFRAME
# ───────────────────────────────────────────────────────────
if "escalas" not in st.session_state:
    st.session_state["escalas"] = {}

if selected_df_name not in st.session_state["escalas"]:
    st.session_state["escalas"][selected_df_name] = {}

escalas_dict = st.session_state["escalas"][selected_df_name]


with st.expander("Adicionar +1 a itens (base zero)", expanded=False):
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Selecione os itens cujas respostas estão de 0 até n e que precisam de ajuste para começar em 1.")
    
    itens_ajuste = st.multiselect(
        "Itens com base 0 que precisam de +1:",
        options=df.columns.tolist(),  # ou selected_cols, ou todos_itens, dependendo do escopo
        key="itens_ajuste_base_zero"
    )

    aplicar_ajuste = st.button("Aplicar +1 nos itens selecionados", key="btn_aplicar_ajuste")

    if aplicar_ajuste:
        for item in itens_ajuste:
            df[item] = df[item] + 1
        st.success(f"Ajuste aplicado: {len(itens_ajuste)} item(ns) foram incrementados em +1.")
        st.session_state.dataframes[selected_df_name] = df


with st.expander("Reverter itens (inversão de escala)", expanded=False):
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Selecione os itens cujos valores devem ser revertidos e defina a escala máxima de cada um.")
    
    itens_reversos = st.multiselect(
        "Itens a serem revertidos:",
        options=df.columns.tolist(),  # ou selected_cols / todos_itens
        key="itens_para_reversao"
    )

    maximos = {}
    if itens_reversos:
        st.markdown("#### Defina o valor máximo da escala para cada item:")
        for item in itens_reversos:
            maximos[item] = st.number_input(
                f"Máximo da escala para o item `{item}`:",
                min_value=2,
                max_value=10,
                value=4,
                step=1,
                key=f"maximo_{item}"
            )

        aplicar_reversao = st.button("Aplicar reversão", key="btn_aplicar_reversao")

        if aplicar_reversao:
            for item, max_val in maximos.items():
                df[item] = (max_val + 1) - df[item]
            st.success(f"Reversão aplicada: {len(maximos)} item(ns) foram invertidos.")
            st.session_state.dataframes[selected_df_name] = df


# ───────────────────────────────────────────────────────────
# SEÇÃO 1 — CRIAR NOVA ESCALA (agora com st.form)
# ───────────────────────────────────────────────────────────
st.subheader("Criar uma nova escala")

with st.form("form_criar_escala"):
    selected_cols = st.multiselect("Itens para compor a escala", num_cols, key="escala_itens")
    scale_name = st.text_input("Nome da nova escala (ex: BIS_Total)", key="escala_nome")
    
    placeholder_scales = st.empty()
    
    submitted = st.form_submit_button("Criar escala", use_container_width=True)

    if submitted:
        if not selected_cols:
            st.error("Selecione pelo menos um item.")
        elif not scale_name:
            st.error("Defina um nome para a escala.")
        elif scale_name in df.columns:
            st.error("Já existe uma coluna com esse nome.")
        else:
            df[scale_name] = df[selected_cols].sum(axis=1)
            escalas_dict[scale_name] = {
                "itens": selected_cols,
                "valores": df[scale_name].tolist(),
                "fatores": {}
            }
            st.session_state.dataframes[selected_df_name] = df
            placeholder_scales.success(f"Escala '{scale_name}' criada com sucesso.")


# ───────────────────────────────────────────────────────────
# SEÇÃO 3 — DEFINIR FATORES EM UMA ESCALA
# ───────────────────────────────────────────────────────────
if escalas_dict:
    st.subheader("Definir fatores")
    escala_selecionada = st.selectbox("Escolha a escala-alvo", list(escalas_dict.keys()), key="escala_fator")
    escala_data = escalas_dict[escala_selecionada]
    todos_itens = escala_data["itens"]
    fatores_atuais = escala_data.get("fatores", {})

    with st.form("form_fator"):
        nome_fator = st.text_input("Nome do novo fator")
        itens_fator = st.multiselect("Itens que compõem este fator", todos_itens)

        placehold_add_factor = st.empty()

        criar_fator = st.form_submit_button("Adicionar fator", use_container_width=True)
    
    if criar_fator:
        if not nome_fator or not itens_fator:
            st.error("Preencha o nome e selecione pelo menos um item.")
        else:
            valores_fator = df[itens_fator].sum(axis=1).tolist()
            escala_data["fatores"][nome_fator] = {
                "itens": itens_fator,
                "valores": valores_fator
            }
            # Adiciona a nova coluna no dataframe
            nome_coluna = f"{escala_selecionada}_{nome_fator}"
            df[nome_coluna] = valores_fator
            st.session_state.dataframes[selected_df_name] = df
            placehold_add_factor.success(f"Fator '{nome_fator}' adicionado à escala '{escala_selecionada}'.")


# ───────────────────────────────────────────────────────────
# SEÇÃO 4 — VISUALIZAR ESCALAS E FATORES
# ───────────────────────────────────────────────────────────
for nome, dados in escalas_dict.items():
    st.markdown(f"### 🧮 Escala: **{nome}**")
    st.markdown(f"Composta por: `{', '.join(dados['itens'])}`")
    st.line_chart(dados["valores"])

    fatores = dados.get("fatores", {})

    if fatores:
        st.markdown("#### 📚 Fatores:")
        for fator_nome, fator_info in list(fatores.items()):
            with st.expander(f"{fator_nome}", expanded=False):
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Itens:** `{', '.join(fator_info['itens'])}`")
                st.line_chart(fator_info["valores"])

                delete = st.button(
                    f"🗑️ Deletar fator",
                    key=f"delete_fator_{nome}_{fator_nome}",
                    use_container_width=True
                )

                if delete:
                    del fatores[fator_nome]
                    col_to_drop = f"{nome}_{fator_nome}"
                    if col_to_drop in df.columns:
                        df.drop(columns=[col_to_drop], inplace=True)
                    st.session_state.dataframes[selected_df_name] = df
                    st.rerun()


render_psychometric_properties(df, escalas_dict)