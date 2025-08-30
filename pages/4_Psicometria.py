# REQUIRED IMPORTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit         as st
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from factor_analyzer       import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from utils.design          import load_css

# CUSTOM FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

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

def rescale_items(df: pd.DataFrame, selected_df_name: str):
    """
    <docstrings>
    Permite reescalar itens entre base 0 e base 1:
      • Base 0 → Base 1 (incrementa +1 em cada valor).
      • Base 1 → Base 0 (decrementa -1 em cada valor).

    Args:
        df (pd.DataFrame): DataFrame original.
        selected_df_name (str): Nome do dataframe salvo no session_state.

    Calls:
        st.expander(): Cria seção expansível | instanciado por st.
        st.radio(): Escolha da direção de ajuste | instanciado por st.
        st.multiselect(): Seleção de itens a ajustar | instanciado por st.
        st.button(): Botão para aplicar ajuste | instanciado por st.
        df.__setitem__(): Atualiza valores da coluna | método de DataFrame.
        st.session_state.dataframes.__setitem__(): Atualiza dataframe global | instanciado por session_state.
        st.session_state.__setitem__(): Armazena CSV atualizado | instanciado por session_state.

    Returns:
        None.
    """
    import streamlit as st

    with st.expander("Reescalar itens (base 0 ↔ base 1)", expanded=False):
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "Este módulo permite corrigir a base de itens tipo Likert:\n\n"
            "- **Base 0 → Base 1**: incrementa todos os valores em +1 (ex.: 0–3 → 1–4).\n"
            "- **Base 1 → Base 0**: decrementa todos os valores em -1 (ex.: 1–4 → 0–3).\n\n"
            "Assim, preserva-se a ordem ordinal enquanto se ajusta a codificação da escala."
        )

        direcao = st.radio(
            "Escolha a direção do ajuste:",
            options=["Base 0 → Base 1", "Base 1 → Base 0"],
            key="direcao_reescala"
        )

        itens_ajuste = st.multiselect(
            "Itens a serem reescalados:",
            options=df.columns.tolist(),
            key="itens_ajuste_reescala"
        )

        placeholder_ajuste = st.empty()
        aplicar_ajuste = st.button("Aplicar reescala", use_container_width=True, key="btn_aplicar_reescala")

        if aplicar_ajuste:
            if direcao == "Base 0 → Base 1":
                for item in itens_ajuste:
                    df[item] = df[item] + 1
                placeholder_ajuste.success(f"Ajuste aplicado: {len(itens_ajuste)} item(ns) incrementados em +1.")
            elif direcao == "Base 1 → Base 0":
                for item in itens_ajuste:
                    df[item] = df[item] - 1
                placeholder_ajuste.success(f"Ajuste aplicado: {len(itens_ajuste)} item(ns) decrementados em -1.")

            st.session_state.dataframes[selected_df_name] = df
            st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")


# PAGE 4 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_css()

st.title("Psicometria")

st.caption("""
A seção **Psicometria** oferece ferramentas para criação, escoragem e avaliação estatística de escalas psicológicas. 
É possível construir novas escalas a partir de itens selecionados, definir fatores internos e calcular indicadores fundamentais, 
como o **alfa de Cronbach** (consistência interna), o **KMO** e o **teste de esfericidade de Bartlett** (adequação da amostra). 
Também estão disponíveis métodos de análise fatorial como a **EFA** (Exploratória), **PCA** (Análise de Componentes Principais) 
e, em breve, **CFA** (Confirmatória). Ideal para pesquisadores que desejam validar construtos latentes de forma empírica.
""")

# Verify dataframe
if "dataframes" not in st.session_state or not st.session_state.dataframes:
    st.warning("Nenhum dataframe carreagdo.")
    st.stop()

df_names = list(st.session_state.dataframes.keys())
selected_df_name = st.selectbox("Selecione o dataframe para análise:", df_names)
df = st.session_state.dataframes[selected_df_name]
st.write(f"**Dimensões:** {df.shape[0]} × {df.shape[1]}")

num_cols = df.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.warning("Este dataframe não possui colunas numéricas.")
    st.stop()

st.divider()

# BODY ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

if "escalas" not in st.session_state:
    st.session_state["escalas"] = {}

if selected_df_name not in st.session_state["escalas"]:
    st.session_state["escalas"][selected_df_name] = {}

escalas_dict = st.session_state["escalas"][selected_df_name]


rescale_items(df, selected_df_name)


with st.expander("Inversão de itens", expanded=False):
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Selecione os itens cujos valores devem ser revertidos; "
        "o sistema detecta automaticamente o valor máximo observado, exibe e permite reverter em lote."
    )
    
    itens_reversos = st.multiselect(
        "Itens a serem revertidos:",
        options=df.columns.tolist(),
        key="itens_para_reversao"
    )

    if itens_reversos:
        # 1) detecta máximo observado em cada coluna
        maximos_detectados = {item: int(df[item].max()) for item in itens_reversos}

        # 2) exibe para o usuário
        st.markdown("#### Máximos detectados para cada item:")
        for item, max_val in maximos_detectados.items():
            st.write(f"- `{item}`: {max_val}")

        placeholder_invert =st.empty()

        # 3) botão único para reverter tudo
        if st.button(
            "Reverter itens selecionados",
            use_container_width=True,
            key="btn_aplicar_reversao"
        ):
            for item, max_val in maximos_detectados.items():
                df[item] = (max_val + 1) - df[item]
            placeholder_invert.success(f"Reversão aplicada para {len(itens_reversos)} item(ns).")
            st.session_state.dataframes[selected_df_name] = df
            st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")

# ───────────────────────────────────────────────────────────
st.subheader("Criar escala a partir de um conjunto de itens")
st.caption("O somatório das novas escalas será mostrado e, opcionalmente, adicionado ao dataframe atual.")

# 1) Escolha de itens e tipo de soma
selected_cols = st.multiselect(
    "Itens para compor a escala",
    num_cols,
    key="escala_itens"
)
sum_type = st.radio(
    "Tipo de somatório:",
    ["Normal", "Binário"],
    horizontal=True,
    key="escala_sum_type"
)

# 2) Threshold (apenas para binário)
threshold = None
if sum_type == "Binário" and selected_cols:
    min_val = int(df[selected_cols].min().min())
    max_val = int(df[selected_cols].max().max())
    st.caption("Cada item com valor maior ou igual ao limiar (threshold) contribui com 1 ponto. Abaixo dele, ponto zero.")
    threshold = st.number_input(
        "Threshold:",
        min_value=min_val,
        max_value=max_val,
        value=(min_val + max_val) // 2,
        step=1,
        format="%d",
        key="escala_threshold"
    )

# 3) Nome da escala
scale_name = st.text_input(
    "Nome da nova escala:",
    key="escala_nome"
)

# 4) Opção de salvar no df
save_to_df = st.checkbox("Salvar a escala no dataframe atual", value=False, key="escala_save")

placeholder_scales = st.empty()

# 5) Criação
if st.button("Criar escala", use_container_width=True):
    # validações
    if not selected_cols:
        st.error("Selecione pelo menos um item.")
    elif not scale_name:
        st.error("Defina um nome para a escala.")
    elif save_to_df and scale_name in df.columns:
        st.error("Já existe uma coluna com esse nome.")
    else:
        # cálculo da série
        if sum_type == "Normal":
            new_series = df[selected_cols].sum(axis=1)
        else:
            new_series = (df[selected_cols] >= threshold).sum(axis=1)

        # mostra na tela
        st.write("#### Primeiro valores da escala criada:")
        st.dataframe(new_series.to_frame(scale_name).head())

        

        # salva se pedido
        if save_to_df:
            df[scale_name] = new_series
            st.session_state.dataframes[selected_df_name] = df
            st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")
            placeholder_scales.success(f"Escala '{scale_name}' adicionada ao dataframe.")
        else:
            placeholder_scales.info(f"Escala '{scale_name}' salva na sessão.")

        # guarda metadados sempre
        escalas_dict[scale_name] = {
            "itens": selected_cols,
            "tipo": sum_type,
            "threshold": threshold,
            "valores": new_series.tolist(),
            "fatores": {}
        }

        st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")

# ───────────────────────────────────────────────────────────
# SEÇÃO 3 — DEFINIR FATORES EM UMA ESCALA
# ───────────────────────────────────────────────────────────
if escalas_dict:
    st.subheader("Definir fatores")
    escala_selecionada = st.selectbox(
        "Escolha a escala-alvo",
        list(escalas_dict.keys()),
        key="escala_fator"
    )
    escala_data = escalas_dict[escala_selecionada]
    todos_itens = escala_data["itens"]
    fatores_atuais = escala_data.get("fatores", {})

    with st.form("form_fator"):
        nome_fator = st.text_input("Nome do novo fator", key="input_fator_nome")
        itens_fator = st.multiselect(
            "Itens que compõem este fator",
            todos_itens,
            key="multiselect_fator_itens"
        )
        save_factor = st.checkbox(
            "Salvar este fator no dataframe atual",
            value=False,
            key="save_fator_checkbox"
        )

        placehold_add_factor = st.empty()
        criar_fator = st.form_submit_button(
            "Adicionar fator",
            use_container_width=True
        )
    
    if criar_fator:
        if not nome_fator or not itens_fator:
            st.error("Preencha o nome do fator e selecione pelo menos um item.")
        else:
            # Calcula os valores do fator
            valores_fator = df[itens_fator].sum(axis=1).tolist()
            
            # Armazena no dicionário interno
            escala_data.setdefault("fatores", {})[nome_fator] = {
                "itens": itens_fator,
                "valores": valores_fator
            }

            # Se o usuário marcou para salvar, cria a coluna no DataFrame
            nome_coluna = f"{escala_selecionada}_{nome_fator}"
            if save_factor:
                df[nome_coluna] = valores_fator
                st.session_state.dataframes[selected_df_name] = df
                st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")
                placehold_add_factor.success(
                    f"Fator '{nome_fator}' adicionado e salvo como coluna '{nome_coluna}'."
                )
            else:
                placehold_add_factor.info(
                    f"Fator '{nome_fator}' criado em memória, mas NÃO salvo no DataFrame."
                )


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

st.divider()

st.session_state["csv_transformado"] = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download (dataframe)",
    data=st.session_state["csv_transformado"],
    file_name=f"{selected_df_name}_curado.csv",
    mime="text/csv",
    use_container_width=True
    )