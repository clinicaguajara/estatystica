# pages/7_Machine_Learning.py
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd

from utils.dataframe_state import select_active_dataframe

try:
    from utils.design import load_css  # opcional
except Exception:
    load_css = None


def render_header():
    """
    <docstrings>
    Renderiza o título e a legenda introdutória (leve).
    """
    st.title("Aprendizado de Máquina")
    st.caption(
        "Inteligência Artificial (IA) busca criar sistemas capazes de tomar decisões. "
        "Machine Learning (ML) é a subárea da IA que aprende padrões a partir de dados "
        "para prever ou decidir sem regras explícitas para cada tarefa."
    )


def render_dataframe_summary(df: pd.DataFrame):
    """
    <docstrings>
    Exibe um resumo rápido do DataFrame: dimensões, contagem por tipo de dado e amostra.
    """
    # Dimensões
    st.write(f"**Dimensões:** {df.shape[0]} × {df.shape[1]}")

    # Prévia
    st.dataframe(df.head(5))


def render_area_selector() -> str:
    """
    <docstrings>
    Exibe o seletor da grande área de ML.
    """
    return st.radio(
        "Escolha a grande área:",
        (
            "Aprendizado Supervisionado",
            "Aprendizado Não Supervisionado",
            "Aprendizado Semi-supervisionado",
            "Aprendizado por Reforço",
        ),
        horizontal=True,
        key="ml_area_choice",
    )


def main():
    """
    <docstrings>
    UI básica (título, caption, seletores).
    """
    if load_css:
        try:
            load_css()
        except Exception:
            pass

    render_header()

    # Desenha o seletor de grande área.
    area = render_area_selector()
    df_name, df = select_active_dataframe(
        state_key="selected_df_name",
        label="Selecione o dataframe para análise:",
        widget_key="ml_selected_df",
    )

    if not isinstance(df, pd.DataFrame):
        st.error("O objeto selecionado não é um DataFrame válido.")
        st.stop()

    # Mini resumo encapsulado
    render_dataframe_summary(df)
    st.divider()

    st.session_state["ml_selected_df_name"] = df_name
    st.session_state["ml_selected_area"]   = area
    
    if area == "Aprendizado Não Supervisionado":
        from modules.unsupervised import render_unsupervised  # import local (leve)
        render_unsupervised(df)

    # ⬇ NOVO: Supervisionado (Classificação)
    elif area == "Aprendizado Supervisionado":
        from modules.supervised import render_supervised_classification  # import local
        render_supervised_classification(df)


if __name__ == "__main__":
    main()

