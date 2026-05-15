import streamlit as st


def require_dataframes() -> dict:
    """
    Ensure dataframes exist in session state and return them.
    Stops page execution with a warning if none are available.
    """
    dataframes = st.session_state.get("dataframes", {})
    if not dataframes:
        st.warning("Nenhum dataframe carregado.")
        st.stop()
    return dataframes


def select_active_dataframe(
    *,
    state_key: str = "selected_df_name",
    label: str = "Selecione o dataframe para análise:",
    widget_key: str = "selected_df_name_select",
) -> tuple[str, object]:
    """
    Persist and return the active dataframe selection for the current session.
    """
    dataframes = require_dataframes()
    df_names = list(dataframes.keys())

    current_name = st.session_state.get(state_key)
    if current_name not in df_names:
        current_name = df_names[0]

    selected_name = st.selectbox(
        label,
        df_names,
        index=df_names.index(current_name),
        key=widget_key,
    )
    st.session_state[state_key] = selected_name
    return selected_name, dataframes[selected_name]
