import pandas as pd


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna as colunas numéricas do DataFrame.
    """
    return df.select_dtypes(include="number").columns.tolist()


def prepare_feature_matrix(
    df: pd.DataFrame,
    cols: list[str],
    missing_strategy: str,
    scaler_choice: str,
):
    """
    Prepara matriz de features para métodos não supervisionados.

    Returns:
        X_df (pd.DataFrame): Dados após tratamento de missing (índice preservado).
        X_model (np.ndarray): Matriz pronta para treino (escalada ou não).
        scaler: Objeto de escalonamento usado ou None.
    """
    X_df = df[cols].copy()
    if missing_strategy == "Excluir linhas com NA":
        X_df = X_df.dropna(axis=0)
    else:
        X_df = X_df.fillna(X_df.mean(numeric_only=True))

    if X_df.shape[0] < 2:
        raise ValueError("Dados insuficientes após tratamento de missing.")

    scaler = None
    if scaler_choice != "Nenhum":
        if scaler_choice == "StandardScaler":
            from sklearn.preprocessing import StandardScaler as _Scaler
        elif scaler_choice == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler as _Scaler
        else:
            from sklearn.preprocessing import RobustScaler as _Scaler
        scaler = _Scaler()
        X_model = scaler.fit_transform(X_df.values)
    else:
        X_model = X_df.values

    return X_df, X_model, scaler
