import numpy as np
import pandas as pd
import streamlit as st

from factor_analyzer import FactorAnalyzer
from scipy.optimize import linear_sum_assignment


def _tucker_congruence(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tucker congruence coefficient between two loading vectors."""
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _align_loadings_to_reference(
    reference_loadings: pd.DataFrame,
    target_loadings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align target factors to a reference loading matrix using:
    1) optimal permutation by max absolute congruence
    2) sign correction per matched factor
    """
    n_factors = reference_loadings.shape[1]
    congruence_matrix = np.zeros((n_factors, n_factors), dtype=float)

    for ref_idx in range(n_factors):
        ref_vec = reference_loadings.iloc[:, ref_idx].to_numpy(dtype=float)
        for tgt_idx in range(n_factors):
            tgt_vec = target_loadings.iloc[:, tgt_idx].to_numpy(dtype=float)
            congruence_matrix[ref_idx, tgt_idx] = _tucker_congruence(ref_vec, tgt_vec)

    row_ind, col_ind = linear_sum_assignment(-np.abs(congruence_matrix))

    aligned = pd.DataFrame(index=target_loadings.index)
    alignment_rows = []

    for ref_idx, tgt_idx in sorted(zip(row_ind, col_ind), key=lambda pair: pair[0]):
        congr = float(congruence_matrix[ref_idx, tgt_idx])
        sign = -1.0 if congr < 0 else 1.0
        factor_label = f"Fator {ref_idx + 1}"

        aligned[factor_label] = target_loadings.iloc[:, tgt_idx] * sign
        alignment_rows.append(
            {
                "fator_alinhado": factor_label,
                "fator_original": f"Fator {tgt_idx + 1}",
                "congruencia": congr,
                "sinal_invertido": sign < 0,
            }
        )

    return aligned, pd.DataFrame(alignment_rows)


def _interpret_factor_column(
    factor_loadings: pd.Series,
    top_n: int = 3,
    min_abs_loading: float = 0.30,
) -> str:
    """Compact text summary with the most salient item loadings."""
    ordered = factor_loadings.reindex(
        factor_loadings.abs().sort_values(ascending=False).index
    )
    salient = ordered[ordered.abs() >= float(min_abs_loading)]
    chosen = salient if not salient.empty else ordered
    chosen = chosen.iloc[:top_n]

    parts = [f"{item} ({value:+.2f})" for item, value in chosen.items()]
    return "; ".join(parts)


def render_efa_group_alignment(
    df_base: pd.DataFrame,
    cols: list[str],
    aplicar_filtro: bool,
    coluna_filtro: str | None,
    valores_selecionados: list,
    n_fatores: int,
    rotacao_ajustada: str | None,
    metodo_extracao: str,
    loadings_referencia: pd.DataFrame,
) -> None:
    """Render aligned EFA comparison across selected groups (e.g., countries)."""
    if not (aplicar_filtro and coluna_filtro is not None and valores_selecionados):
        return

    grupos_comparacao = list(dict.fromkeys(valores_selecionados))
    if len(grupos_comparacao) < 2:
        return

    st.markdown("##### Interpretacao de fatores alinhada por grupo")
    st.caption(
        "Os fatores de cada grupo foram reordenados e sinalizados para "
        "maximizar a congruencia com a solucao de referencia da amostra filtrada."
    )

    min_n_grupo = max(10, n_fatores + 2)
    tabela_interpretacao = []
    tabela_alinhamento = []
    grupos_ignorados = []
    loadings_por_grupo = {}

    for grupo in grupos_comparacao:
        df_grupo = (
            df_base.loc[df_base[coluna_filtro] == grupo, cols]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        n_grupo = int(df_grupo.shape[0])

        if n_grupo < min_n_grupo:
            grupos_ignorados.append(f"{grupo} (N={n_grupo})")
            continue

        try:
            fa_grupo = FactorAnalyzer(
                n_factors=n_fatores,
                rotation=rotacao_ajustada,
                method=metodo_extracao,
            )
            fa_grupo.fit(df_grupo)
        except Exception:
            grupos_ignorados.append(f"{grupo} (falha na estimacao)")
            continue

        loadings_grupo = pd.DataFrame(
            fa_grupo.loadings_,
            index=cols,
            columns=[f"Fator {i + 1}" for i in range(n_fatores)],
        )
        loadings_alinhados, info_alinhamento = _align_loadings_to_reference(
            loadings_referencia,
            loadings_grupo,
        )
        loadings_por_grupo[str(grupo)] = loadings_alinhados.copy()

        for fator_idx in range(n_fatores):
            fator_label = f"Fator {fator_idx + 1}"
            linha_info = info_alinhamento[
                info_alinhamento["fator_alinhado"] == fator_label
            ]
            congruencia = float(linha_info["congruencia"].iloc[0]) if not linha_info.empty else np.nan
            origem = linha_info["fator_original"].iloc[0] if not linha_info.empty else "-"
            interpretacao = _interpret_factor_column(loadings_alinhados[fator_label])

            tabela_interpretacao.append(
                {
                    "Grupo": str(grupo),
                    "N": n_grupo,
                    "Fator alinhado": fator_label,
                    "Fator original (grupo)": origem,
                    "Congruencia (Tucker)": round(congruencia, 2),
                    "Interpretacao (itens/cargas)": interpretacao,
                }
            )

        info_exibicao = info_alinhamento.copy()
        info_exibicao.insert(0, "Grupo", str(grupo))
        info_exibicao.insert(1, "N", n_grupo)
        info_exibicao["congruencia"] = info_exibicao["congruencia"].round(2)
        tabela_alinhamento.append(info_exibicao)

    if tabela_interpretacao:
        st.dataframe(
            pd.DataFrame(tabela_interpretacao).sort_values(
                by=["Fator alinhado", "Grupo"],
                kind="stable",
            )
        )

        st.markdown("##### Cargas alinhadas por item e grupo")
        st.caption(
            f"Cada celula mostra `Reference(valor)` + todos os valores de `{coluna_filtro}` "
            "no formato `GRUPO(valor)`."
        )
        fatores_labels = [f"Fator {i + 1}" for i in range(n_fatores)]
        tabela_consolidada_rows = []

        for item in cols:
            row = {"Item": item}
            for fator_label in fatores_labels:
                tokens = [f"Reference({loadings_referencia.loc[item, fator_label]:.2f})"]
                for grupo_nome, matriz_alinhada in loadings_por_grupo.items():
                    tokens.append(f"{grupo_nome}({matriz_alinhada.loc[item, fator_label]:.2f})")
                row[fator_label] = "  ".join(tokens)
            tabela_consolidada_rows.append(row)

        st.dataframe(pd.DataFrame(tabela_consolidada_rows))

        with st.expander("Tabela completa: item x grupo x fator", expanded=False):
            tabela_longa = []
            for grupo_nome, matriz_alinhada in loadings_por_grupo.items():
                bloco = matriz_alinhada.copy()
                bloco["Item"] = bloco.index
                bloco["Grupo"] = grupo_nome
                tabela_longa.append(
                    bloco.melt(
                        id_vars=["Grupo", "Item"],
                        var_name="Fator alinhado",
                        value_name="Carga",
                    )
                )
            if tabela_longa:
                tabela_longa_df = pd.concat(tabela_longa, ignore_index=True).sort_values(
                    by=["Fator alinhado", "Item", "Grupo"],
                    kind="stable",
                )
                tabela_longa_df["Carga"] = tabela_longa_df["Carga"].round(2)
                st.dataframe(tabela_longa_df)

        with st.expander("Detalhes do alinhamento (permutacao e sinal)", expanded=False):
            st.dataframe(
                pd.concat(tabela_alinhamento, ignore_index=True).rename(
                    columns={
                        "fator_alinhado": "Fator alinhado",
                        "fator_original": "Fator original",
                        "congruencia": "Congruencia (Tucker)",
                        "sinal_invertido": "Sinal invertido",
                    }
                )
            )
    else:
        st.info(
            "Nenhum grupo teve N suficiente para comparacao alinhada "
            f"(minimo atual: {min_n_grupo})."
        )

    if grupos_ignorados:
        st.caption(
            "Grupos ignorados na comparacao alinhada: " + ", ".join(grupos_ignorados)
        )
