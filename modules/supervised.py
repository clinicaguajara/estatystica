# modules/supervised.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, List, Optional, Dict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _infer_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Inferência simples de colunas numéricas e categóricas."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Cria o ColumnTransformer com imputação e escala/one-hot."""
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def _get_model(name: str, class_weight: bool, C: float = 1.0, n_estimators: int = 300, max_depth: Optional[int] = None):
    """Factory simples dos modelos suportados."""
    weight = "balanced" if class_weight else None

    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=500,
            C=C,
            class_weight=weight,
            n_jobs=None,
            solver="lbfgs",
            multi_class="auto",
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=weight,
            n_jobs=-1,
            random_state=42,
        )
    if name == "SVM (RBF)":
        # probas True p/ ROC
        return SVC(
            kernel="rbf",
            C=C,
            class_weight=weight,
            probability=True,
            random_state=42,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Modelo não suportado: {name}")


def _plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Real")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def _plot_roc(y_true, y_proba, classes):
    """Plota ROC para binário (1 curva) ou multiclass (macro-avg)."""
    n_classes = len(classes)
    if n_classes == 2:
        # Binário: pegue a prob da classe positiva (assumimos a última posição em classes)
        pos_idx = 1 if hasattr(y_proba, "ndim") and y_proba.ndim == 2 else None
        if pos_idx is None:
            st.info("Probabilidades não disponíveis para ROC.")
            return
        y_true_bin = (y_true == classes[1]).astype(int)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, 1])
        auc_val = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "--", alpha=0.6)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC (binário)")
        ax.legend()
        st.pyplot(fig)
    else:
        # Multiclasse One-vs-Rest (macro)
        try:
            from sklearn.metrics import roc_curve, auc
            Y = label_binarize(y_true, classes=classes)
            fprs, tprs, aucs = [], [], []
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
                aucs.append(auc(fpr, tpr))
                fprs.append(fpr); tprs.append(tpr)
            fig, ax = plt.subplots()
            for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
                ax.plot(fpr, tpr, label=f"{classes[i]} (AUC={aucs[i]:.3f})", alpha=0.8)
            ax.plot([0, 1], [0, 1], "--", alpha=0.6)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC One-vs-Rest (multiclass)")
            ax.legend()
            st.pyplot(fig)
        except Exception:
            st.info("Não foi possível calcular ROC multiclasse para este modelo.")


def _safe_predict_proba(clf, X):
    """Tenta obter probabilidades de forma robusta."""
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(X)
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# UI principal
# ──────────────────────────────────────────────────────────────────────────────

def render_supervised_classification(df: pd.DataFrame):
    """
    <docstrings>
    Renderiza um fluxo de classificação supervisionada com:
    - Seleção de alvo (y) e features (X).
    - Split estratificado.
    - Pré-processamento num/cat com imputação, escala e one-hot.
    - Escolha de modelo e hiperparâmetros básicos.
    - Treino, métricas, matriz de confusão, ROC e importâncias (permutation).
    """
    st.subheader("Classificação (Supervisionado)")

    with st.expander("1) Seleção de Variáveis", expanded=True):
        target_col = st.selectbox("Coluna-alvo (y):", df.columns, index=None, placeholder="Selecione…")
        if not target_col:
            st.stop()

        feature_candidates = [c for c in df.columns if c != target_col]
        default_feats = feature_candidates  # padrão: usar todas as demais
        selected_features = st.multiselect(
            "Features (X):",
            feature_candidates,
            default=default_feats
        )

        # Drop linhas com y ausente
        work_df = df[selected_features + [target_col]].copy()
        if work_df[target_col].isna().any():
            st.warning("Foram encontradas linhas com alvo (y) ausente. Elas serão removidas.")
            work_df = work_df[~work_df[target_col].isna()].copy()

        y = work_df[target_col]
        X = work_df[selected_features]

        st.caption(f"Classes em **{target_col}**: {sorted(pd.Series(y).dropna().unique().tolist())}")

    with st.expander("2) Configurações de Treino/Validação", expanded=True):
        test_size = st.slider("Tamanho do conjunto de teste (proporção):", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state:", min_value=0, value=42, step=1)

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        st.write(f"Treino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas")

    with st.expander("3) Pré-processamento", expanded=False):
        num_cols, cat_cols = _infer_cols(X_train)
        st.write("Numéricas:", num_cols if num_cols else "—")
        st.write("Categóricas:", cat_cols if cat_cols else "—")
        preprocessor = _build_preprocessor(num_cols, cat_cols)

    with st.expander("4) Modelo e Hiperparâmetros", expanded=True):
        model_name = st.selectbox(
            "Modelo:",
            ["Logistic Regression", "Random Forest", "SVM (RBF)", "Gradient Boosting"],
            index=0
        )
        class_weight_balanced = st.checkbox("Usar class_weight='balanced' (quando suportado)", value=False)

        # Hiperparâmetros básicos por modelo
        params = {}
        if model_name in ("Logistic Regression", "SVM (RBF)"):
            params["C"] = st.slider("C (regularização inversa)", 0.01, 10.0, 1.0, 0.01)
        if model_name == "Random Forest":
            params["n_estimators"] = st.slider("n_estimators", 100, 1000, 300, 50)
            max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20], index=0)
            params["max_depth"] = None if max_depth_opt == "None" else int(max_depth_opt)

    with st.expander("5) Treinar & Avaliar", expanded=True):
        go = st.button("Treinar modelo agora", type="primary", use_container_width=True)
        if go:
            # Monta pipeline completo
            model = _get_model(model_name, class_weight_balanced, **params)
            clf = Pipeline(steps=[
                ("pre", preprocessor),
                ("clf", model),
            ])

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Métricas principais
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")

            st.success(f"**Accuracy:** {acc:.3f} | **F1 (macro):** {f1m:.3f}")

            # ROC-AUC (se possível)
            classes = sorted(pd.Series(y).dropna().unique().tolist())
            proba = _safe_predict_proba(clf, X_test)
            if proba is not None:
                try:
                    if len(classes) == 2:
                        auc_val = roc_auc_score((y_test == classes[1]).astype(int), proba[:, 1])
                        st.write(f"**ROC-AUC (binário):** {auc_val:.3f}")
                    else:
                        auc_val = roc_auc_score(pd.get_dummies(y_test, columns=classes), proba, multi_class="ovr")
                        st.write(f"**ROC-AUC (multiclasse, OVR macro):** {auc_val:.3f}")
                except Exception:
                    st.info("Não foi possível calcular ROC-AUC para este caso/modelo.")

            # Relatório de classificação
            st.text("Relatório de Classificação:\n" + classification_report(y_test, y_pred, zero_division=0))

            # Matriz de confusão
            _plot_confusion_matrix(y_test, y_pred, labels=classes)

            # Curva ROC
            if proba is not None:
                _plot_roc(y_test, proba, classes)

            st.divider()

            # Importância de features (Permutation Importance após o pré-processamento)
            with st.spinner("Calculando importâncias (permutation)…"):
                try:
                    r = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                    # Nomes das features pós-transformação:
                    feat_names = clf.named_steps["pre"].get_feature_names_out().tolist()
                    importances = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False)

                    top_k = st.slider("Top-K features mais importantes:", 5, min(30, len(importances)), min(10, len(importances)))
                    fig, ax = plt.subplots(figsize=(6, max(3, int(top_k * 0.35))))
                    importances.head(top_k).iloc[::-1].plot(kind="barh", ax=ax)
                    ax.set_title("Permutation Importance (Top-K)")
                    ax.set_xlabel("Average impact on the metric")
                    st.pyplot(fig)
                except Exception:
                    st.info("Não foi possível calcular importâncias de forma compatível com o pipeline/modelo.")

            st.divider()

            # Predição em todo o dataset (opcional)
            st.subheader("Predição no DataFrame completo (opcional)")
            if st.checkbox("Gerar coluna de predição no DF selecionado", value=False):
                y_full_pred = clf.predict(X)
                col_name = st.text_input("Nome da coluna de saída:", value=f"pred_{target_col}")
                work_df[col_name] = y_full_pred
                st.dataframe(work_df.head(20))
                # Se quiser armazenar na sessão:
                st.session_state[f"ml_pred_{col_name}"] = work_df

            # Expor o pipeline treinado na sessão (para reuso)
            st.caption("O pipeline treinado foi armazenado em `st.session_state['ml_last_classifier']`.")
            st.session_state["ml_last_classifier"] = clf
