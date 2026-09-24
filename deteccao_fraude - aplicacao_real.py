# -*- coding: utf-8 -*-
"""Detecção de fraude em cartões com limiar orientado ao negócio.

Este script foi preparado para execução no Google Colab.
Problema de negócio: detectar pelo menos 80% das fraudes, reduzindo a
quantidade de transações legítimas enviadas para análise manual.
"""

# No Colab, execute antes:
# !pip -q install joblib scikit-learn pandas matplotlib seaborn

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Carregamento ------------------------------------------------------------
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
RANDOM_STATE = 42
RECALL_ALVO = 0.80

df = pd.read_csv(URL)

if "Class" not in df.columns:
    raise ValueError("A coluna alvo 'Class' não foi encontrada no dataset.")

# 2. Feature engineering sem vazamento --------------------------------------
# O Amount_log é calculado antes da separação, mas não usa o alvo. O scaler,
# por outro lado, será ajustado somente nos dados de treino dentro do Pipeline.
df["Amount_log"] = np.log1p(df["Amount"])
df = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df.drop(columns="Class")
y = df["Class"].astype(int)

X_treino, X_temp, y_treino, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
)
X_validacao, X_teste, y_validacao, y_teste = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

# 3. Modelo -------------------------------------------------------------------
# class_weight='balanced' dá mais peso à classe rara durante o treinamento.
# O scaler pertence ao Pipeline para ser ajustado apenas no treino.
colunas_numericas = X.columns.tolist()
preprocessador = ColumnTransformer(
    transformers=[("numericas", StandardScaler(), colunas_numericas)],
    remainder="drop",
)

modelo = Pipeline(
    steps=[
        ("preprocessador", preprocessador),
        (
            "classificador",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

modelo.fit(X_treino, y_treino)

# 4. Escolha do limiar no conjunto de validação -----------------------------
# O limiar padrão 0.50 não é automaticamente adequado para fraude. Aqui,
# escolhemos o maior limiar que ainda mantém o recall mínimo desejado.
prob_validacao = modelo.predict_proba(X_validacao)[:, 1]
precisoes, recalls, limiares = precision_recall_curve(y_validacao, prob_validacao)

candidatos = np.where(recalls[:-1] >= RECALL_ALVO)[0]
if len(candidatos) == 0:
    indice_limiar = int(np.argmax(recalls[:-1]))
    print("Aviso: o recall alvo não foi atingido na validação; usando o melhor recall disponível.")
else:
    # Entre os limiares que cumprem o recall, escolher o de maior precisão.
    indice_limiar = candidatos[np.argmax(precisoes[:-1][candidatos])]

limiar = float(limiares[indice_limiar])
recall_validacao = float(recalls[indice_limiar])
precisao_validacao = float(precisoes[indice_limiar])

# 5. Avaliação final em dados nunca usados nas decisões ----------------------
prob_teste = modelo.predict_proba(X_teste)[:, 1]
pred_teste = (prob_teste >= limiar).astype(int)

print(f"Registros: {len(df):,} | Fraudes: {int(y.sum()):,} ({y.mean():.3%})")
print(f"Divisão: treino={len(X_treino):,}, validação={len(X_validacao):,}, teste={len(X_teste):,}")
print(f"Limiar escolhido: {limiar:.4f}")
print(f"Validação — precisão: {precisao_validacao:.3f} | recall: {recall_validacao:.3f}")
print("\nRelatório no conjunto de teste:")
print(classification_report(y_teste, pred_teste, target_names=["Normal", "Fraude"], zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_teste, prob_teste):.4f}")
print(f"PR-AUC: {average_precision_score(y_teste, prob_teste):.4f}")

# 6. Visualizações úteis para decisão ----------------------------------------
matriz = confusion_matrix(y_teste, pred_teste)
plt.figure(figsize=(5, 4))
sns.heatmap(
    matriz,
    annot=True,
    fmt=",d",
    cmap="Blues",
    xticklabels=["Pred. normal", "Pred. fraude"],
    yticklabels=["Real normal", "Real fraude"],
)
plt.title("Matriz de confusão — conjunto de teste")
plt.tight_layout()
plt.show()

precisao_plot, recall_plot, _ = precision_recall_curve(y_teste, prob_teste)
plt.figure(figsize=(6, 4))
plt.plot(recall_plot, precisao_plot, label=f"PR-AUC = {average_precision_score(y_teste, prob_teste):.3f}")
plt.axvline(RECALL_ALVO, color="red", linestyle="--", label=f"Recall alvo = {RECALL_ALVO:.0%}")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.title("Curva Precisão-Recall")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Exemplo de uso em novas transações -------------------------------------
# Para classificar novos dados, use as mesmas colunas de X, exceto Class.
# O modelo devolve probabilidade e decisão operacional.
exemplos = X_teste.head(5).copy()
exemplos["probabilidade_fraude"] = modelo.predict_proba(exemplos)[:, 1]
exemplos["enviar_para_analise"] = (exemplos["probabilidade_fraude"] >= limiar).astype(int)
print("\nExemplo de saída operacional:")
print(exemplos[["probabilidade_fraude", "enviar_para_analise"]])

# Persistência para reutilização em uma aplicação ou rotina batch.
joblib.dump({"modelo": modelo, "limiar": limiar, "colunas": X.columns.tolist()}, "detector_fraude.joblib")
print("\nArtefato salvo: detector_fraude.joblib")

# Para carregar depois:
# artefato = joblib.load("detector_fraude.joblib")
# prob = artefato["modelo"].predict_proba(novas_transacoes)[:, 1]
# decisao = (prob >= artefato["limiar"]).astype(int)
