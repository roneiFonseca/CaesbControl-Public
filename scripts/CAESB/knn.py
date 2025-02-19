import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import os
import joblib


# Configurar o estilo dos gráficos
plt.style.use("default")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Definir o diretório para salvar os gráficos
GRAPHICS_DIR = "/home/ronei/Documentos/CaesbControl/scripts/CAESB/graficos"

# Carregar os dados
data = pd.read_csv("/home/ronei/Documentos/CaesbControl/scripts/CAESB/dados_limpos.csv")

# Analisar a distribuição das classes
class_distribution = data["LOCAL"].value_counts()
print("\nDistribuição das classes:")
print(class_distribution)
min_samples = class_distribution.min()
print(f"\nMenor número de amostras em uma classe: {min_samples}")

# 1. Visualizar a distribuição dos valores por LOCAL
plt.figure(figsize=(12, 6))
box_data = [data[data["LOCAL"] == local]["R$"] for local in data["LOCAL"].unique()]
plt.boxplot(box_data, labels=data["LOCAL"].unique())
plt.xticks(rotation=45)
plt.title("Distribuição dos Valores por Local")
plt.ylabel("Valor (R$)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHICS_DIR, "distribuicao_valores_por_local.png"))
plt.close()

# 2. Visualizar a contagem de registros por LOCAL
plt.figure(figsize=(10, 6))
class_distribution.plot(kind="bar", color=colors[0])
plt.title("Quantidade de Registros por Local")
plt.xlabel("Local")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHICS_DIR, "contagem_por_local.png"))
plt.close()

# Preparar os dados
X = data["R$"].values.reshape(-1, 1)
y = data["LOCAL"]

# Dividir os dados em treino e teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nDistribuição das classes no conjunto de treino:")
print(pd.Series(y_train).value_counts())
print("\nDistribuição das classes no conjunto de teste:")
print(pd.Series(y_test).value_counts())

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Visualizar a distribuição dos dados de treino e teste
plt.figure(figsize=(10, 6))
plt.hist(X_train.ravel(), bins=30, alpha=0.5, label="Treino", color=colors[0])
plt.hist(X_test.ravel(), bins=30, alpha=0.5, label="Teste", color=colors[1])
plt.title("Distribuição dos Dados de Treino e Teste")
plt.xlabel("Valor (R$)")
plt.ylabel("Frequência")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPHICS_DIR, "distribuicao_treino_teste.png"))
plt.close()

# Encontrar o melhor valor de k
k_values = range(
    1, min(21, len(X_train))
)  # k não pode ser maior que o número de amostras
cv_scores = []

# Usar StratifiedKFold com número de folds baseado no tamanho da menor classe
n_splits = min(5, min_samples)
print(
    f"\nUsando {n_splits}-fold cross-validation (ajustado pelo tamanho da menor classe)"
)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Usar StratifiedKFold para manter a proporção das classes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=skf)
    cv_scores.append(scores.mean())

# 4. Plotar os resultados da validação cruzada
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, "o-", color=colors[0])
plt.xlabel("Valor de K")
plt.ylabel("Acurácia média (CV)")
plt.title(f"Acurácia do KNN para diferentes valores de K ({n_splits}-fold CV)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHICS_DIR, "knn_cv_scores.png"))
plt.close()

# Treinar o modelo com o melhor k
best_k = k_values[np.argmax(cv_scores)]
print(f"\nMelhor valor de K: {best_k}")

# Treinar o modelo final
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = knn.predict(X_test_scaled)

# 5. Criar e visualizar a matriz de confusão
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Matriz de Confusão")
plt.colorbar()

# Adicionar rótulos
classes = np.unique(y)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Adicionar valores na matriz
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(
        j,
        i,
        format(cm[i, j], "d"),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )

plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHICS_DIR, "matriz_confusao.png"))
plt.close()

# 6. Visualizar as previsões vs valores reais
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, c=colors[0], label="Real", alpha=0.5)
plt.scatter(X_test, y_pred, c=colors[1], label="Previsto", alpha=0.5)
plt.xlabel("Valor (R$)")
plt.ylabel("Local")
plt.title("Valores Reais vs Previsões")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPHICS_DIR, "real_vs_predicoes.png"))
plt.close()

# Avaliar o modelo
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMétricas individuais:")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Salvar o modelo treinado
joblib.dump(knn, os.path.join(GRAPHICS_DIR, "knn_model.joblib"))
joblib.dump(scaler, os.path.join(GRAPHICS_DIR, "scaler.joblib"))
