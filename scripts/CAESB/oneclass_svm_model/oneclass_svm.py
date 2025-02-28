import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

# Carregar dados
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
csv_path = os.path.join(base_dir, "data", "dados_com_labels.csv")

# Carregar o CSV
print("Carregando dados...")
data_raw = pd.read_csv(csv_path)
data = data_raw[["LOCAL", "Ano", "Mês", "R$", "m3", "anormal"]]
# data = data_raw.copy()
# Verificar informações sobre os dados
print(f"Formato dos dados antes de feature engineering: {data.shape}")
print(f"Valores nulos no dataset: {data.isnull().sum().sum()}")
print(f"Distribuição de classes: \n{data['anormal'].value_counts()}")
print(f"Porcentagem de anomalias: {100 * sum(data['anormal']) / len(data):.2f}%")
print(f"Total de anomalias: {sum(data['anormal'])}")

# Pré-processamento e Feature Engineering
print("Realizando feature engineering...")

# Selecionar features
numeric_features = ["R$", "m3"]
# selecionar todas as colunas númericas exceto anomalias
# numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
# numeric_features.remove("anormal")
# print(f"Features numéricas selecionadas: {numeric_features}")

# Selecionar as features numéricas
X_numeric = data[numeric_features]

# Tratar valores nulos nas features numéricas
X_numeric = X_numeric.fillna(X_numeric.median())

# Converter LOCAL para valores numéricos (one-hot encoding)
X_categorical = pd.get_dummies(data[["LOCAL"]], drop_first=True)

# Combinar features numéricas e categóricas
X = pd.concat([X_numeric, X_categorical], axis=1)
print(f"Formato dos dados após feature engineering: {X.shape}")
# X = X_numeric
print(f"Formato dos dados após feature engineering: {X.shape}")

# Obter o target (anormal) - 1 para anomalias, 0 para normal
y = data["anormal"]

# Dividir em conjuntos de treino e teste
print("Dividindo em conjuntos de treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamanho do conjunto de treino: {X_train.shape}")
print(f"Tamanho do conjunto de teste: {X_test.shape}")
print(
    f"Proporção de anomalias no conjunto de treino: {100 * sum(y_train) / len(y_train):.2f}%"
)
print(
    f"Proporção de anomalias no conjunto de teste: {100 * sum(y_test) / len(y_test):.2f}%"
)

# Para One-Class SVM, é comum treinar apenas com dados normais (não anômalos)
# Vamos selecionar apenas os dados normais para treinamento
X_train_normal = X_train[y_train == 0]
print(f"Tamanho do conjunto de treino (apenas dados normais): {X_train_normal.shape}")

# Criar o scaler uma vez para reutilizar
scaler = RobustScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testar diferentes valores de nu
nu_values = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
results = []

print("\n--- Testando diferentes valores de nu ---")
for nu in nu_values:
    print(f"\nTestando nu = {nu}")

    # Criar e treinar o modelo
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu,
    )

    # Treinar o modelo apenas com dados normais
    model.fit(X_train_normal_scaled)

    # Fazer previsões
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Converter saída do One-Class SVM (-1 para anomalias, 1 para normal)
    # para o formato do nosso target (1 para anomalias, 0 para normal)
    y_pred_train = np.where(y_pred_train == -1, 1, 0)
    y_pred_test = np.where(y_pred_test == -1, 1, 0)

    # Calcular score de anomalia
    anomaly_score_train = -model.decision_function(X_train_scaled)
    anomaly_score_test = -model.decision_function(X_test_scaled)

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)

    # Calcular AUC-ROC se possível
    try:
        fpr, tpr, _ = roc_curve(y_test, anomaly_score_test)
        roc_auc = auc(fpr, tpr)
    except:
        roc_auc = 0

    # Calcular AUC-PR se possível
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test, anomaly_score_test
        )
        pr_auc = auc(recall_curve, precision_curve)
    except:
        pr_auc = 0

    # Imprimir métricas
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")

    # Armazenar resultados
    results.append(
        {
            "nu": nu,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "y_pred_test": y_pred_test,
            "anomaly_score_test": anomaly_score_test,
        }
    )

# Criar DataFrame com os resultados
results_df = pd.DataFrame(
    [
        {k: v for k, v in r.items() if k not in ["y_pred_test", "anomaly_score_test"]}
        for r in results
    ]
)

# Imprimir tabela de resultados
print("\n--- Resumo dos Resultados ---")
print(results_df.to_string(index=False))

# Salvar resultados em um arquivo
with open(os.path.join(script_dir, "oneclass_svm_nu_comparison.txt"), "w") as f:
    f.write("--- Comparação de Diferentes Valores de nu para One-Class SVM ---\n\n")
    f.write(results_df.to_string(index=False) + "\n\n")

    # Adicionar detalhes sobre o melhor valor de nu para cada métrica
    f.write("Melhor valor de nu para cada métrica:\n")
    f.write(
        f"Acurácia: nu = {results_df.loc[results_df['accuracy'].idxmax(), 'nu']} ({results_df['accuracy'].max():.4f})\n"
    )
    f.write(
        f"Precisão: nu = {results_df.loc[results_df['precision'].idxmax(), 'nu']} ({results_df['precision'].max():.4f})\n"
    )
    f.write(
        f"Recall: nu = {results_df.loc[results_df['recall'].idxmax(), 'nu']} ({results_df['recall'].max():.4f})\n"
    )
    f.write(
        f"F1-Score: nu = {results_df.loc[results_df['f1'].idxmax(), 'nu']} ({results_df['f1'].max():.4f})\n"
    )
    f.write(
        f"AUC-ROC: nu = {results_df.loc[results_df['roc_auc'].idxmax(), 'nu']} ({results_df['roc_auc'].max():.4f})\n"
    )
    f.write(
        f"AUC-PR: nu = {results_df.loc[results_df['pr_auc'].idxmax(), 'nu']} ({results_df['pr_auc'].max():.4f})\n"
    )

print(
    f"Resultados salvos em: {os.path.join(script_dir, 'oneclass_svm_nu_comparison.txt')}"
)

# Gerar gráficos
plt.figure(figsize=(15, 10))

# 1. Gráfico de linha para todas as métricas
plt.subplot(2, 2, 1)
plt.plot(results_df["nu"], results_df["accuracy"], marker="o", label="Acurácia")
plt.plot(results_df["nu"], results_df["precision"], marker="s", label="Precisão")
plt.plot(results_df["nu"], results_df["recall"], marker="^", label="Recall")
plt.plot(results_df["nu"], results_df["f1"], marker="d", label="F1-Score")
plt.xlabel("Valor de nu")
plt.ylabel("Valor da Métrica")
plt.title("Métricas por Valor de nu")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Gráfico de Precisão vs Recall
plt.subplot(2, 2, 2)
plt.plot(results_df["recall"], results_df["precision"], marker="o")
for i, nu in enumerate(results_df["nu"]):
    plt.annotate(
        f"{nu}",
        (results_df["recall"][i], results_df["precision"][i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.title("Precisão vs Recall para Diferentes Valores de nu")
plt.grid(True, alpha=0.3)

# 3. Gráfico de barras para F1-Score
plt.subplot(2, 2, 3)
plt.bar(results_df["nu"].astype(str), results_df["f1"], color="skyblue")
plt.xlabel("Valor de nu")
plt.ylabel("F1-Score")
plt.title("F1-Score por Valor de nu")
plt.grid(True, alpha=0.3, axis="y")

# 4. Gráfico de linha para AUC-ROC e AUC-PR
plt.subplot(2, 2, 4)
plt.plot(results_df["nu"], results_df["roc_auc"], marker="o", label="AUC-ROC")
plt.plot(results_df["nu"], results_df["pr_auc"], marker="s", label="AUC-PR")
plt.xlabel("Valor de nu")
plt.ylabel("Área sob a curva")
plt.title("AUC-ROC e AUC-PR por Valor de nu")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "oneclass_svm_nu_comparison.png"))
print(f"Gráfico salvo em: {os.path.join(script_dir, 'oneclass_svm_nu_comparison.png')}")

# Gerar curvas ROC e PR para o melhor valor de nu segundo F1-Score
best_f1_idx = results_df["f1"].idxmax()
best_nu = results_df.loc[best_f1_idx, "nu"]
best_result = results[best_f1_idx]

plt.figure(figsize=(15, 6))

# Curva ROC
plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(y_test, best_result["anomaly_score_test"])
plt.plot(fpr, tpr, label=f"AUC = {best_result['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title(f"Curva ROC (nu = {best_nu})")
plt.legend()
plt.grid(True, alpha=0.3)

# Curva Precisão-Recall
plt.subplot(1, 2, 2)
precision_curve, recall_curve, _ = precision_recall_curve(
    y_test, best_result["anomaly_score_test"]
)
plt.plot(recall_curve, precision_curve, label=f"AUC = {best_result['pr_auc']:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.title(f"Curva Precisão-Recall (nu = {best_nu})")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "oneclass_svm_best_curves.png"))
print(
    f"Curvas para o melhor valor de nu salvas em: {os.path.join(script_dir, 'oneclass_svm_best_curves.png')}"
)

# Gerar matriz de confusão para o melhor valor de nu
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_result["y_pred_test"])
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Matriz de Confusão (nu = {best_nu})")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Normal", "Anomalia"])
plt.yticks(tick_marks, ["Normal", "Anomalia"])

# Adicionar valores na matriz
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.ylabel("Classe Real")
plt.xlabel("Classe Prevista")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "oneclass_svm_best_confusion_matrix.png"))
print(
    f"Matriz de confusão salva em: {os.path.join(script_dir, 'oneclass_svm_best_confusion_matrix.png')}"
)

print("\nAnálise concluída!")
