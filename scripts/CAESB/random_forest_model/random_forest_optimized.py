import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler

# Registrar o tempo de início
start_time = time.time()

# Carregar dados
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
csv_path = os.path.join(base_dir, "data", "dados_com_labels.csv")

# Carregar o CSV
print("Carregando dados...")
data_raw = pd.read_csv(csv_path)
data = data_raw[["LOCAL", "Ano", "Mês", "R$", "m3", "anormal"]]

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

# Selecionar as features numéricas
X_numeric = data[numeric_features]

# Tratar valores nulos nas features numéricas
X_numeric = X_numeric.fillna(X_numeric.median())

# Converter LOCAL para valores numéricos (one-hot encoding)
X_categorical = pd.get_dummies(data[["LOCAL"]], drop_first=True)

# Combinar features numéricas e categóricas
X = pd.concat([X_numeric, X_categorical], axis=1)
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

# Normalizar os dados usando RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Usar os melhores parâmetros encontrados pelo GridSearchCV
best_params = {
    "class_weight": "balanced",
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 10,
    "n_estimators": 200,
}

print("\n--- Treinando o modelo Random Forest com os melhores parâmetros ---")
print(f"Parâmetros utilizados: {best_params}")

# Criar e treinar o modelo com os melhores parâmetros
model = RandomForestClassifier(random_state=42, **best_params)

# Realizar validação cruzada para avaliar a estabilidade do modelo
print("\nRealizando validação cruzada com 5 folds...")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")
print(f"Scores de validação cruzada (F1): {cv_scores}")
print(f"Média dos scores de validação cruzada (F1): {cv_scores.mean():.4f}")
print(f"Desvio padrão dos scores de validação cruzada (F1): {cv_scores.std():.4f}")

# Treinar o modelo final com todos os dados de treinamento
model.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Obter probabilidades para ROC e PR curves
y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

# Calcular métricas
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

# Calcular AUC-ROC
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
roc_auc = auc(fpr, tpr)

# Calcular AUC-PR
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test)
pr_auc = auc(recall_curve, precision_curve)

# Calcular e exibir a matriz de confusão
cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

# Calcular métricas adicionais
specificity = tn / (tn + fp)
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print("\n--- Resultados do Modelo Random Forest Otimizado ---")
print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall (Sensibilidade): {rec:.4f}")
print(f"Especificidade: {specificity:.4f}")
print(f"Valor Preditivo Negativo: {npv:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR: {pr_auc:.4f}")

print("\n--- Matriz de Confusão ---")
print(f"Verdadeiros Negativos (TN): {tn}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Falsos Negativos (FN): {fn}")
print(f"Verdadeiros Positivos (TP): {tp}")

# Calcular o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTempo de execução: {execution_time:.2f} segundos")

# Salvar resultados em um arquivo
with open(os.path.join(script_dir, "random_forest_optimized_results.txt"), "w") as f:
    f.write("--- Resultados do Modelo Random Forest Otimizado ---\n\n")
    f.write(f"Parâmetros utilizados: {best_params}\n\n")
    f.write(
        f"Scores de validação cruzada (F1): {[f'{score:.4f}' for score in cv_scores]}\n"
    )
    f.write(f"Média dos scores de validação cruzada (F1): {cv_scores.mean():.4f}\n")
    f.write(
        f"Desvio padrão dos scores de validação cruzada (F1): {cv_scores.std():.4f}\n\n"
    )
    f.write(f"Acurácia: {acc:.4f}\n")
    f.write(f"Precisão: {prec:.4f}\n")
    f.write(f"Recall (Sensibilidade): {rec:.4f}\n")
    f.write(f"Especificidade: {specificity:.4f}\n")
    f.write(f"Valor Preditivo Negativo: {npv:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n")
    f.write(f"AUC-PR: {pr_auc:.4f}\n\n")

    f.write("--- Matriz de Confusão ---\n")
    f.write(f"Verdadeiros Negativos (TN): {tn}\n")
    f.write(f"Falsos Positivos (FP): {fp}\n")
    f.write(f"Falsos Negativos (FN): {fn}\n")
    f.write(f"Verdadeiros Positivos (TP): {tp}\n\n")

    f.write(f"Tempo de execução: {execution_time:.2f} segundos\n")

# Visualizações
plt.figure(figsize=(20, 15))

# 1. Curva ROC
plt.subplot(2, 3, 1)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falsos Positivos", fontsize=12)
plt.ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
plt.title("Curva ROC", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)

# 2. Curva Precision-Recall
plt.subplot(2, 3, 2)
plt.plot(
    recall_curve,
    precision_curve,
    color="green",
    lw=2,
    label=f"PR curve (area = {pr_auc:.2f})",
)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Curva Precision-Recall", fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)

# 3. Matriz de Confusão como Heatmap
plt.subplot(2, 3, 3)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusão", fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Normal (0)", "Anormal (1)"], rotation=45, fontsize=10)
plt.yticks(tick_marks, ["Normal (0)", "Anormal (1)"], fontsize=10)

# Adicionar valores à matriz de confusão
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )

plt.ylabel("Classe Real", fontsize=12)
plt.xlabel("Classe Prevista", fontsize=12)

# 4. Importância das Features
plt.subplot(2, 3, 4)
feature_importance = model.feature_importances_
feature_names = X.columns

# Ordenar por importância e selecionar apenas as 10 mais importantes
indices = np.argsort(feature_importance)[::-1][:10]  # Inverte a ordem e pega as 10 primeiras
top_importances = feature_importance[indices]
top_names = [feature_names[i] for i in indices]

plt.barh(range(len(indices)), top_importances, align="center")
plt.yticks(range(len(indices)), top_names, fontsize=10)
plt.title("Top 10 Features Mais Importantes", fontsize=14)
plt.xlabel("Importância Relativa", fontsize=12)

# 5. Distribuição das Probabilidades
plt.subplot(2, 3, 5)
plt.hist(y_prob_test[y_test == 0], bins=20, alpha=0.5, color="blue", label="Normal (0)")
plt.hist(y_prob_test[y_test == 1], bins=20, alpha=0.5, color="red", label="Anormal (1)")
plt.axvline(x=0.5, color="black", linestyle="--", label="Limiar (0.5)")
plt.xlabel("Probabilidade de Anomalia", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.title("Distribuição das Probabilidades por Classe", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 6. Validação Cruzada
plt.subplot(2, 3, 6)
plt.bar(range(len(cv_scores)), cv_scores, color="skyblue")
plt.axhline(
    y=cv_scores.mean(),
    color="red",
    linestyle="--",
    label=f"Média: {cv_scores.mean():.4f}",
)
plt.xlabel("Fold", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.title("Scores de Validação Cruzada (F1)", fontsize=14)
plt.xticks(range(len(cv_scores)), [f"Fold {i + 1}" for i in range(len(cv_scores))])
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(script_dir, "random_forest_optimized_results.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("\nResultados salvos em:")
print(f"- {os.path.join(script_dir, 'random_forest_optimized_results.txt')}")
print(f"- {os.path.join(script_dir, 'random_forest_optimized_results.png')}")

# Salvar o modelo treinado (opcional)
from joblib import dump

dump(model, os.path.join(script_dir, "random_forest_optimized_model.joblib"))
print(f"- {os.path.join(script_dir, 'random_forest_optimized_model.joblib')}")

# Função para fazer previsões em novos dados
def predict_anomalies(new_data):
    """
    Função para fazer previsões em novos dados.

    Parâmetros:
    new_data (DataFrame): DataFrame com as mesmas colunas que o conjunto de treinamento

    Retorna:
    tuple: (previsões, probabilidades)
    """
    # Aplicar o mesmo pré-processamento
    if isinstance(new_data, pd.DataFrame):
        # Selecionar apenas as features numéricas
        numeric_data = (
            new_data[numeric_features]
            if all(feat in new_data.columns for feat in numeric_features)
            else None
        )

        if numeric_data is not None:
            # Tratar valores nulos
            numeric_data = numeric_data.fillna(X_numeric.median())

            # Aplicar one-hot encoding para LOCAL se presente
            if "LOCAL" in new_data.columns:
                categorical_data = pd.get_dummies(new_data[["LOCAL"]], drop_first=True)
                # Garantir que as colunas sejam as mesmas do treinamento
                for col in X_categorical.columns:
                    if col not in categorical_data.columns:
                        categorical_data[col] = 0
                categorical_data = categorical_data[X_categorical.columns]

                # Combinar features
                processed_data = pd.concat([numeric_data, categorical_data], axis=1)
            else:
                processed_data = numeric_data

            # Aplicar scaling
            scaled_data = scaler.transform(processed_data)

            # Fazer previsões
            predictions = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[:, 1]

            return predictions, probabilities

    return None, None


print(
    "\nModelo pronto para uso. Use a função predict_anomalies() para fazer previsões em novos dados."
)
