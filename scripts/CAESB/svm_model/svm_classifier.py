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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

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

# Definir parâmetros para Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear', 'poly'],
    'class_weight': ['balanced', None]
}

# Criar e treinar o modelo SVM com GridSearchCV
print("\n--- Executando Grid Search para encontrar os melhores parâmetros ---")
grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Adicionar informação de progresso após o início do GridSearchCV
print("\n--- Progresso do Grid Search ---")
print("GridSearchCV está em andamento. Isso pode levar algum tempo...")
print("Processando os melhores parâmetros para o modelo...")

# Obter os melhores parâmetros
best_params = grid_search.best_params_
print(f"\nMelhores parâmetros encontrados: {best_params}")

# Treinar o modelo com os melhores parâmetros
best_model = SVC(
    probability=True,
    random_state=42,
    **best_params
)

best_model.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

# Obter probabilidades para ROC e PR curves
y_prob_train = best_model.predict_proba(X_train_scaled)[:, 1]
y_prob_test = best_model.predict_proba(X_test_scaled)[:, 1]

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

print("\n--- Resultados do Modelo SVM ---")
print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR: {pr_auc:.4f}")

print("\n--- Matriz de Confusão ---")
print(f"Verdadeiros Negativos (TN): {tn}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Falsos Negativos (FN): {fn}")
print(f"Verdadeiros Positivos (TP): {tp}")

# Salvar resultados em um arquivo
with open(os.path.join(script_dir, "svm_classifier_results.txt"), "w") as f:
    f.write("--- Resultados do Modelo SVM ---\n\n")
    f.write(f"Melhores parâmetros: {best_params}\n\n")
    f.write(f"Acurácia: {acc:.4f}\n")
    f.write(f"Precisão: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n")
    f.write(f"AUC-PR: {pr_auc:.4f}\n\n")
    
    f.write("--- Matriz de Confusão ---\n")
    f.write(f"Verdadeiros Negativos (TN): {tn}\n")
    f.write(f"Falsos Positivos (FP): {fp}\n")
    f.write(f"Falsos Negativos (FN): {fn}\n")
    f.write(f"Verdadeiros Positivos (TP): {tp}\n")

# Visualizações
plt.figure(figsize=(16, 12))

# 1. Curva ROC
plt.subplot(2, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 2. Curva Precision-Recall
plt.subplot(2, 2, 2)
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)

# 3. Matriz de Confusão como Heatmap
plt.subplot(2, 2, 3)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal (0)', 'Anormal (1)'], rotation=45)
plt.yticks(tick_marks, ['Normal (0)', 'Anormal (1)'])

# Adicionar valores à matriz de confusão
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')

# 4. Importância das Features (para SVM linear)
if best_params.get('kernel') == 'linear':
    plt.subplot(2, 2, 4)
    feature_importance = np.abs(best_model.coef_[0])
    feature_names = X.columns
    
    # Ordenar por importância
    indices = np.argsort(feature_importance)
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Importância das Features (SVM Linear)')
    plt.xlabel('Importância Relativa')
else:
    # Se não for kernel linear, mostrar distribuição das previsões
    plt.subplot(2, 2, 4)
    plt.hist(y_prob_test, bins=20, alpha=0.7, color='blue', label='Probabilidades')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Limiar (0.5)')
    plt.xlabel('Probabilidade de Anomalia')
    plt.ylabel('Contagem')
    plt.title('Distribuição das Probabilidades de Previsão')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "svm_classifier_results.png"), dpi=300, bbox_inches='tight')
plt.show()

print("\nResultados salvos em:")
print(f"- {os.path.join(script_dir, 'svm_classifier_results.txt')}")
print(f"- {os.path.join(script_dir, 'svm_classifier_results.png')}")
