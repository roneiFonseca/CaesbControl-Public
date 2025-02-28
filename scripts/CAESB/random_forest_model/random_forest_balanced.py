import os
import time
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline

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
    'max_depth': None,
    'min_samples_leaf': 2,
    'min_samples_split': 10,
    'n_estimators': 200
}

# Definir diferentes técnicas de balanceamento
resampling_techniques = {
    "Sem Balanceamento": None,
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "Random Under-Sampling": RandomUnderSampler(random_state=42),
    "SMOTE + ENN": SMOTEENN(random_state=42),
    "SMOTE + Tomek": SMOTETomek(random_state=42)
}

# Armazenar resultados
results = {}

print("\n--- Comparando Diferentes Técnicas de Balanceamento ---")

for name, resampler in resampling_techniques.items():
    print(f"\n{name}:")
    
    # Criar modelo
    model = RandomForestClassifier(random_state=42, **best_params)
    
    # Aplicar técnica de balanceamento se não for None
    if resampler is not None:
        print(f"Aplicando {name}...")
        X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)
        print(f"Distribuição após balanceamento: {np.bincount(y_resampled)}")
        print(f"Proporção de anomalias após balanceamento: {100 * sum(y_resampled) / len(y_resampled):.2f}%")
    else:
        X_resampled, y_resampled = X_train_scaled, y_train
    
    # Realizar validação cruzada
    print("Realizando validação cruzada...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='f1')
    
    print(f"Scores de validação cruzada (F1): {cv_scores}")
    print(f"Média dos scores de validação cruzada (F1): {cv_scores.mean():.4f}")
    print(f"Desvio padrão dos scores de validação cruzada (F1): {cv_scores.std():.4f}")
    
    # Treinar modelo final
    model.fit(X_resampled, y_resampled)
    
    # Fazer previsões
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calcular AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calcular AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calcular métricas adicionais
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Imprimir resultados
    print(f"\nResultados para {name}:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall (Sensibilidade): {rec:.4f}")
    print(f"Especificidade: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    
    # Armazenar resultados
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "cv_scores": cv_scores,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve
    }

# Identificar a melhor técnica com base no F1-Score
best_technique = max(results.items(), key=lambda x: x[1]["f1"])
print(f"\n--- Melhor Técnica: {best_technique[0]} (F1-Score: {best_technique[1]['f1']:.4f}) ---")

# Calcular o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTempo de execução total: {execution_time:.2f} segundos")

# Salvar resultados em um arquivo
with open(os.path.join(script_dir, "random_forest_balanced_comparison.txt"), "w") as f:
    f.write("--- Comparação de Técnicas de Balanceamento para Random Forest ---\n\n")
    
    for name, result in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Acurácia: {result['accuracy']:.4f}\n")
        f.write(f"  Precisão: {result['precision']:.4f}\n")
        f.write(f"  Recall (Sensibilidade): {result['recall']:.4f}\n")
        f.write(f"  Especificidade: {result['specificity']:.4f}\n")
        f.write(f"  Valor Preditivo Negativo: {result['npv']:.4f}\n")
        f.write(f"  F1-Score: {result['f1']:.4f}\n")
        f.write(f"  AUC-ROC: {result['roc_auc']:.4f}\n")
        f.write(f"  AUC-PR: {result['pr_auc']:.4f}\n")
        f.write(f"  Média CV (F1): {result['cv_scores'].mean():.4f}\n")
        f.write(f"  Desvio Padrão CV (F1): {result['cv_scores'].std():.4f}\n\n")
    
    f.write(f"Melhor Técnica: {best_technique[0]} (F1-Score: {best_technique[1]['f1']:.4f})\n\n")
    f.write(f"Tempo de execução total: {execution_time:.2f} segundos\n")

# Visualizações
plt.figure(figsize=(20, 15))

# 1. Comparação de F1-Scores
plt.subplot(2, 2, 1)
techniques = list(results.keys())
f1_scores = [results[t]["f1"] for t in techniques]
plt.bar(techniques, f1_scores, color='skyblue')
plt.axhline(y=results["Sem Balanceamento"]["f1"], color='red', linestyle='--', 
           label=f'Baseline (F1: {results["Sem Balanceamento"]["f1"]:.4f})')
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1-Score', fontsize=12)
plt.title('Comparação de F1-Scores por Técnica de Balanceamento', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Comparação de Recall (Sensibilidade)
plt.subplot(2, 2, 2)
recall_scores = [results[t]["recall"] for t in techniques]
plt.bar(techniques, recall_scores, color='lightgreen')
plt.axhline(y=results["Sem Balanceamento"]["recall"], color='red', linestyle='--',
           label=f'Baseline (Recall: {results["Sem Balanceamento"]["recall"]:.4f})')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Recall (Sensibilidade)', fontsize=12)
plt.title('Comparação de Recall por Técnica de Balanceamento', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Curvas ROC
plt.subplot(2, 2, 3)
for name, result in results.items():
    plt.plot(result["fpr"], result["tpr"], lw=2, 
             label=f'{name} (AUC = {result["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
plt.title('Comparação de Curvas ROC', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)

# 4. Curvas Precision-Recall
plt.subplot(2, 2, 4)
for name, result in results.items():
    plt.plot(result["recall_curve"], result["precision_curve"], lw=2,
             label=f'{name} (AUC = {result["pr_auc"]:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Comparação de Curvas Precision-Recall', fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "random_forest_balanced_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# Treinar modelo final com a melhor técnica
print(f"\n--- Treinando modelo final com {best_technique[0]} ---")

# Criar pipeline com a melhor técnica
best_resampler = resampling_techniques[best_technique[0]]
best_model = RandomForestClassifier(random_state=42, **best_params)

if best_resampler is not None:
    # Aplicar o melhor resampler
    X_resampled, y_resampled = best_resampler.fit_resample(X_train_scaled, y_train)
    best_model.fit(X_resampled, y_resampled)
else:
    # Usar dados originais
    best_model.fit(X_train_scaled, y_train)

# Fazer previsões finais
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Calcular matriz de confusão final
cm_final = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm_final.ravel()

# Visualizar matriz de confusão final
plt.figure(figsize=(10, 8))
plt.imshow(cm_final, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Matriz de Confusão - {best_technique[0]}', fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal (0)', 'Anormal (1)'], rotation=45, fontsize=12)
plt.yticks(tick_marks, ['Normal (0)', 'Anormal (1)'], fontsize=12)

# Adicionar valores à matriz de confusão
thresh = cm_final.max() / 2.
for i in range(cm_final.shape[0]):
    for j in range(cm_final.shape[1]):
        plt.text(j, i, format(cm_final[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_final[i, j] > thresh else "black",
                 fontsize=14)

plt.ylabel('Classe Real', fontsize=12)
plt.xlabel('Classe Prevista', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "random_forest_best_technique_cm.png"), dpi=300)
plt.show()

# Salvar o modelo final
from joblib import dump
dump(best_model, os.path.join(script_dir, 'random_forest_balanced_model.joblib'))
print(f"Modelo final salvo em: {os.path.join(script_dir, 'random_forest_balanced_model.joblib')}")

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
        numeric_data = new_data[numeric_features] if all(feat in new_data.columns for feat in numeric_features) else None
        
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
            predictions = best_model.predict(scaled_data)
            probabilities = best_model.predict_proba(scaled_data)[:, 1]
            
            return predictions, probabilities
    
    return None, None

print("\nModelo pronto para uso. Use a função predict_anomalies() para fazer previsões em novos dados.")
