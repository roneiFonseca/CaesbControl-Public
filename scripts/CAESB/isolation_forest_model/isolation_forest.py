import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Carregar dados
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
csv_path = os.path.join(base_dir, "data", "dados_com_labels.csv")

# Carregar o CSV
print("Carregando dados...")
data_raw = pd.read_csv(csv_path)
data = data_raw[["LOCAL", "Ano", "Mês", "R$", "m3", "anormal"]]
# Verificar se há valores nulos
print(f"Formato dos dados antes de feature engineering: {data.shape}")
print(f"Valores nulos no dataset: {data.isnull().sum().sum()}")
print(f"Distribuição de classes: \n{data['anormal'].value_counts()}")
print(f"Porcentagem de anomalias: {100 * sum(data['anormal']) / len(data):.2f}%")
print(sum(data["anormal"]))
# Pré-processamento e Feature Engineering
print("Realizando feature engineering...")

# Selecionar features relevantes para análise
# Adicionando mais features além de R$ e m3
numeric_features = ["R$", "m3"]

# Selecionar as features numéricas
X_numeric = data[numeric_features]

# Tratar valores nulos nas features numéricas
X_numeric = X_numeric.fillna(X_numeric.median())

# Converter LOCAL para valores numéricos (one-hot encoding)
X_categorical = pd.get_dummies(data[["LOCAL"]], drop_first=True)

# Combinar features numéricas e categóricas
X = pd.concat([X_numeric, X_categorical], axis=1)

# print(X.head())
# X = X_numeric
# Obter o target (anormal) - 1 para anomalias, 0 para normal
y = data["anormal"]

print(f"Formato dos dados após feature engineering: {X.shape}")
print(f"Distribuição de classes: \n{y.value_counts()}")
print(f"Porcentagem de anomalias: {100 * sum(y) / len(y):.2f}%")

# Dividir em treino e teste (80% treino, 20% teste)
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
# Criar pipeline com normalização e modelo
print("Criando pipeline com normalização e modelo...")
pipeline = Pipeline(
    [
        ("scaler", RobustScaler()),  # RobustScaler é menos sensível a outliers
        (
            "model",
            IsolationForest(
                n_estimators=200,
                max_samples="auto",
                contamination=float(
                    # sum(y) / len(y)
                    0.03
                ),  # Proporção de anomalias nos dados
                max_features=len(X.columns),
                random_state=42,
                bootstrap=True,
            ),
        ),
    ]
)

# Treinar o modelo
print("Treinando modelo Isolation Forest...")
pipeline.fit(X_train)

# Extrair o modelo treinado
model = pipeline.named_steps["model"]

# Fazer previsões (1 para normal, -1 para anomalias)
y_pred_train = model.predict(pipeline.named_steps["scaler"].transform(X_train))
y_pred_test = model.predict(pipeline.named_steps["scaler"].transform(X_test))

# Converter saída do Isolation Forest (-1 para anomalias, 1 para normal)
# para o formato do nosso target (1 para anomalias, 0 para normal)
y_pred_train = np.where(y_pred_train == -1, 1, 0)
y_pred_test = np.where(y_pred_test == -1, 1, 0)

# Calcular score de anomalia
anomaly_score_train = model.decision_function(
    pipeline.named_steps["scaler"].transform(X_train)
)
anomaly_score_test = model.decision_function(
    pipeline.named_steps["scaler"].transform(X_test)
)

# Avaliar o modelo
print("\n--- Avaliação do Modelo ---")

# Métricas para conjunto de treino
print("\nMétricas no conjunto de TREINO:")
print(f"Acurácia: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Precisão: {precision_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_train, y_pred_train, zero_division=0):.4f}")
print("\nMatriz de Confusão (Treino):")
print(confusion_matrix(y_train, y_pred_train))
print("\nRelatório de Classificação (Treino):")
print(classification_report(y_train, y_pred_train, zero_division=0))

# Métricas para conjunto de teste
print("\nMétricas no conjunto de TESTE:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
print("\nMatriz de Confusão (Teste):")
print(confusion_matrix(y_test, y_pred_test))
print("\nRelatório de Classificação (Teste):")
print(classification_report(y_test, y_pred_test, zero_division=0))

""" 
# Visualização dos resultados
plt.figure(figsize=(12, 6))

# Histograma dos scores de anomalia para treino
plt.subplot(1, 2, 1)
plt.hist(anomaly_score_train, bins=50, alpha=0.7)
plt.title("Distribuição de Scores de Anomalia (Treino)")
plt.xlabel("Score de Anomalia")
plt.ylabel("Frequência")
plt.grid(True, alpha=0.3)

# Histograma dos scores de anomalia para teste
plt.subplot(1, 2, 2)
plt.hist(anomaly_score_test, bins=50, alpha=0.7)
plt.title("Distribuição de Scores de Anomalia (Teste)")
plt.xlabel("Score de Anomalia")
plt.ylabel("Frequência")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "isolation_forest_scores.png"))
print(f"Gráfico salvo em: {os.path.join(script_dir, 'isolation_forest_scores.png')}")

# Visualizar anomalias em 2D (usando as duas primeiras features após normalização)
plt.figure(figsize=(15, 10))

# Selecionar as duas features mais importantes (R$ e m3)
X_train_scaled = pipeline.named_steps["scaler"].transform(X_train)
X_test_scaled = pipeline.named_steps["scaler"].transform(X_test)

# Plot para conjunto de treino
plt.subplot(2, 2, 1)
plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_pred_train,
    cmap="coolwarm",
    alpha=0.7,
)
plt.title("Anomalias Detectadas (Treino)")
plt.xlabel("Feature 1 (Normalizada)")
plt.ylabel("Feature 2 (Normalizada)")
plt.colorbar(label="Anomalia (1) / Normal (0)")
plt.grid(True, alpha=0.3)

# Plot para conjunto de teste
plt.subplot(2, 2, 2)
plt.scatter(
    X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_test, cmap="coolwarm", alpha=0.7
)
plt.title("Anomalias Detectadas (Teste)")
plt.xlabel("Feature 1 (Normalizada)")
plt.ylabel("Feature 2 (Normalizada)")
plt.colorbar(label="Anomalia (1) / Normal (0)")
plt.grid(True, alpha=0.3)

# Plot para conjunto de treino (real vs predito)
plt.subplot(2, 2, 3)
plt.scatter(
    X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap="coolwarm", alpha=0.7
)
plt.title("Anomalias Reais (Treino)")
plt.xlabel("Feature 1 (Normalizada)")
plt.ylabel("Feature 2 (Normalizada)")
plt.colorbar(label="Anomalia (1) / Normal (0)")
plt.grid(True, alpha=0.3)

# Plot para conjunto de teste (real vs predito)
plt.subplot(2, 2, 4)
plt.scatter(
    X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap="coolwarm", alpha=0.7
)
plt.title("Anomalias Reais (Teste)")
plt.xlabel("Feature 1 (Normalizada)")
plt.ylabel("Feature 2 (Normalizada)")
plt.colorbar(label="Anomalia (1) / Normal (0)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "isolation_forest_anomalies.png"))
print(f"Gráfico salvo em: {os.path.join(script_dir, 'isolation_forest_anomalies.png')}")

# Análise de importância das features
print("\n--- Análise de Importância das Features ---")
# O Isolation Forest não fornece diretamente importância de features
# Podemos analisar a correlação entre as features e as anomalias detectadas

# Criar um DataFrame com as features e as previsões
X_test_df = pd.DataFrame(X_test.values, columns=X_test.columns)
X_test_df["anomalia_prevista"] = y_pred_test
X_test_df["anomalia_real"] = y_test.values

# Calcular correlação entre as features e as anomalias
correlation_with_pred = X_test_df.corr()["anomalia_prevista"].sort_values(
    ascending=False
)
correlation_with_real = X_test_df.corr()["anomalia_real"].sort_values(ascending=False)

print("\nCorrelação entre features e anomalias previstas:")
print(correlation_with_pred.head(10))

print("\nCorrelação entre features e anomalias reais:")
print(correlation_with_real.head(10))

# Salvar resultados em um arquivo
with open(os.path.join(script_dir, "isolation_forest_results.txt"), "w") as f:
    f.write("--- Avaliação do Modelo Isolation Forest ---\n\n")
    f.write(f"Número total de amostras: {len(data)}\n")
    f.write(f"Número de features: {X.shape[1]}\n")
    f.write(f"Porcentagem de anomalias: {100 * sum(y) / len(y):.2f}%\n\n")

    f.write("Métricas no conjunto de TESTE:\n")
    f.write(f"Acurácia: {accuracy_score(y_test, y_pred_test):.4f}\n")
    f.write(f"Precisão: {precision_score(y_test, y_pred_test, zero_division=0):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred_test, zero_division=0):.4f}\n")
    f.write(f"F1-Score: {f1_score(y_test, y_pred_test, zero_division=0):.4f}\n\n")

    f.write("Matriz de Confusão (Teste):\n")
    f.write(str(confusion_matrix(y_test, y_pred_test)) + "\n\n")

    f.write("Relatório de Classificação (Teste):\n")
    f.write(str(classification_report(y_test, y_pred_test, zero_division=0)) + "\n\n")

    f.write("Correlação entre features e anomalias previstas:\n")
    f.write(str(correlation_with_pred.head(10)) + "\n\n")

    f.write("Correlação entre features e anomalias reais:\n")
    f.write(str(correlation_with_real.head(10)) + "\n\n")

print(
    f"Resultados salvos em: {os.path.join(script_dir, 'isolation_forest_results.txt')}"
)
print("\nAnálise concluída!")
