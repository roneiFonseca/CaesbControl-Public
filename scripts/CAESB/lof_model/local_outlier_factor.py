import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
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

# Criar pipeline com normalização e modelo
print("Criando pipeline com normalização e modelo Local Outlier Factor...")

# Para LOF, vamos treinar com todos os dados, mas o modelo será ajustado para detectar anomalias
# Diferente do One-Class SVM, o LOF não tem uma fase de treinamento separada
# Ele calcula a densidade local para cada ponto em relação aos seus vizinhos

# Criar pipeline com o modelo LOF
pipeline = Pipeline(
    [
        ("scaler", RobustScaler()),  # RobustScaler é menos sensível a outliers
        (
            "model",
            LocalOutlierFactor(
                n_neighbors=20,  # Número de vizinhos a considerar
                contamination=0.03,  # Proporção esperada de outliers
                novelty=True,  # Permite usar predict() após o fit()
                n_jobs=-1,  # Usar todos os processadores disponíveis
            ),
        ),
    ]
)

# Treinar o modelo
print("Treinando modelo Local Outlier Factor...")
# LOF com novelty=True precisa ser treinado apenas com dados normais
X_train_normal = X_train[y_train == 0]
print(f"Tamanho do conjunto de treino (apenas dados normais): {X_train_normal.shape}")

# Ajustar o modelo
pipeline.fit(X_train_normal)

# Extrair o modelo treinado
model = pipeline.named_steps["model"]

# Fazer previsões (1 para normal, -1 para anomalias)
y_pred_train = model.predict(pipeline.named_steps["scaler"].transform(X_train))
y_pred_test = model.predict(pipeline.named_steps["scaler"].transform(X_test))

# Converter saída do LOF (-1 para anomalias, 1 para normal)
# para o formato do nosso target (1 para anomalias, 0 para normal)
y_pred_train = np.where(y_pred_train == -1, 1, 0)
y_pred_test = np.where(y_pred_test == -1, 1, 0)

# Calcular score de anomalia (negativo da função de decisão, quanto maior, mais anômalo)
anomaly_score_train = -model.decision_function(
    pipeline.named_steps["scaler"].transform(X_train)
)
anomaly_score_test = -model.decision_function(
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

# Análise de importância das features
print("\n--- Análise de Importância das Features ---")

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
with open(os.path.join(script_dir, "local_outlier_factor_results.txt"), "w") as f:
    f.write("--- Avaliação do Modelo Local Outlier Factor ---\n\n")
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
    f"Resultados salvos em: {os.path.join(script_dir, 'local_outlier_factor_results.txt')}"
)
print("\nAnálise concluída!")
