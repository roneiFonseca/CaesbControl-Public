import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

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
numeric_features = ["m3"]  # Vamos prever R$ baseado em m3

# Selecionar as features numéricas
X_numeric = data[numeric_features]

# Tratar valores nulos nas features numéricas
X_numeric = X_numeric.fillna(X_numeric.median())

# Converter LOCAL para valores numéricos (one-hot encoding)
X_categorical = pd.get_dummies(data[["LOCAL"]], drop_first=True)

# Combinar features numéricas e categóricas
# X = pd.concat([X_numeric, X_categorical], axis=1)
X = X_numeric
print(f"Formato dos dados após feature engineering: {X.shape}")

# Obter o target (R$)
y = data["R$"]

# Dividir em conjuntos de treino e teste
print("Dividindo em conjuntos de treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Tamanho do conjunto de treino: {X_train.shape}")
print(f"Tamanho do conjunto de teste: {X_test.shape}")

# Criar o scaler uma vez para reutilizar
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testar diferentes graus de polinômio para a regressão
poly_degrees = [1, 2, 3]
results = []

print("\n--- Testando diferentes graus polinomiais ---")
for degree in poly_degrees:
    print(f"\nTestando grau polinomial = {degree}")
    
    # Criar e treinar o modelo
    if degree == 1:
        # Regressão linear simples
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Fazer previsões
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        # Regressão polinomial
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_test_poly = poly_features.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Fazer previsões
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
    
    # Calcular resíduos
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    # Calcular métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Detectar anomalias (valores fora de 3 desvios padrão)
    residual_std = np.std(residuals_test)
    anomaly_threshold = 3 * residual_std
    anomalies = np.abs(residuals_test) > anomaly_threshold
    anomaly_count = np.sum(anomalies)
    anomaly_percent = 100 * anomaly_count / len(y_test)
    
    # Imprimir métricas
    print(f"R² (Treino): {r2_train:.4f}")
    print(f"R² (Teste): {r2_test:.4f}")
    print(f"RMSE (Treino): {rmse_train:.4f}")
    print(f"RMSE (Teste): {rmse_test:.4f}")
    print(f"MAE (Treino): {mae_train:.4f}")
    print(f"MAE (Teste): {mae_test:.4f}")
    print(f"Anomalias detectadas: {anomaly_count} ({anomaly_percent:.2f}%)")
    
    # Armazenar resultados
    results.append({
        'degree': degree,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'anomaly_count': anomaly_count,
        'anomaly_percent': anomaly_percent,
        'residuals_test': residuals_test,
        'y_pred_test': y_pred_test,
        'anomaly_threshold': anomaly_threshold,
        'anomalies': anomalies
    })

# Criar DataFrame com os resultados
results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['residuals_test', 'y_pred_test', 'anomalies']}
    for r in results
])

# Imprimir tabela de resultados
print("\n--- Resumo dos Resultados ---")
print(results_df.to_string(index=False))

# Salvar resultados em um arquivo
with open(os.path.join(script_dir, "linear_regression_comparison.txt"), "w") as f:
    f.write("--- Comparação de Diferentes Graus Polinomiais para Regressão Linear ---\n\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    
    # Adicionar detalhes sobre o melhor valor para cada métrica
    f.write("Melhor grau polinomial para cada métrica:\n")
    f.write(f"R² (Teste): grau = {results_df.loc[results_df['r2_test'].idxmax(), 'degree']} ({results_df['r2_test'].max():.4f})\n")
    f.write(f"RMSE (Teste): grau = {results_df.loc[results_df['rmse_test'].idxmin(), 'degree']} ({results_df['rmse_test'].min():.4f})\n")
    f.write(f"MAE (Teste): grau = {results_df.loc[results_df['mae_test'].idxmin(), 'degree']} ({results_df['mae_test'].min():.4f})\n")

print(f"Resultados salvos em: {os.path.join(script_dir, 'linear_regression_comparison.txt')}")

# Gerar gráficos
plt.figure(figsize=(15, 10))

# 1. Gráfico de linha para métricas de erro
plt.subplot(2, 2, 1)
plt.plot(results_df['degree'], results_df['rmse_train'], marker='o', label='RMSE (Treino)')
plt.plot(results_df['degree'], results_df['rmse_test'], marker='s', label='RMSE (Teste)')
plt.plot(results_df['degree'], results_df['mae_train'], marker='^', label='MAE (Treino)')
plt.plot(results_df['degree'], results_df['mae_test'], marker='d', label='MAE (Teste)')
plt.xlabel('Grau Polinomial')
plt.ylabel('Erro')
plt.title('Métricas de Erro por Grau Polinomial')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Gráfico de linha para R²
plt.subplot(2, 2, 2)
plt.plot(results_df['degree'], results_df['r2_train'], marker='o', label='R² (Treino)')
plt.plot(results_df['degree'], results_df['r2_test'], marker='s', label='R² (Teste)')
plt.xlabel('Grau Polinomial')
plt.ylabel('R²')
plt.title('R² por Grau Polinomial')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Gráfico de barras para anomalias detectadas
plt.subplot(2, 2, 3)
plt.bar(results_df['degree'].astype(str), results_df['anomaly_percent'], color='skyblue')
plt.xlabel('Grau Polinomial')
plt.ylabel('Anomalias Detectadas (%)')
plt.title('Porcentagem de Anomalias Detectadas por Grau Polinomial')
plt.grid(True, alpha=0.3, axis='y')

# 4. Gráfico de dispersão para o melhor modelo
best_idx = results_df['r2_test'].idxmax()
best_degree = results_df.loc[best_idx, 'degree']
best_result = results[best_idx]

plt.subplot(2, 2, 4)
plt.scatter(X_test, y_test, alpha=0.5, label='Dados Reais')
plt.scatter(X_test, best_result['y_pred_test'], alpha=0.5, color='green', label='Previsões')
plt.scatter(X_test[best_result['anomalies']], y_test[best_result['anomalies']], 
           color='red', label='Anomalias', s=100, alpha=0.7)
plt.xlabel('Consumo (m³)')
plt.ylabel('Valor (R$)')
plt.title(f'Regressão Linear (Grau {best_degree}) - R² = {best_result["r2_test"]:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'linear_regression_comparison.png'))
print(f"Gráfico salvo em: {os.path.join(script_dir, 'linear_regression_comparison.png')}")

# Gerar gráficos de resíduos para o melhor modelo
plt.figure(figsize=(15, 6))

# Histograma dos resíduos
plt.subplot(1, 2, 1)
plt.hist(best_result['residuals_test'], bins=30, alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--')
plt.axvline(x=best_result['anomaly_threshold'], color='green', linestyle='--', 
           label=f'Limiar de Anomalia (±{best_result["anomaly_threshold"]:.2f})')
plt.axvline(x=-best_result['anomaly_threshold'], color='green', linestyle='--')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')
plt.title(f'Distribuição dos Resíduos (Grau {best_degree})')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico de dispersão dos resíduos
plt.subplot(1, 2, 2)
plt.scatter(best_result['y_pred_test'], best_result['residuals_test'], alpha=0.5)
plt.scatter(best_result['y_pred_test'][best_result['anomalies']], 
           best_result['residuals_test'][best_result['anomalies']], 
           color='red', s=100, alpha=0.7, label='Anomalias')
plt.axhline(y=0, color='red', linestyle='--')
plt.axhline(y=best_result['anomaly_threshold'], color='green', linestyle='--')
plt.axhline(y=-best_result['anomaly_threshold'], color='green', linestyle='--')
plt.xlabel('Valor Previsto (R$)')
plt.ylabel('Resíduo')
plt.title(f'Resíduos vs. Valores Previstos (Grau {best_degree})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'linear_regression_residuals.png'))
print(f"Gráfico de resíduos salvo em: {os.path.join(script_dir, 'linear_regression_residuals.png')}")

# Criar gráfico temporal dos resíduos para o melhor modelo
# Criar índice temporal
data_test = data.iloc[y_test.index].copy()
data_test["residual"] = best_result['residuals_test']
data_test["anomaly"] = best_result['anomalies'].astype(int)
data_test["Data"] = pd.to_datetime(data_test["Ano"].astype(str) + "-" + data_test["Mês"].astype(str) + "-01")
data_test = data_test.sort_values("Data")

plt.figure(figsize=(12, 6))
plt.scatter(data_test["Data"][data_test["anomaly"] == 0], 
           data_test["residual"][data_test["anomaly"] == 0], 
           alpha=0.5, label="Normal")
plt.scatter(data_test["Data"][data_test["anomaly"] == 1], 
           data_test["residual"][data_test["anomaly"] == 1], 
           color="red", s=100, label="Anomalia")
plt.axhline(y=0, color="green", linestyle="--", alpha=0.5)
plt.axhline(y=best_result['anomaly_threshold'], color="red", linestyle="--", 
           alpha=0.3, label="±3 Desvios Padrão")
plt.axhline(y=-best_result['anomaly_threshold'], color="red", linestyle="--", alpha=0.3)
plt.title(f"Evolução Temporal dos Resíduos (Grau {best_degree})")
plt.xlabel("Data")
plt.ylabel("Resíduo (R$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'linear_regression_temporal.png'))
print(f"Gráfico temporal salvo em: {os.path.join(script_dir, 'linear_regression_temporal.png')}")

print("\nAnálise concluída!")
