import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configurar o estilo dos gráficos
plt.style.use("default")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Definir o diretório para salvar os gráficos
GRAPHICS_DIR = os.path.join(os.path.dirname(__file__), "graficos")

# Carregar os dados
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "dados_limpos2.csv"))

# Ordenar os dados por LOCAL, Ano e Mês para garantir a sequência temporal correta
data = data.sort_values(["LOCAL", "Ano", "Mês"])

# === Análise para R$ ===
# Calcular média móvel e desvio padrão para 6 meses
data["media_6_meses_reais"] = data.groupby("LOCAL")["R$"].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)
data["desvio_padrao_6_meses_reais"] = data.groupby("LOCAL")["R$"].transform(
    lambda x: x.rolling(window=6, min_periods=1).std()
)

# Calcular limites superior e inferior (média ± desvio padrão)
data["limite_superior_6_meses_reais"] = (
    data["media_6_meses_reais"] + data["desvio_padrao_6_meses_reais"]
)
data["limite_inferior_6_meses_reais"] = (
    data["media_6_meses_reais"] - data["desvio_padrao_6_meses_reais"]
)

# Identificar valores anormais (fora do intervalo média ± desvio padrão)
data["anormal_6_meses_reais"] = np.where(
    (data["R$"] > data["limite_superior_6_meses_reais"])
    | (data["R$"] < data["limite_inferior_6_meses_reais"]),
    1,
    0,
)

# Calcular média móvel e desvio padrão para 12 meses
data["media_12_meses_reais"] = data.groupby("LOCAL")["R$"].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()
)
data["desvio_padrao_12_meses_reais"] = data.groupby("LOCAL")["R$"].transform(
    lambda x: x.rolling(window=12, min_periods=1).std()
)

# Calcular limites superior e inferior (média ± desvio padrão)
data["limite_superior_12_meses_reais"] = (
    data["media_12_meses_reais"] + data["desvio_padrao_12_meses_reais"]
)
data["limite_inferior_12_meses_reais"] = (
    data["media_12_meses_reais"] - data["desvio_padrao_12_meses_reais"]
)

# Identificar valores anormais (fora do intervalo média ± desvio padrão)
data["anormal_12_meses_reais"] = np.where(
    (data["R$"] > data["limite_superior_12_meses_reais"])
    | (data["R$"] < data["limite_inferior_12_meses_reais"]),
    1,
    0,
)

# Identificar quando o consumo é anormal em ambos os períodos
data["anormal_reais"] = np.where(
    (data["anormal_6_meses_reais"]) & (data["anormal_12_meses_reais"]),
    1,
    0,
)

# === Análise para m3 ===
# Calcular média móvel e desvio padrão para 6 meses
data["media_6_meses_m3"] = data.groupby("LOCAL")["m3"].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)
data["desvio_padrao_6_meses_m3"] = data.groupby("LOCAL")["m3"].transform(
    lambda x: x.rolling(window=6, min_periods=1).std()
)

# Calcular limites superior e inferior (média ± desvio padrão)
data["limite_superior_6_meses_m3"] = (
    data["media_6_meses_m3"] + data["desvio_padrao_6_meses_m3"]
)
data["limite_inferior_6_meses_m3"] = (
    data["media_6_meses_m3"] - data["desvio_padrao_6_meses_m3"]
)

# Identificar valores anormais (fora do intervalo média ± desvio padrão)
data["anormal_6_meses_m3"] = np.where(
    (data["m3"] > data["limite_superior_6_meses_m3"])
    | (data["m3"] < data["limite_inferior_6_meses_m3"]),
    1,
    0,
)

# Calcular média móvel e desvio padrão para 12 meses
data["media_12_meses_m3"] = data.groupby("LOCAL")["m3"].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()
)
data["desvio_padrao_12_meses_m3"] = data.groupby("LOCAL")["m3"].transform(
    lambda x: x.rolling(window=12, min_periods=1).std()
)

# Calcular limites superior e inferior (média ± desvio padrão)
data["limite_superior_12_meses_m3"] = (
    data["media_12_meses_m3"] + data["desvio_padrao_12_meses_m3"]
)
data["limite_inferior_12_meses_m3"] = (
    data["media_12_meses_m3"] - data["desvio_padrao_12_meses_m3"]
)

# Identificar valores anormais (fora do intervalo média ± desvio padrão)
data["anormal_12_meses_m3"] = np.where(
    (data["m3"] > data["limite_superior_12_meses_m3"])
    | (data["m3"] < data["limite_inferior_12_meses_m3"]),
    1,
    0,
)

# Identificar quando o consumo é anormal em ambos os períodos
data["anormal_m3"] = np.where(
    (data["anormal_6_meses_m3"]) & (data["anormal_12_meses_m3"]),
    1,
    0,
)

# Arredondar valores para melhor visualização
colunas_para_arredondar = [
    # Colunas R$
    "media_6_meses_reais",
    "desvio_padrao_6_meses_reais",
    "limite_superior_6_meses_reais",
    "limite_inferior_6_meses_reais",
    "media_12_meses_reais",
    "desvio_padrao_12_meses_reais",
    "limite_superior_12_meses_reais",
    "limite_inferior_12_meses_reais",
    # Colunas m3
    "media_6_meses_m3",
    "desvio_padrao_6_meses_m3",
    "limite_superior_6_meses_m3",
    "limite_inferior_6_meses_m3",
    "media_12_meses_m3",
    "desvio_padrao_12_meses_m3",
    "limite_superior_12_meses_m3",
    "limite_inferior_12_meses_m3",
]
for coluna in colunas_para_arredondar:
    data[coluna] = data[coluna].round(2)


# Identificar quando o consumo é anormal em R$ ou m3
data["anormal"] = np.where(
    (data["anormal_reais"]) | (data["anormal_m3"]),
    1,
    0,
)

print("\nPrimeiras linhas dos dados com análise de anomalias (R$ e m3):")
print(
    data[
        [
            "LOCAL",
            "Ano",
            "Mês",
            # Dados originais
            "R$",
            "m3",
            # Análise R$
            "media_6_meses_reais",
            "limite_inferior_6_meses_reais",
            "limite_superior_6_meses_reais",
            "anormal_6_meses_reais",
            "media_12_meses_reais",
            "limite_inferior_12_meses_reais",
            "limite_superior_12_meses_reais",
            "anormal_12_meses_reais",
            "anormal_reais",
            # Análise m3
            "media_6_meses_m3",
            "limite_inferior_6_meses_m3",
            "limite_superior_6_meses_m3",
            "anormal_6_meses_m3",
            "media_12_meses_m3",
            "limite_inferior_12_meses_m3",
            "limite_superior_12_meses_m3",
            "anormal_12_meses_m3",
            "anormal_m3",
            # Análise combinada
            "anormal",
        ]
    ].head(48)
)

# salvar em csv
data.to_csv(
    os.path.join(os.path.dirname(__file__), "dados_com_labels.csv"), index=False
)
