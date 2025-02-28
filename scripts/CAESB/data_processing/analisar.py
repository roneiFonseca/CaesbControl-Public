import os

import pandas as pd

# Carregar o arquivo CSV
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "dados_com_labels.csv"))

# Carregar as colunas "LOCAL", "Ano", "Mês", "R$", "m3", "anormal_reais", "anormal_m3"
data = data[
    ["LOCAL", "Ano", "Mês", "R$", "m3", "anormal_reais", "anormal_m3", "anormal"]
]


print(data.head(48))
