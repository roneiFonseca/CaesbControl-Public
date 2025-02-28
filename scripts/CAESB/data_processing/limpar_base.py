import pandas as pd
import os
import locale

# Configurar locale para português
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")

# Mapeamento de meses em português para números
MONTH_MAP = {
    "JANEIRO": 1,
    "FEVEREIRO": 2,
    "MARÇO": 3,
    "ABRIL": 4,
    "MAIO": 5,
    "JUNHO": 6,
    "JULHO": 7,
    "AGOSTO": 8,
    "SETEMBRO": 9,
    "OUTUBRO": 10,
    "NOVEMBRO": 11,
    "DEZEMBRO": 12,
}


def clean_value(value):
    """Limpa valores monetários mantendo a vírgula como separador decimal"""
    if pd.isna(value) or value == "":
        return "0,00"
    if isinstance(value, str):
        # Remove R$ e espaços
        value = value.replace("R$", "").strip()
        # Se já tem vírgula, retorna como está
        if "," in value:
            return value
    # Para valores numéricos, converte para string com vírgula
    return f"{value:.2f}".replace(".", ",")


# Definir o caminho absoluto para o arquivo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "dados_raw.csv")

# Ler o arquivo CSV com as configurações corretas
data = pd.read_csv(DATA_FILE, encoding="utf-8")

print("Iniciando limpeza dos dados...")

# 1. Limpeza básica
print("1. Realizando limpeza básica...")
# remover hifen da coluna INSCRIÇÃO
data["INSCRIÇÃO"] = data["INSCRIÇÃO"].str.replace("-", "")

# converter para maiusculo as colunas mes e local
for col in ["Mês", "LOCAL"]:
    if col in ["Mês", "LOCAL"]:
        data[col] = data[col].str.upper()

# remover espacos antes e depois de todas as colunas
for col in data.columns:
    data[col] = data[col].map(lambda x: x.strip() if isinstance(x, str) else x)

# 2. Padronização de nomes de locais
print("2. Padronizando nomes de locais...")
# Identificar inscrições que têm mais de um nome de local
duplicates = data.groupby("INSCRIÇÃO")["LOCAL"].agg(list).reset_index()
duplicates["num_names"] = duplicates["LOCAL"].apply(len)
duplicates = duplicates[duplicates["num_names"] > 1]

if not duplicates.empty:
    print("\nInscrições com múltiplos nomes de local:")
    for _, row in duplicates.iterrows():
        print(f"\nInscrição: {row['INSCRIÇÃO']}")
        print(f"Nomes encontrados: {row['LOCAL']}")

    # Criar um dicionário de padronização
    name_mapping = {}
    for _, row in duplicates.iterrows():
        names = row["LOCAL"]
        # Usar o nome mais comum ou o primeiro como padrão
        standard_name = max(names, key=names.count)
        for name in names:
            name_mapping[name] = standard_name

    # Aplicar a padronização no DataFrame principal
    data["LOCAL"] = data["LOCAL"].map(lambda x: name_mapping.get(x, x))

# 3. Limpeza e processamento dos dados numéricos
print("3. Processando dados numéricos...")
# Converter mês de texto para número
data["Mês"] = data["Mês"].map(MONTH_MAP)

# Limpar coluna de hidrômetro (pegar apenas o primeiro número)
data["HIDROMETRO"] = data["HIDROMETRO"].str.split().str[0]

# Remover espaços extras da coluna m3 e renomear
data[" m3"] = data[" m3"].fillna(0)
data = data.rename(columns={" m3": "m3"})

# Preencher valores vazios
data["LEITURA ANTERIOR"] = data["LEITURA ANTERIOR"].fillna(0)
data["LEITURA ATUAL"] = data["LEITURA ATUAL"].fillna(0)
data["m3"] = data["m3"].fillna(0)
data["R$"] = data["R$"].fillna("R$ 0,00")

# Limpar e converter valores
data["R$"] = data["R$"].apply(clean_value)

# Converter valores numéricos (exceto R$)
data["m3"] = pd.to_numeric(data["m3"], errors="coerce").fillna(0).astype(int)
data["LEITURA ANTERIOR"] = (
    pd.to_numeric(data["LEITURA ANTERIOR"], errors="coerce").fillna(0).astype(int)
)
data["LEITURA ATUAL"] = (
    pd.to_numeric(data["LEITURA ATUAL"], errors="coerce").fillna(0).astype(int)
)

# Converter R$ para float
data["R$"] = data["R$"].str.replace("R$", "").str.replace(",", ".").astype(float)


# 4. Ordenar os dados
print("4. Ordenando dados...")
data = data.sort_values(by=["LOCAL", "Mês", "INSCRIÇÃO"])

# 5. Salvar dados limpos
print("5. Salvando arquivo de dados limpos...")
DADOS_LIMPOS = os.path.join(SCRIPT_DIR, "dados_limpos.csv")
data.to_csv(DADOS_LIMPOS, index=False, encoding="utf-8")
print("Arquivo dados_limpos.csv gerado com sucesso!")

# 6. Gerar arquivo de locais únicos a partir dos dados limpos
print("6. Gerando arquivo de locais únicos...")
locais = data.drop_duplicates(subset=["INSCRIÇÃO"], keep="first")
locais = locais[["LOCAL", "INSCRIÇÃO", "HIDROMETRO"]]

# Salvar arquivo de locais únicos
LOCAIS_UNICOS = os.path.join(SCRIPT_DIR, "locais_unicos.csv")
locais.to_csv(LOCAIS_UNICOS, index=False, encoding="utf-8")
print("Arquivo locais_unicos.csv gerado com sucesso!")

# Mostrar estatísticas finais
print("\nProcessamento concluído!")
print(f"Total de registros: {len(data)}")
print(f"Total de locais únicos: {len(locais)}")
