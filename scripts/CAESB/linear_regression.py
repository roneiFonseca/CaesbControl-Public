import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import argparse


def detect_anomalies_by_regression(df, local_name, output_dir):
    """
    Detecta anomalias usando regressão linear entre consumo e valor da conta
    """
    # Preparar os dados
    X = df[["m3"]]
    y = df["R$"]

    # Criar e treinar o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Fazer previsões e calcular resíduos
    df["predicted"] = model.predict(X)
    df["residual"] = df["R$"] - df["predicted"]

    # Detectar anomalias (valores fora de 3 desvios padrão)
    residual_std = np.std(df["residual"])
    df["anomaly"] = df["residual"].apply(
        lambda x: 1 if np.abs(x) > 3 * residual_std else 0
    )

    # Criar o gráfico de regressão
    plt.figure(figsize=(12, 8))
    plt.scatter(df["m3"], df["R$"], label="Dados", alpha=0.5)
    plt.scatter(
        df["m3"][df["anomaly"] == 1],
        df["R$"][df["anomaly"] == 1],
        color="red",
        label="Anomalias",
        s=100,
    )

    # Adicionar linha de regressão
    plt.plot(
        df["m3"],
        df["predicted"],
        color="green",
        label=f"Regressão Linear (R² = {model.score(X, y):.3f})",
    )

    # Configurar o gráfico
    plt.title(f"Detecção de Anomalias por Regressão Linear - {local_name}")
    plt.xlabel("Consumo (m³)")
    plt.ylabel("Valor (R$)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Salvar o gráfico
    safe_filename = "".join(
        c for c in local_name.lower() if c.isalnum() or c in (" ", "_", "-")
    ).replace(" ", "_")
    output_file = os.path.join(output_dir, f"regressao_linear_{safe_filename}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Criar gráfico de evolução temporal dos resíduos
    plot_residuals_timeline(df, local_name, output_dir)

    # Retornar anomalias encontradas
    anomalias = df[df["anomaly"] == 1].sort_values("residual", key=abs, ascending=False)
    return anomalias, model.score(X, y)


def plot_residuals_timeline(df, local_name, output_dir):
    """
    Plota a evolução temporal dos resíduos (desvios do valor esperado)
    """
    # Criar índice temporal
    df["Data"] = pd.to_datetime(
        df["Ano"].astype(str) + "-" + df["Mês"].astype(str) + "-01"
    )
    df = df.sort_values("Data")

    # Calcular limites para as bandas de desvio padrão
    residual_std = np.std(df["residual"])
    mean_residual = np.mean(df["residual"])

    # Criar o gráfico
    plt.figure(figsize=(12, 6))

    # Plotar bandas de desvio padrão
    plt.axhline(
        y=mean_residual, color="green", linestyle="--", alpha=0.5, label="Média"
    )
    plt.axhline(
        y=mean_residual + 3 * residual_std,
        color="red",
        linestyle="--",
        alpha=0.3,
        label="±3 Desvios Padrão",
    )
    plt.axhline(
        y=mean_residual - 3 * residual_std, color="red", linestyle="--", alpha=0.3
    )

    # Plotar resíduos
    plt.scatter(
        df["Data"][df["anomaly"] == 0],
        df["residual"][df["anomaly"] == 0],
        alpha=0.5,
        label="Normal",
    )
    plt.scatter(
        df["Data"][df["anomaly"] == 1],
        df["residual"][df["anomaly"] == 1],
        color="red",
        s=100,
        label="Anomalia",
    )

    # Configurar o gráfico
    plt.title(f"Evolução Temporal dos Desvios - {local_name}")
    plt.xlabel("Data")
    plt.ylabel("Desvio do Valor Esperado (R$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Ajustar layout
    plt.tight_layout()

    # Salvar o gráfico
    safe_filename = "".join(
        c for c in local_name.lower() if c.isalnum() or c in (" ", "_", "-")
    ).replace(" ", "_")
    output_file = os.path.join(output_dir, f"residuos_temporais_{safe_filename}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description="Detectar anomalias usando regressão linear"
    )
    parser.add_argument(
        "--local",
        type=str,
        help="Nome do local específico para análise (use aspas para nomes com espaços)",
        nargs="+",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="graficos",
        help="Diretório para salvar os gráficos (padrão: graficos)",
    )
    args = parser.parse_args()

    # Carregar dados
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(script_dir, "./CAESB/contas_para_importar.csv")

    try:
        # Verificar se o arquivo existe
        if not os.path.exists(csv_path):
            print(f"Erro: Arquivo não encontrado: {csv_path}")
            return

        # Carregar o CSV
        df = pd.read_csv(csv_path)

        # Converter valores monetários de string para float
        if "R$" in df.columns:
            df["R$"] = df["R$"].str.replace(".", "").str.replace(",", ".").astype(float)

        # Verificar colunas necessárias
        required_columns = {"LOCAL", "m3", "R$"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            print(f"Erro: Colunas obrigatórias ausentes no CSV: {missing_columns}")
            print("Colunas encontradas:", df.columns.tolist())
            return

        # Criar diretório de saída se não existir
        os.makedirs(args.output_dir, exist_ok=True)

        if args.local:
            # Análise para um local específico
            local_name = " ".join(args.local)
            local_df = df[df["LOCAL"] == local_name].copy()

            if local_df.empty:
                print(f"Local '{local_name}' não encontrado. Locais disponíveis:")
                print("\n".join(sorted(df["LOCAL"].unique())))
                return

            anomalias, r2_score = detect_anomalies_by_regression(
                local_df, local_name, args.output_dir
            )

            # Imprimir resultados
            print(f"\nResultados para {local_name}:")
            print(f"R² Score: {r2_score:.3f}")
            print(f"Total de medições: {len(local_df)}")
            print(
                f"Anomalias detectadas: {len(anomalias)} ({len(anomalias)/len(local_df)*100:.1f}%)"
            )

            if not anomalias.empty:
                print("\nAnomalias detectadas (ordenadas por desvio):")
                for _, row in anomalias.iterrows():
                    desvio = row["R$"] - row["predicted"]
                    print(f"Data: {row['Ano']}-{row['Mês']:02d}")
                    print(f"Consumo: {row['m3']:.1f} m³")
                    print(f"Valor Real: R$ {row['R$']:.2f}")
                    print(f"Valor Previsto: R$ {row['predicted']:.2f}")
                    print(f"Desvio: R$ {desvio:.2f}")
                    print("---")

        else:
            # Análise para todos os locais
            locais = sorted(df["LOCAL"].unique())
            print(f"Analisando {len(locais)} locais...")

            resultados = []
            for local in locais:
                local_df = df[df["LOCAL"] == local].copy()
                if len(local_df) >= 12:  # Apenas locais com pelo menos 12 medições
                    anomalias, r2_score = detect_anomalies_by_regression(
                        local_df, local, args.output_dir
                    )
                    if not anomalias.empty:
                        resultados.append(
                            {
                                "local": local,
                                "total_medicoes": len(local_df),
                                "anomalias": len(anomalias),
                                "r2_score": r2_score,
                            }
                        )

            if resultados:
                print("\nResumo dos resultados:")
                for resultado in sorted(
                    resultados, key=lambda x: x["anomalias"], reverse=True
                ):
                    print(f"\n{resultado['local']}:")
                    print(f"R² Score: {resultado['r2_score']:.3f}")
                    print(f"Total de medições: {resultado['total_medicoes']}")
                    print(
                        f"Anomalias detectadas: {resultado['anomalias']} "
                        f"({resultado['anomalias']/resultado['total_medicoes']*100:.1f}%)"
                    )
            else:
                print("Nenhuma anomalia detectada nos locais analisados.")

    except Exception as e:
        print(f"Erro ao processar os dados: {str(e)}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
