import numpy as np
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import argparse


def preprocess_data(data):
    """Pré-processa os dados usando StandardScaler."""
    scaler = StandardScaler()
    X = data.values.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def detect_anomalies(data, contamination=0.1, labeled_anomalies=None, random_state=42):
    """Detecta anomalias em uma série de dados usando Isolation Forest.

    Args:
        data: Série de dados para análise
        contamination: Proporção esperada de anomalias (0.0 a 0.5)
        labeled_anomalies: Dados rotulados para avaliação (opcional)
        random_state: Semente aleatória para reprodutibilidade
    """
    # Pré-processamento dos dados
    X_scaled, _ = preprocess_data(data)

    # Treinar o modelo Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,  # Número de árvores
        max_samples="auto",  # Tamanho da subamostra
    )

    # Fazer predições (-1 para anomalias, 1 para normais)
    predictions = model.fit_predict(X_scaled)

    # Converter predições para formato binário (0: normal, 1: anomalia)
    anomalies = np.where(predictions == -1, 1, 0)

    # Calcular métricas se houver dados rotulados
    metrics = {}
    if labeled_anomalies is not None:
        metrics["precision"] = precision_score(labeled_anomalies, anomalies)
        metrics["recall"] = recall_score(labeled_anomalies, anomalies)
        metrics["f1"] = f1_score(labeled_anomalies, anomalies)

        # Adicionar score de anomalia
        anomaly_scores = -model.score_samples(X_scaled)
        metrics["anomaly_scores"] = anomaly_scores

    return anomalies, metrics


def evaluate_model(true_labels, predictions):
    """Avalia o modelo usando métricas padrão."""
    results = {
        "precision": precision_score(true_labels, predictions),
        "recall": recall_score(true_labels, predictions),
        "f1": f1_score(true_labels, predictions),
    }
    return results


def plot_anomalies(df, title, output_file, metric="m3"):
    """Plota o gráfico de consumo ou valor com anomalias."""
    try:
        # Criar diretório de saída se não existir
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Limpar qualquer figura anterior
        plt.clf()

        # Definir rótulos baseado na métrica
        ylabel = "Consumo (m³)" if metric == "m3" else "Valor (R$)"

        # Criar nova figura
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Data"], df[metric], label=ylabel.split(" ")[0])
        ax.scatter(
            df["Data"][df["anomaly"] == 1],
            df[metric][df["anomaly"] == 1],
            color="red",
            label="Anomalias",
        )
        ax.set_title(title)
        ax.set_xlabel("Data")
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()

        # Formatar eixo Y para valores monetários se necessário
        if metric != "m3":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R$ {x:,.2f}"))

        # Salvar e fechar a figura
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Gráfico salvo com sucesso em: {output_file}")
    except Exception as e:
        print(f"Erro ao salvar o gráfico: {str(e)}")
    finally:
        # Garantir que todas as figuras sejam fechadas
        plt.close("all")


def main():
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description="Detectar anomalias no consumo de água usando Isolation Forest"
    )
    parser.add_argument(
        "--local",
        type=str,
        help="Nome do local específico para análise (use aspas para nomes com espaços)",
        nargs="+",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Proporção esperada de anomalias (0.0 a 0.5, padrão: 0.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="graficos",
        help="Diretório para salvar os gráficos (padrão: graficos)",
    )
    parser.add_argument(
        "--labeled-data",
        type=str,
        help="Caminho para arquivo CSV com dados rotulados para avaliação",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["m3", "valor"],
        default="m3",
        help="Métrica para análise: m3 (consumo) ou valor (R$)",
    )
    args = parser.parse_args()

    # Validar contamination
    if not 0 < args.contamination <= 0.5:
        print("Erro: contamination deve estar entre 0 e 0.5")
        return

    # Carregar dados
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "contas_para_importar.csv")

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
        required_columns = {"Ano", "Mês", "LOCAL", "m3"}
        if args.metric == "valor":
            required_columns.add("R$")

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            print(f"Erro: Colunas obrigatórias ausentes no CSV: {missing_columns}")
            print("Colunas encontradas:", df.columns.tolist())
            return

        # Criar índice temporal
        df["Data"] = pd.to_datetime(
            df["Ano"].astype(str) + "-" + df["Mês"].astype(str) + "-01"
        )
        df = df.sort_values("Data")

        # Definir a coluna métrica e rótulos
        metric_col = "m3" if args.metric == "m3" else "R$"
        metric_label = "consumo" if args.metric == "m3" else "valor"
        # metric_unit = "m³" if args.metric == "m3" else "R$"

        # Carregar dados rotulados se fornecidos
        labeled_anomalies = None
        if args.labeled_data and os.path.exists(args.labeled_data):
            labeled_df = pd.read_csv(args.labeled_data)
            labeled_anomalies = labeled_df["anomaly"].values

        if args.local:
            # Juntar as palavras do nome do local
            local_name = " ".join(args.local)
            # Análise para um local específico
            local_df = df[df["LOCAL"] == local_name].copy()
            if local_df.empty:
                print(f"Local '{local_name}' não encontrado. Locais disponíveis:")
                print("\n".join(sorted(df["LOCAL"].unique())))
                return

            # Detectar anomalias com Isolation Forest
            anomalies, metrics = detect_anomalies(
                local_df[metric_col],
                contamination=args.contamination,
                labeled_anomalies=labeled_anomalies
                if labeled_anomalies is not None
                else None,
                random_state=args.random_state,
            )
            local_df["anomaly"] = anomalies

            # Adicionar scores de anomalia se disponíveis
            if "anomaly_scores" in metrics:
                local_df["anomaly_score"] = metrics["anomaly_scores"]

            # Sanitizar nome do arquivo
            safe_filename = "".join(
                c for c in local_name.lower() if c.isalnum() or c in (" ", "_", "-")
            ).replace(" ", "_")
            output_file = os.path.join(
                script_dir,
                args.output_dir,
                f"anomalias_{safe_filename}_{args.metric}.png",
            )

            plot_anomalies(
                local_df,
                f"Detecção de Anomalias de {metric_label.title()} - {local_name}",
                output_file,
                metric=metric_col,
            )

            # Imprimir estatísticas
            anomalias = local_df[local_df["anomaly"] == 1]
            print(f"\nEstatísticas para {local_name}:")
            print(f"Total de medições: {len(local_df)}")
            print(
                f"Anomalias detectadas: {len(anomalias)} ({len(anomalias)/len(local_df)*100:.1f}%)"
            )

            # Imprimir métricas de avaliação se disponíveis
            if metrics:
                print("\nMétricas de Avaliação:")
                print(f"Precisão: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1-Score: {metrics['f1']:.3f}")

            print(
                f"\nDatas com {metric_label} anômalo (ordenado por score de anomalia):"
            )
            anomalias_df = anomalias.copy()
            if "anomaly_score" in anomalias_df.columns:
                anomalias_df = anomalias_df.sort_values(
                    "anomaly_score", ascending=False
                )

            for _, row in anomalias_df.iterrows():
                score_info = (
                    f", Score: {row['anomaly_score']:.3f}"
                    if "anomaly_score" in row
                    else ""
                )
                valor_info = (
                    f"{row[metric_col]:.1f}"
                    if args.metric == "m3"
                    else f"R$ {row[metric_col]:.2f}"
                )
                print(
                    f"Data: {row['Data'].strftime('%Y-%m')}, {metric_label.title()}: {valor_info}{score_info}"
                )

        else:
            # Análise geral - detectar anomalias por local
            locais = sorted(df["LOCAL"].unique())
            all_anomalies = []
            all_metrics = []

            for local in locais:
                local_df = df[df["LOCAL"] == local].copy()
                if len(local_df) >= 12:  # Apenas locais com pelo menos 12 medições
                    anomalies, metrics = detect_anomalies(
                        local_df[metric_col],
                        contamination=args.contamination,
                        labeled_anomalies=labeled_anomalies
                        if labeled_anomalies is not None
                        else None,
                        random_state=args.random_state,
                    )
                    local_df["anomaly"] = anomalies

                    # Adicionar scores de anomalia se disponíveis
                    if "anomaly_scores" in metrics:
                        local_df["anomaly_score"] = metrics["anomaly_scores"]

                    if metrics:
                        metrics["local"] = local
                        all_metrics.append(metrics)
                    if any(anomalies == 1):
                        all_anomalies.append(local_df)

            if all_anomalies:
                # Concatenar todos os resultados
                result_df = pd.concat(all_anomalies)

                # Sanitizar nome do arquivo para todos os locais
                output_file = os.path.join(
                    script_dir,
                    args.output_dir,
                    f"anomalias_todos_locais_{args.metric}.png",
                )

                plot_anomalies(
                    result_df,
                    f"Detecção de Anomalias de {metric_label.title()} - Todos os Locais",
                    output_file,
                    metric=metric_col,
                )

                # Imprimir estatísticas gerais
                print("\nEstatísticas Gerais:")
                print(f"Total de locais analisados: {len(locais)}")
                locais_com_anomalias = len(
                    set(result_df[result_df["anomaly"] == 1]["LOCAL"])
                )
                print(
                    f"Locais com anomalias: {locais_com_anomalias} ({locais_com_anomalias/len(locais)*100:.1f}%)"
                )

                # Imprimir métricas médias se disponíveis
                if all_metrics:
                    print("\nMétricas Médias de Avaliação:")
                    avg_metrics = {
                        "precision": np.mean([m["precision"] for m in all_metrics]),
                        "recall": np.mean([m["recall"] for m in all_metrics]),
                        "f1": np.mean([m["f1"] for m in all_metrics]),
                    }
                    print(f"Precisão Média: {avg_metrics['precision']:.3f}")
                    print(f"Recall Médio: {avg_metrics['recall']:.3f}")
                    print(f"F1-Score Médio: {avg_metrics['f1']:.3f}")

                print(f"\nAnomalias de {metric_label} detectadas por local:")
                for local in sorted(set(result_df[result_df["anomaly"] == 1]["LOCAL"])):
                    local_anomalies = result_df[
                        (result_df["LOCAL"] == local) & (result_df["anomaly"] == 1)
                    ]
                    if "anomaly_score" in local_anomalies.columns:
                        local_anomalies = local_anomalies.sort_values(
                            "anomaly_score", ascending=False
                        )

                    print(f"\n{local}:")
                    for _, row in local_anomalies.iterrows():
                        score_info = (
                            f", Score: {row['anomaly_score']:.3f}"
                            if "anomaly_score" in row
                            else ""
                        )
                        valor_info = (
                            f"{row[metric_col]:.1f}"
                            if args.metric == "m3"
                            else f"R$ {row[metric_col]:.2f}"
                        )
                        print(
                            f"Data: {row['Data'].strftime('%Y-%m')}, {metric_label.title()}: {valor_info}{score_info}"
                        )

    except Exception as e:
        print(f"Erro ao processar os dados: {str(e)}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
