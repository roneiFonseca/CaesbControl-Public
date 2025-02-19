from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Sum
from django.http import JsonResponse, HttpResponse
from .models import Property, WaterBill
from .forms import (
    WaterBillForm,
    PropertyForm,
    CSVUploadForm,
    BillsCSVUploadForm,
    DashboardFilterForm,
)
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import csv
import io
import datetime
from django.conf import settings


def detect_anomalies(df, contamination=0.05):
    """
    Detecta anomalias usando Isolation Forest
    """
    # Preparar os dados
    features = ["consumption", "amount"]
    X = df[features].values

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treinar o modelo
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.score_samples(X_scaled)

    return anomalies == -1, scores


def detect_regression_anomalies(df, std_threshold=3):
    """
    Detecta anomalias usando regressão linear
    """
    # Preparar os dados com nomes de features explícitos
    X = pd.DataFrame(df["consumption"].values, columns=["consumption"])
    y = df["amount"].values

    # Treinar o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Calcular resíduos
    predictions = model.predict(X)
    residuals = y - predictions

    # Detectar anomalias (valores fora de n desvios padrão)
    residual_std = np.std(residuals)
    anomalies = np.abs(residuals) > (std_threshold * residual_std)

    return anomalies, residuals, model.score(X, y)


def call_ollama(messages, model="llama3"):
    """
    Faz uma chamada para o Ollama API.

    Args:
        messages (list): Lista de mensagens no formato do chat
        model (str): Nome do modelo Ollama a ser usado

    Returns:
        str: Resposta do modelo
    """
    import requests

    # Converter mensagens do formato ChatGPT para o formato Ollama
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"Instructions: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"

    # Preparar a requisição para o Ollama
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        return f"Erro ao chamar Ollama: {str(e)}"


def call_gpt(messages, model="gpt-4"):
    """
    Faz uma chamada para a API do OpenAI GPT.

    Args:
        messages (list): Lista de mensagens no formato do chat
        model (str): Nome do modelo GPT a ser usado

    Returns:
        str: Resposta do modelo
    """
    try:
        from dotenv import load_dotenv
        import openai
        import os

        # Configurar a chave da API
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "Erro: OPENAI_API_KEY não encontrada no arquivo .env"

        openai.api_key = openai_api_key

        # Fazer a chamada para a API do GPT
        response = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=0.7, max_tokens=500
        )

        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Erro ao chamar GPT: {str(e)}"


def get_llm_response(messages, model_type="gpt", model_name=None):
    """
    Obtém resposta do modelo de linguagem escolhido.

    Args:
        messages (list): Lista de mensagens no formato do chat
        model_type (str): Tipo de modelo ('gpt' ou 'ollama')
        model_name (str): Nome específico do modelo (opcional)

    Returns:
        str: Resposta do modelo
    """
    if model_type == "gpt":
        return call_gpt(messages, model=model_name or "gpt-4o-mini")
    elif model_type == "ollama":
        return call_ollama(messages, model=model_name or "llama3")
    else:
        return f"Erro: Tipo de modelo '{model_type}' não suportado"


def explain_anomalies_with_LLM(anomalies_data, model_type="gpt", model_name=None):
    """
    Gera uma explicação em linguagem natural das anomalias detectadas usando o modelo escolhido.

    Args:
        anomalies_data (dict): Dicionário contendo informações sobre as anomalias detectadas.
        model_type (str): Tipo de modelo a ser usado ('gpt' ou 'ollama')
        model_name (str): Nome específico do modelo (opcional)

    Returns:
        str: Explicação em texto das anomalias.
    """
    try:
        # Preparar os dados das anomalias para o prompt
        isolation_forest_anomalies = anomalies_data.get(
            "isolation_forest_anomalies", []
        )
        regression_anomalies = anomalies_data.get("regression_anomalies", [])
        all_records = anomalies_data.get("all_records", [])

        # Formatar os dados para o prompt
        formatted_if_anomalies = []
        for anomaly in isolation_forest_anomalies:
            formatted_if_anomalies.append(
                f"Data: {anomaly['bill_date']}, "
                f"Consumo: {anomaly['consumption']:.1f}m³, "
                f"Valor: R${anomaly['amount']:.2f}, "
                f"Score: {anomaly.get('anomaly_score', 0):.3f}"
            )

        formatted_reg_anomalies = []
        for anomaly in regression_anomalies:
            formatted_reg_anomalies.append(
                f"Data: {anomaly['bill_date']}, "
                f"Consumo: {anomaly['consumption']:.1f}m³, "
                f"Valor: R${anomaly['amount']:.2f}, "
                f"Desvio: {anomaly.get('residuals', 0):.3f}"
            )

        formatted_all_records = []
        for record in all_records:
            formatted_all_records.append(
                f"Data: {record['bill_date']}, "
                f"Consumo: {record['consumption']:.1f}m³, "
                f"Valor: R${record['amount']:.2f}"
            )

        # Preparar o contexto para o modelo
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um especialista em análise de dados com foco em consumo sustentável de recursos hídricos. "
                    "Utilize o histórico completo de consumo, resultados da regressão linear e do Isolation Forest para criar uma análise detalhada. "
                    "Concentre-se em identificar padrões históricos, tendências sazonais, correlacionar dados relevantes e sugerir ações práticas para mitigar ou resolver os problemas detectados."
                ),
            },
            {
                "role": "user",
                "content": f"""
        Aqui estão os dados de consumo de água. Forneça uma análise aprofundada considerando o histórico completo e as anomalias detectadas:

        Histórico completo de consumo (ordenado por data):
        {formatted_all_records if formatted_all_records else 'Nenhum registro encontrado'}

        Anomalias detectadas pelo Isolation Forest (consumos que fogem do padrão geral):
        {formatted_if_anomalies if formatted_if_anomalies else 'Nenhuma anomalia detectada'}

        Anomalias detectadas pela Regressão Linear (desvios significativos da tendência esperada):
        {formatted_reg_anomalies if formatted_reg_anomalies else 'Nenhuma anomalia detectada'}

        Forneça uma análise detalhada que inclua:
        1. Visão geral do histórico de consumo:
           - Padrões sazonais ou tendências ao longo do tempo
           - Média e variação típica do consumo
           - Períodos de maior e menor consumo

        2. Análise das anomalias:
           - Identificação de padrões nos dados anômalos
           - Relação entre as anomalias e o histórico geral
           - Possíveis causas (vazamentos, consumo inesperado, erro de medição)

        3. Recomendações práticas:
           - Sugestões específicas baseadas nos padrões identificados
           - Ações preventivas para evitar anomalias futuras
           - Medidas para otimização do consumo

        4. Insights adicionais:
           - Correlações importantes identificadas
           - Métricas ou análises adicionais recomendadas
           - Pontos de atenção para monitoramento futuro

        Certifique-se de apresentar suas conclusões de forma clara, lógica e orientada para a resolução de problemas.
        """,
            },
        ]

        # Fazer a chamada para o modelo escolhido
        response = get_llm_response(
            messages, model_type=model_type, model_name=model_name
        )

        # Retornar a explicação gerada
        return response.strip()

    except Exception as e:
        return f"Erro ao gerar explicação: {str(e)}"


@login_required
def dashboard(request):
    """Display the main dashboard with consumption statistics and charts."""
    # Check if user has permission to view all properties
    if request.user.has_perm("bills.view_all_properties"):
        properties = Property.objects.all()
        bills_queryset = WaterBill.objects.all()
    else:
        properties = Property.objects.filter(owner=request.user)
        bills_queryset = WaterBill.objects.filter(property__owner=request.user)

    # Initialize filter form
    filter_form = DashboardFilterForm(request.GET, user=request.user)

    # Apply filters
    if filter_form.is_valid():
        # Verificar se os filtros mudaram
        current_filters = {
            "property_search": filter_form.cleaned_data.get("property_search", ""),
            "property": filter_form.cleaned_data.get("property", None).id
            if filter_form.cleaned_data.get("property")
            else None,
            "year": sorted(filter_form.cleaned_data.get("year", [])),
            "month": filter_form.cleaned_data.get("month", None),
        }

        if current_filters != request.session.get("previous_filters", {}):
            # Limpar explicação anterior apenas se os filtros mudaram
            if "anomalies_explanation" in request.session:
                del request.session["anomalies_explanation"]
            request.session["previous_filters"] = current_filters

        if filter_form.cleaned_data["property_search"]:
            search_term = filter_form.cleaned_data["property_search"]
            bills_queryset = bills_queryset.filter(
                property__name__icontains=search_term
            )
        if filter_form.cleaned_data["property"]:
            bills_queryset = bills_queryset.filter(
                property=filter_form.cleaned_data["property"]
            )
        if filter_form.cleaned_data["year"]:
            bills_queryset = bills_queryset.filter(
                bill_date__year__in=filter_form.cleaned_data["year"]
            )
        if filter_form.cleaned_data["month"]:
            bills_queryset = bills_queryset.filter(
                bill_date__month=filter_form.cleaned_data["month"]
            )

    # Get recent bills
    recent_bills = bills_queryset.order_by("-bill_date")[:5]

    # Calculate statistics
    total_consumption = bills_queryset.aggregate(total=Sum("consumption"))["total"] or 0

    avg_consumption = bills_queryset.aggregate(avg=Avg("consumption"))["avg"] or 0

    # Calcular valor total e médio
    total_amount = bills_queryset.aggregate(total=Sum("amount"))["total"] or 0

    avg_amount = bills_queryset.aggregate(avg=Avg("amount"))["avg"] or 0

    # Prepare data for charts
    bills_data = bills_queryset.values(
        "bill_date", "consumption", "amount", "property__name"
    )

    # Converter dados para float antes de criar o DataFrame
    bills_data_float = []
    for bill in bills_data:
        bill_dict = {
            "bill_date": bill["bill_date"],
            "consumption": float(bill["consumption"]),
            "amount": float(bill["amount"]),
            "property__name": bill["property__name"],
        }
        bills_data_float.append(bill_dict)

    df = pd.DataFrame(bills_data_float)
    consumption_chart = None
    cost_chart = None
    top_consumers_chart = None
    isolation_forest_chart = None
    regression_anomalies_chart = None

    if not df.empty:
        if "bill_date" in df.columns:
            df["bill_date"] = pd.to_datetime(df["bill_date"])
            df["year"] = df["bill_date"].dt.year

        # Detectar anomalias
        df["consumption"] = df["consumption"].astype(float)
        df["amount"] = df["amount"].astype(float)

        # Isolation Forest
        anomalies_if, scores_if = detect_anomalies(df)
        df["anomaly_if"] = anomalies_if
        df["anomaly_score"] = scores_if

        # Regressão Linear
        anomalies_reg, residuals, r2_score = detect_regression_anomalies(df)
        df["anomaly_reg"] = anomalies_reg
        df["residuals"] = residuals

        # Gráfico de consumo por mês
        consumption_df = (
            df.groupby("bill_date")
            .agg(
                {
                    "consumption": "sum",
                }
            )
            .reset_index()
        )

        # Calcular média anual de consumo - soma por ano dividido pelo número de meses únicos
        yearly_consumption = (
            df.groupby(["year", df["bill_date"].dt.month])["consumption"]
            .sum()
            .reset_index()
        )
        yearly_avg_consumption = (
            yearly_consumption.groupby("year")
            .agg(
                {
                    "consumption": "mean"  # média dos totais mensais
                }
            )
            .reset_index()
        )

        # Criar figura para consumo
        consumption_fig = go.Figure()

        # Adicionar linha de consumo mensal
        consumption_fig.add_trace(
            go.Scatter(
                x=consumption_df["bill_date"],
                y=consumption_df["consumption"],
                name="Consumo Mensal",
                line=dict(color="#023E73", width=2),
            )
        )

        # Adicionar linhas de média anual
        for _, row in yearly_avg_consumption.iterrows():
            year = row["year"]
            avg = row["consumption"]
            year_data = consumption_df[consumption_df["bill_date"].dt.year == year]
            if not year_data.empty:
                consumption_fig.add_trace(
                    go.Scatter(
                        x=[year_data["bill_date"].min(), year_data["bill_date"].max()],
                        y=[avg, avg],
                        name="Média Anual",
                        line=dict(color="#038C33", width=2, dash="dash"),
                        showlegend=year_data["bill_date"].min()
                        == consumption_df[
                            "bill_date"
                        ].min(),  # Mostra legenda apenas uma vez
                    )
                )

        consumption_fig.update_layout(
            title="Consumo de Água por Mês",
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Data",
            yaxis_title="Consumo (m³)",
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        consumption_chart = consumption_fig.to_html()

        # Gráfico de custo por mês
        cost_df = (
            df.groupby("bill_date")
            .agg(
                {
                    "amount": "sum",
                }
            )
            .reset_index()
        )

        # Calcular média anual de custo - soma por ano dividido pelo número de meses únicos
        yearly_cost = (
            df.groupby(["year", df["bill_date"].dt.month])["amount"].sum().reset_index()
        )
        yearly_avg_cost = (
            yearly_cost.groupby("year")
            .agg(
                {
                    "amount": "mean"  # média dos totais mensais
                }
            )
            .reset_index()
        )

        # Criar figura para custo
        cost_fig = go.Figure()

        # Adicionar linha de custo mensal
        cost_fig.add_trace(
            go.Scatter(
                x=cost_df["bill_date"],
                y=cost_df["amount"],
                name="Custo Mensal",
                line=dict(color="#023E73", width=2),
            )
        )

        # Adicionar linhas de média anual
        for _, row in yearly_avg_cost.iterrows():
            year = row["year"]
            avg = row["amount"]
            year_data = cost_df[cost_df["bill_date"].dt.year == year]
            if not year_data.empty:
                cost_fig.add_trace(
                    go.Scatter(
                        x=[year_data["bill_date"].min(), year_data["bill_date"].max()],
                        y=[avg, avg],
                        name="Média Anual",
                        line=dict(color="#038C33", width=2, dash="dash"),
                        showlegend=year_data["bill_date"].min()
                        == cost_df["bill_date"].min(),  # Mostra legenda apenas uma vez
                    )
                )

        cost_fig.update_layout(
            title="Custo por Mês",
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Data",
            yaxis_title="Valor (R$)",
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        cost_chart = cost_fig.to_html()

        # Gráfico dos 5 maiores consumidores
        top_consumers_df = (
            df.groupby("property__name")
            .agg({"consumption": "sum", "amount": "sum"})
            .reset_index()
            .sort_values("consumption", ascending=False)
            .head(5)
        )

        # Criar figura para maiores consumidores
        top_consumers_fig = go.Figure()

        # Adicionar barras de consumo
        top_consumers_fig.add_trace(
            go.Bar(
                x=top_consumers_df["property__name"],
                y=top_consumers_df["consumption"],
                text=top_consumers_df["consumption"].round(1).astype(str)
                + " m³<br>R$ "
                + top_consumers_df["amount"].round(2).astype(str),
                textposition="auto",
                hovertemplate="Local: %{x}<br>Consumo: %{y:.1f} m³<br>Valor: R$ %{customdata:.2f}<extra></extra>",
                customdata=top_consumers_df["amount"],
                marker_color="#023E73",
            )
        )

        top_consumers_fig.update_layout(
            title="5 Maiores Consumidores",
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Local",
            yaxis_title="Consumo Total (m³)",
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            template="plotly_white",
            showlegend=False,
            xaxis_tickangle=-45,
            bargap=0.3,
            height=500,  # Aumentar altura do gráfico para melhor visualização
        )

        top_consumers_chart = top_consumers_fig.to_html()

        # Gráfico Isolation Forest
        iso_fig = go.Figure()

        # Pontos normais
        iso_fig.add_trace(
            go.Scatter(
                x=df[~df["anomaly_if"]]["consumption"],
                y=df[~df["anomaly_if"]]["amount"],
                mode="markers",
                name="Normal",
                marker=dict(color="#023E73", size=8),
                text=df[~df["anomaly_if"]]["property__name"],
            )
        )

        # Pontos anômalos
        iso_fig.add_trace(
            go.Scatter(
                x=df[df["anomaly_if"]]["consumption"],
                y=df[df["anomaly_if"]]["amount"],
                mode="markers",
                name="Anomalia",
                marker=dict(color="red", size=12, symbol="x"),
                text=df[df["anomaly_if"]]["property__name"],
            )
        )

        iso_fig.update_layout(
            title="Detecção de Anomalias - Isolation Forest",
            title_x=0.5,
            xaxis_title="Consumo (m³)",
            yaxis_title="Valor (R$)",
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        isolation_forest_chart = iso_fig.to_html()

        # Gráfico Regressão Linear
        reg_fig = go.Figure()

        # Pontos normais
        reg_fig.add_trace(
            go.Scatter(
                x=df[~df["anomaly_reg"]]["consumption"],
                y=df[~df["anomaly_reg"]]["amount"],
                mode="markers",
                name="Normal",
                marker=dict(color="#023E73", size=8),
                text=df[~df["anomaly_reg"]]["property__name"],
            )
        )

        # Pontos anômalos
        reg_fig.add_trace(
            go.Scatter(
                x=df[df["anomaly_reg"]]["consumption"],
                y=df[df["anomaly_reg"]]["amount"],
                mode="markers",
                name="Anomalia",
                marker=dict(color="red", size=12, symbol="x"),
                text=df[df["anomaly_reg"]]["property__name"],
            )
        )

        # Linha de regressão
        x_range = np.linspace(df["consumption"].min(), df["consumption"].max(), 100)
        reg_fig.add_trace(
            go.Scatter(
                x=x_range,
                y=LinearRegression()
                .fit(df[["consumption"]], df["amount"])
                .predict(x_range.reshape(-1, 1)),
                mode="lines",
                name=f"Regressão (R² = {r2_score:.3f})",
                line=dict(color="#038C33", dash="dash"),
            )
        )

        reg_fig.update_layout(
            title="Detecção de Anomalias - Regressão Linear",
            title_x=0.5,
            xaxis_title="Consumo (m³)",
            yaxis_title="Valor (R$)",
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        regression_anomalies_chart = reg_fig.to_html()

        # Preparar estatísticas de anomalias
        anomalies_stats = []
        for idx, row in df[df["anomaly_if"] | df["anomaly_reg"]].iterrows():
            anomaly_type = []
            if row["anomaly_if"]:
                anomaly_type.append("Isolation Forest")
            if row["anomaly_reg"]:
                anomaly_type.append("Regressão Linear")

            anomalies_stats.append(
                {
                    "date": row["bill_date"].strftime("%Y-%m"),
                    "property": row["property__name"],
                    "consumption": f"{row['consumption']:.1f}",
                    "amount": f"R$ {row['amount']:.2f}",
                    "type": " + ".join(anomaly_type),
                    "score": f"{row['anomaly_score']:.3f}",
                }
            )

        # Ordenar por score de anomalia
        anomalies_stats.sort(key=lambda x: float(x["score"]))

        # Armazenar dados de anomalias e histórico completo na sessão para uso posterior
        # Converter timestamps para strings antes de armazenar
        all_records = df.to_dict(orient="records")
        isolation_forest_anomalies = df[df["anomaly_if"]].to_dict(orient="records")
        regression_anomalies = df[df["anomaly_reg"]].to_dict(orient="records")

        # Converter timestamps e outros objetos para formatos serializáveis
        # e remover campos desnecessários
        for record in all_records + isolation_forest_anomalies + regression_anomalies:
            if "bill_date" in record:
                record["bill_date"] = record["bill_date"].strftime("%Y-%m-%d")
            # Converter valores numéricos para float nativo do Python
            if "consumption" in record:
                record["consumption"] = float(record["consumption"])
            if "amount" in record:
                record["amount"] = float(record["amount"])
            if "anomaly_score" in record:
                record["anomaly_score"] = float(record["anomaly_score"])
            if "residuals" in record:
                record["residuals"] = float(record["residuals"])
            # Remover o campo property__name
            if "property__name" in record:
                del record["property__name"]

        request.session["anomalies_data"] = {
            "all_records": all_records,
            "isolation_forest_anomalies": isolation_forest_anomalies,
            "regression_anomalies": regression_anomalies,
        }

        context = {
            "filter_form": filter_form,
            "properties": properties,
            "recent_bills": recent_bills,
            "total_consumption": total_consumption,
            "avg_consumption": avg_consumption,
            "total_amount": total_amount,
            "avg_amount": avg_amount,
            "consumption_chart": consumption_chart,
            "cost_chart": cost_chart,
            "top_consumers_chart": top_consumers_chart,
            "isolation_forest_chart": isolation_forest_chart,
            "regression_anomalies_chart": regression_anomalies_chart,
            "anomalies_stats": anomalies_stats,
        }

        return render(request, "bills/dashboard.html", context)

    context = {
        "filter_form": filter_form,
        "properties": properties,
        "recent_bills": recent_bills,
        "total_consumption": total_consumption,
        "avg_consumption": avg_consumption,
        "total_amount": total_amount,
        "avg_amount": avg_amount,
    }

    return render(request, "bills/dashboard.html", context)


@login_required
def explain_anomalies_api(request):
    """Gerar explicação de anomalias usando o modelo configurado."""
    try:
        # Obter dados das anomalias da sessão
        bills_data = request.session.get("anomalies_data", {})
        if not bills_data:
            messages.error(request, "Dados de anomalias não encontrados")
            return redirect(request.META.get("HTTP_REFERER", "dashboard"))

        # Obter configuração do modelo das settings ou usar padrão
        model_type = getattr(settings, "LLM_MODEL_TYPE", "gpt")  # 'gpt' ou 'ollama'
        model_name = getattr(settings, "LLM_MODEL_NAME", None)  # específico do modelo

        # Gerar explicação
        explanation = explain_anomalies_with_LLM(
            bills_data, model_type=model_type, model_name=model_name
        )

        # Atualizar a sessão com a explicação
        request.session["anomalies_explanation"] = explanation
        messages.success(request, "Explicação gerada com sucesso!")

        # Obter a URL anterior e adicionar o hash
        referer_url = request.META.get("HTTP_REFERER", "dashboard")
        if "#" in referer_url:
            referer_url = referer_url.split("#")[0]
        return redirect(f"{referer_url}#anomalies-section")

    except Exception as e:
        messages.error(request, f"Erro ao gerar explicação: {str(e)}")
        return redirect(request.META.get("HTTP_REFERER", "dashboard"))


@login_required
def search_properties(request):
    """API endpoint para buscar locais dinamicamente."""
    search_term = request.GET.get("term", "")

    if request.user.has_perm("bills.view_all_properties"):
        properties = Property.objects.filter(name__icontains=search_term).values(
            "id", "name"
        )[:10]
    else:
        properties = Property.objects.filter(
            owner=request.user, name__icontains=search_term
        ).values("id", "name")[:10]

    results = [
        {"id": p["id"], "value": p["name"], "label": p["name"]} for p in properties
    ]
    return JsonResponse(results, safe=False)


@login_required
def property_list(request):
    """Display list of properties."""
    if request.user.has_perm("bills.view_all_properties"):
        properties = Property.objects.all()
    else:
        properties = Property.objects.filter(owner=request.user)

    return render(request, "bills/property_list.html", {"properties": properties})


@login_required
def property_detail(request, pk):
    """Display property details and its bills."""
    if request.user.has_perm("bills.view_all_properties"):
        property = get_object_or_404(Property, pk=pk)
    else:
        property = get_object_or_404(Property, pk=pk, owner=request.user)

    bills = property.water_bills.all().order_by("-bill_date")
    return render(
        request, "bills/property_detail.html", {"property": property, "bills": bills}
    )


@login_required
def bill_create(request, property_pk=None):
    """Create a new water bill."""
    if request.method == "POST":
        form = WaterBillForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            bill = form.save()
            messages.success(request, "Conta criada com sucesso!")
            return redirect("property_detail", pk=bill.property.pk)
    else:
        initial = {}
        if property_pk:
            property = get_object_or_404(Property, pk=property_pk, owner=request.user)
            initial["property"] = property
        form = WaterBillForm(user=request.user, initial=initial)

    return render(request, "bills/bill_form.html", {"form": form})


@login_required
def bill_update(request, pk):
    """Update an existing water bill."""
    bill = get_object_or_404(WaterBill, pk=pk, property__owner=request.user)

    if request.method == "POST":
        form = WaterBillForm(
            request.POST, request.FILES, instance=bill, user=request.user
        )
        if form.is_valid():
            form.save()
            messages.success(request, "Conta atualizada com sucesso!")
            return redirect("property_detail", pk=bill.property.pk)
    else:
        form = WaterBillForm(instance=bill, user=request.user)

    return render(request, "bills/bill_form.html", {"form": form, "bill": bill})


@login_required
def property_create(request):
    """Create a new property."""
    if request.method == "POST":
        form = PropertyForm(request.POST)
        if form.is_valid():
            property = form.save(commit=False)
            property.owner = request.user
            property.save()

            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse(
                    {"success": True, "message": "Imóvel criado com sucesso!"}
                )

            messages.success(request, "Imóvel criado com sucesso!")
            return redirect("property_list")
        else:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse(
                    {
                        "success": False,
                        "error": "Erro ao criar imóvel. Verifique os dados informados.",
                        "form_errors": form.errors,
                    }
                )
    else:
        form = PropertyForm()

    return render(request, "bills/property_form.html", {"form": form})


@login_required
def property_update(request, pk):
    """Update an existing property."""
    property = get_object_or_404(Property, pk=pk, owner=request.user)

    if request.method == "POST":
        form = PropertyForm(request.POST, instance=property)
        if form.is_valid():
            form.save()
            messages.success(request, "Imóvel atualizado com sucesso!")
            return redirect("property_list")
    else:
        form = PropertyForm(instance=property)

    return render(
        request, "bills/property_form.html", {"form": form, "property": property}
    )


@login_required
def upload_properties_csv(request):
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES["csv_file"]

            # Log do início do processo
            print(f"[CSV Import] Iniciando importação do arquivo: {csv_file.name}")

            # Verifica se é um arquivo CSV
            if not csv_file.name.endswith(".csv"):
                print(f"[CSV Import] Erro: Arquivo {csv_file.name} não é um CSV")
                messages.error(request, "Por favor, envie um arquivo CSV válido.")
                return redirect("upload_properties_csv")

            # Processa o arquivo CSV
            try:
                print("[CSV Import] Tentando decodificar o arquivo...")
                decoded_file = csv_file.read().decode("utf-8")
                io_string = io.StringIO(decoded_file)
                reader = csv.DictReader(io_string)

                # Log das colunas encontradas
                print(f"[CSV Import] Colunas encontradas: {reader.fieldnames}")

                # Contadores para feedback
                success_count = 0
                error_count = 0
                errors = []

                for row_num, row in enumerate(reader, start=1):
                    try:
                        print(f"\n[CSV Import] Processando linha {row_num}")
                        print(
                            f"[CSV Import] Dados da linha: LOCAL='{row.get('LOCAL', '')}', "
                            f"INSCRIÇÃO='{row.get('INSCRIÇÃO', '')}', "
                            f"HIDROMETRO='{row.get('HIDROMETRO', '')}'"
                        )

                        # Validações antes de criar o objeto
                        if not row.get("LOCAL"):
                            raise ValueError("LOCAL está vazio")
                        if not row.get("INSCRIÇÃO"):
                            raise ValueError("INSCRIÇÃO está vazia")
                        if not row.get("HIDROMETRO"):
                            raise ValueError("HIDROMETRO está vazio")

                        # Tenta criar o imóvel
                        try:
                            property = Property(
                                name=row["LOCAL"],
                                registration_number=row["INSCRIÇÃO"],
                                hidrometer_number=row["HIDROMETRO"],
                                owner=request.user,
                            )

                            print(
                                f"[CSV Import] Tentando salvar: name='{property.name}', "
                                f"registration='{property.registration_number}', "
                                f"hidrometer='{property.hidrometer_number}'"
                            )

                            property.save()
                            print(
                                f"[CSV Import] Sucesso: Imóvel ID {property.id} criado"
                            )
                            success_count += 1
                        except Exception as e:
                            if (
                                "UNIQUE constraint failed: bills_property.hidrometer_number"
                                in str(e)
                            ):
                                error_msg = f"Linha {row_num}: O número de hidrômetro '{row['HIDROMETRO']}' já está em uso por outro imóvel"
                            else:
                                error_msg = f"Linha {row_num}: {str(e)}"
                            print(f"[CSV Import] ERRO: {error_msg}")
                            error_count += 1
                            errors.append(error_msg)
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Linha {row_num}: {str(e)}"
                        print(f"[CSV Import] ERRO: {error_msg}")
                        errors.append(error_msg)

                # Feedback para o usuário
                if success_count > 0:
                    messages.success(
                        request, f"{success_count} imóveis importados com sucesso."
                    )
                if error_count > 0:
                    messages.warning(
                        request, f"{error_count} imóveis não puderam ser importados."
                    )
                    for error in errors:
                        messages.error(request, error)

            except Exception as e:
                messages.error(request, f"Erro ao processar o arquivo: {str(e)}")
                return redirect("upload_properties_csv")

            return redirect("property_list")
    else:
        form = CSVUploadForm()

    return render(request, "bills/upload_properties_csv.html", {"form": form})


@login_required
def upload_bills_csv(request):
    if request.method == "POST":
        form = BillsCSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES["csv_file"]

            print(
                f"[Bills CSV Import] Iniciando importação do arquivo: {csv_file.name}"
            )

            if not csv_file.name.endswith(".csv"):
                print(f"[Bills CSV Import] Erro: Arquivo {csv_file.name} não é um CSV")
                messages.error(request, "Por favor, envie um arquivo CSV válido.")
                return redirect("upload_bills_csv")

            try:
                print("[Bills CSV Import] Tentando decodificar o arquivo...")
                decoded_file = csv_file.read().decode("utf-8")
                io_string = io.StringIO(decoded_file)
                reader = csv.DictReader(io_string)

                print(f"[Bills CSV Import] Colunas encontradas: {reader.fieldnames}")

                success_count = 0
                error_count = 0
                errors = []

                for row_num, row in enumerate(reader, start=1):
                    try:
                        print(f"\n[Bills CSV Import] Processando linha {row_num}")
                        print(f"[Bills CSV Import] Dados da linha: {row}")

                        # Validações
                        if not all(
                            key in row
                            for key in [
                                "Ano",
                                "Mês",
                                "LOCAL",
                                "INSCRIÇÃO",
                                "HIDROMETRO",
                                "LEITURA ANTERIOR",
                                "LEITURA ATUAL",
                                "m3",
                                "R$",
                            ]
                        ):
                            raise ValueError("Faltam colunas obrigatórias")

                        # Busca o imóvel pelo número de inscrição e hidrômetro
                        try:
                            property = Property.objects.get(
                                registration_number=row["INSCRIÇÃO"],
                                hidrometer_number=row["HIDROMETRO"],
                            )
                        except Property.DoesNotExist:
                            # Tenta encontrar apenas pelo número de inscrição para dar uma mensagem mais específica
                            try:
                                Property.objects.get(
                                    registration_number=row["INSCRIÇÃO"]
                                )
                                raise ValueError(
                                    f"O número do hidrômetro '{row['HIDROMETRO']}' não corresponde ao cadastrado "
                                    f"para a inscrição '{row['INSCRIÇÃO']}'"
                                )
                            except Property.DoesNotExist:
                                raise ValueError(
                                    f"Imóvel não encontrado com inscrição '{row['INSCRIÇÃO']}' "
                                    f"e hidrômetro '{row['HIDROMETRO']}'"
                                )

                        # Converte valores
                        try:
                            ano = int(row["Ano"])
                            mes = int(row["Mês"])
                            consumo = float(row["m3"].replace(",", "."))
                            valor = float(
                                row["R$"]
                                .replace("R$", "")
                                .replace(".", "")
                                .replace(",", ".")
                                .strip()
                            )
                            # leitura_anterior = float(
                            #     row["LEITURA ANTERIOR"].replace(",", ".")
                            # )
                            leitura_atual = float(
                                row["LEITURA ATUAL"].replace(",", ".")
                            )
                        except ValueError as e:
                            raise ValueError(f"Erro ao converter valores: {str(e)}")

                        # Cria a data da conta (primeiro dia do mês)
                        bill_date = datetime.date(ano, mes, 1)

                        # Cria a data de vencimento (dia 15 do mês seguinte)
                        if mes == 12:
                            due_date = datetime.date(ano + 1, 1, 15)
                        else:
                            due_date = datetime.date(ano, mes + 1, 15)

                        # Cria ou atualiza a conta
                        bill, created = WaterBill.objects.update_or_create(
                            property=property,
                            bill_date=bill_date,
                            defaults={
                                "due_date": due_date,
                                "consumption": consumo,
                                "amount": valor,
                                "meter_reading": leitura_atual,
                                "status": "paid",  # Assumindo que contas históricas já foram pagas
                            },
                        )

                        action = "criada" if created else "atualizada"
                        print(f"[Bills CSV Import] Conta {action} com sucesso: {bill}")
                        success_count += 1

                    except Exception as e:
                        error_count += 1
                        error_msg = f"Linha {row_num}: {str(e)}"
                        print(f"[Bills CSV Import] ERRO: {error_msg}")
                        errors.append(error_msg)

                if success_count > 0:
                    messages.success(
                        request, f"{success_count} contas importadas com sucesso."
                    )
                if error_count > 0:
                    messages.warning(
                        request, f"{error_count} contas não puderam ser importadas."
                    )
                    for error in errors:
                        messages.error(request, error)

            except Exception as e:
                messages.error(request, f"Erro ao processar o arquivo: {str(e)}")
                return redirect("upload_bills_csv")

            return redirect("property_list")
    else:
        form = BillsCSVUploadForm()

    return render(request, "bills/upload_bills_csv.html", {"form": form})


@login_required
def export_bills_csv(request):
    """Exporta todas as contas do usuário em formato CSV"""
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="contas_agua.csv"'

    writer = csv.writer(response)
    writer.writerow(
        [
            "Local",
            "Inscrição",
            "Hidrômetro",
            "Data da Conta",
            "Consumo (m³)",
            "Valor (R$)",
            "Status",
        ]
    )

    if request.user.has_perm("bills.view_all_bills"):
        bills = WaterBill.objects.all()
    else:
        bills = WaterBill.objects.filter(property__owner=request.user)

    for bill in bills:
        writer.writerow(
            [
                bill.property.name,
                bill.property.registration_number,
                bill.property.hidrometer_number,
                bill.bill_date.strftime("%d/%m/%Y"),
                str(bill.consumption).replace(".", ","),
                str(bill.amount).replace(".", ","),
                bill.get_status_display(),
            ]
        )

    return response
