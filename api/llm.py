import os

import openai
import requests
from dotenv import load_dotenv

from api.utils import setup_logger

# Criando uma instancia de logger
logger = setup_logger(__name__, "llm_module.log")


def call_ollama(messages, model="llama3"):
    """
    Faz uma chamada para o Ollama API.

    Args:
        messages (list): Lista de mensagens no formato do chat
        model (str): Nome do modelo Ollama a ser usado

    Returns:
        str: Resposta do modelo
    """
    logger.info("Iniciando a chamada do LLM ollama")
    # Converter mensagens do formato ChatGPT para o formato Ollama
    prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n\n"
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
        return response.json().get("response", "")
    except Exception as e:
        logger.error(f"Erro ao chamar Ollama: {str(e)}")
        return f"Erro ao chamar Ollama: {str(e)}"


def call_gpt(messages, model="gpt-4o-mini"):
    """
    Faz uma chamada para a API do OpenAI GPT.

    Args:
        messages (list): Lista de mensagens no formato do chat
        model (str): Nome do modelo GPT a ser usado

    Returns:
        str: Resposta do modelo
    """
    logger.info("Iniciando a chamada do GPT")
    try:
        # Configurar a chave da API
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("Erro: OPENAI_API_KEY não encontrada no arquivo .env")
            return "Erro: OPENAI_API_KEY não encontrada no arquivo .env"

        openai.api_key = openai_api_key

        # Fazer a chamada para a API do GPT
        response = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=0.7, max_tokens=500
        )

        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Erro ao chamar GPT: {str(e)}")
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
        return call_gpt(messages, model=model_name or "gpt-4")
    elif model_type == "ollama":
        return call_ollama(messages, model=model_name or "llama3")
    else:
        logger.error(f"Erro: Tipo de modelo '{model_type}' não suportado")
        return f"Erro: Tipo de modelo '{model_type}' não suportado"


def explain_bills_with_LLM(data, model_type="gpt", model_name=None):
    """
    Gera uma explicação em linguagem natural sobre as anomalias de um conjunto de contas.

    Args:
        data (dict): Dicionário contendo os dados das contas
        model_type (str): Tipo de modelo ('gpt' ou 'ollama')
        model_name (str): Nome específico do modelo (opcional)

    Returns:
        str: Explicação gerada pelo modelo
    """
    logger.info("Iniciando a chamada do LLM para explicação")

    messages = [
        {
            "role": "system",
            "content": (
                "Você é um especialista em análise de consumo de água. "
                "Analise os dados fornecidos e explique de forma clara e objetiva "
                "se há anomalias no consumo ou valor das contas, comparando com o histórico."
            ),
        },
        {
            "role": "user",
            "content": f"""
            Analise as contas de água para o período {data["period"]}:

            Contas do Período:
            {
                chr(10).join(
                    [
                        f"- {bill['bill_date']}: {bill['consumption']:.1f} m³ (R$ {bill['amount']:.2f})"
                        f" | Anomalia IF: {'Sim' if bill['is_anomaly_if'] else 'Não'}"
                        f" | Score: {bill['anomaly_score']:.3f}"
                        f" | Anomalia Reg: {'Sim' if bill['is_anomaly_reg'] else 'Não'}"
                        f" | Desvio: {bill['residual']:.3f}"
                        for bill in data["current_bills"]
                    ]
                )
            }

            Histórico de Consumo (do mais recente ao mais antigo):
            {
                chr(10).join(
                    [
                        f"- {bill['bill_date']}: {bill['consumption']:.1f} m³ (R$ {bill['amount']:.2f})"
                        for bill in data["historical_data"]
                    ]
                )
            }

            R² do modelo de regressão: {data["r2_score"]:.3f}

            Forneça:
            1. Uma análise do consumo no período em relação ao histórico
            2. Explicação das anomalias detectadas (se houver)
            3. Possíveis causas para variações significativas
            4. Recomendações para o usuário
            """,
        },
    ]

    return get_llm_response(messages, model_type=model_type, model_name=model_name)
