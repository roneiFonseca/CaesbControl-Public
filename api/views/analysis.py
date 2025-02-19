from datetime import datetime

import pandas as pd
from django.conf import settings
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from bills.models import Property, WaterBill
from bills.views import detect_anomalies, detect_regression_anomalies

from ..llm import explain_bills_with_LLM


def validate_year_month(year, month=None):
    """
    Valida se o ano e mês fornecidos são válidos.

    Args:
        year (int): Ano a ser validado
        month (int, optional): Mês a ser validado

    Returns:
        tuple: (bool, str) - (é_válido, mensagem_de_erro)
    """
    current_year = datetime.now().year

    # Converter para inteiro
    try:
        year = int(year)
        if month is not None:
            month = int(month)
    except (TypeError, ValueError):
        return False, "Ano e mês devem ser números inteiros"

    # Validar ano
    if year < 1900 or year > current_year:
        return False, f"Ano deve estar entre 1900 e {current_year}"

    # Validar mês se fornecido
    if month is not None:
        if month < 1 or month > 12:
            return False, "Mês deve estar entre 1 e 12"

        # Se o ano for o atual, o mês não pode ser futuro
        if year == current_year and month > datetime.now().month:
            return False, "Não é possível analisar meses futuros"

    return True, ""


@swagger_auto_schema(
    method="get",
    tags=["Análises"],
    operation_description="Analisa e explica anomalias em contas de água para uma unidade de consumo em um determinado período.",
    manual_parameters=[
        openapi.Parameter(
            "property_id",
            openapi.IN_PATH,
            description="ID da unidade de consumo",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
        openapi.Parameter(
            "year",
            openapi.IN_PATH,
            description="Ano a ser analisado",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
    ],
    responses={
        200: openapi.Response(
            description="Análise de anomalias bem-sucedida",
            examples={
                "application/json": {
                    "message": "string",
                    "anomalies": ["string"],
                    "explanation": "string",
                }
            },
        ),
        400: "Dados inválidos ou erro na análise",
        403: "Sem permissão para acessar",
        404: "Unidade de consumo não encontrada",
    },
)
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def explain_bill_anomalies(request, property_id, year):
    """
    Analisa e explica anomalias para uma unidade de consumo em um determinado período.
    Se o mês não for especificado como query parameter, analisa o ano inteiro.
    """
    try:
        # Verificar permissões
        property = Property.objects.get(pk=property_id)
        if (
            not request.user.has_perm("bills.view_all_properties")
            and property.owner != request.user
        ):
            return Response(
                {"error": "Você não tem permissão para ver esta unidade de consumo"},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Obter mês do query parameter
        month = request.GET.get("month")

        # Validar ano e mês
        is_valid, error_message = validate_year_month(year, month)
        if not is_valid:
            return Response(
                {"error": error_message},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Filtrar contas pelo período
        bills_query = WaterBill.objects.filter(property=property, bill_date__year=year)
        if month:
            bills_query = bills_query.filter(bill_date__month=int(month))
            period_name = f"{month}/{year}"
        else:
            period_name = str(year)

        bills = bills_query.order_by("bill_date")

        if not bills.exists():
            return Response(
                {"error": f"Nenhuma conta encontrada para o período {period_name}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Obter histórico de contas anteriores para comparação
        historical_bills = WaterBill.objects.filter(
            property=property, bill_date__lt=bills.first().bill_date
        ).order_by("-bill_date")[:12]  # 12 meses anteriores

        # Preparar dados para análise
        current_bills_data = list(bills.values("bill_date", "consumption", "amount"))
        historical_data = list(
            historical_bills.values("bill_date", "consumption", "amount")
        )
        df = pd.DataFrame(historical_data + current_bills_data)

        if not df.empty:
            df["bill_date"] = pd.to_datetime(df["bill_date"])
            # Converter Decimal para float
            df["consumption"] = df["consumption"].astype(float)
            df["amount"] = df["amount"].astype(float)

            # Detectar anomalias
            anomalies_if, scores = detect_anomalies(df)
            anomalies_reg, residuals, r2_score = detect_regression_anomalies(df)

            # Preparar dados do período atual
            current_bills_info = []
            start_idx = len(historical_data)  # Índice onde começam as contas atuais
            for i, bill in enumerate(current_bills_data, start=start_idx):
                current_bills_info.append(
                    {
                        "bill_date": df["bill_date"].iloc[i].strftime("%Y-%m"),
                        "consumption": float(df["consumption"].iloc[i]),
                        "amount": float(df["amount"].iloc[i]),
                        "is_anomaly_if": bool(anomalies_if[i]),
                        "anomaly_score": float(scores[i]),
                        "is_anomaly_reg": bool(anomalies_reg[i]),
                        "residual": float(residuals[i]) if len(residuals) > 0 else 0,
                    }
                )

            # Preparar histórico para comparação
            historical_info = []
            for i in range(len(historical_data)):
                historical_info.append(
                    {
                        "bill_date": df["bill_date"].iloc[i].strftime("%Y-%m"),
                        "consumption": float(df["consumption"].iloc[i]),
                        "amount": float(df["amount"].iloc[i]),
                    }
                )

            # Gerar explicação usando LLM
            explanation = explain_bills_with_LLM(
                {
                    "period": period_name,
                    "current_bills": current_bills_info,
                    "historical_data": historical_info,
                    "r2_score": r2_score,
                },
                # valores padrão definidos em caesb_control/settings.py
                model_type=getattr(settings, "LLM_MODEL_TYPE", "gpt"),
                model_name=getattr(settings, "LLM_MODEL_NAME", None),
            )

            return Response(
                {
                    "property": property.name,
                    "period": period_name,
                    "bills": current_bills_info,
                    "explanation": explanation,
                }
            )

        return Response(
            {"error": "Dados históricos insuficientes para análise"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Property.DoesNotExist:
        return Response(
            {"error": "Unidade de consumo não encontrada"},
            status=status.HTTP_404_NOT_FOUND,
        )
    except Exception as e:
        return Response(
            {"error": f"Erro ao analisar anomalias: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@swagger_auto_schema(
    method="get",
    tags=["Análises"],
    operation_description="Analisa anomalias em contas de água usando Isolation Forest para uma unidade de consumo em um determinado período.",
    manual_parameters=[
        openapi.Parameter(
            "property_id",
            openapi.IN_PATH,
            description="ID da unidade de consumo",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
        openapi.Parameter(
            "year",
            openapi.IN_PATH,
            description="Ano a ser analisado",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
        openapi.Parameter(
            "month",
            openapi.IN_PATH,
            description="Mês a ser analisado",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
    ],
    responses={
        200: openapi.Response(
            description="Análise de anomalias bem-sucedida",
            examples={
                "application/json": {
                    "message": "string",
                    "anomalies": ["string"],
                }
            },
        ),
        400: "Dados inválidos ou erro na análise",
        403: "Sem permissão para acessar",
        404: "Unidade de consumo não encontrada",
    },
)
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def analyze_isolation_forest(request, property_id, year, month):
    """
    Analisa anomalias usando Isolation Forest para uma unidade de consumo em um determinado período.
    """
    try:
        # Verificar permissões
        property = Property.objects.get(pk=property_id)
        if (
            not request.user.has_perm("bills.view_all_properties")
            and property.owner != request.user
        ):
            return Response(
                {"error": "Você não tem permissão para ver esta unidade de consumo"},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Validar ano e mês
        is_valid, error_message = validate_year_month(year, month)
        if not is_valid:
            return Response(
                {"error": error_message},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Filtrar contas pelo período
        bills_query = WaterBill.objects.filter(
            property=property, bill_date__year=year, bill_date__month=month
        )
        period_name = f"{month}/{year}"

        bills = bills_query.order_by("bill_date")

        if not bills.exists():
            return Response(
                {"error": f"Nenhuma conta encontrada para o período {period_name}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Obter histórico de contas anteriores para comparação
        historical_bills = WaterBill.objects.filter(
            property=property, bill_date__lt=bills.first().bill_date
        ).order_by("-bill_date")[:12]  # 12 meses anteriores

        # Preparar dados para análise
        current_bills_data = list(bills.values("bill_date", "consumption", "amount"))
        historical_data = list(
            historical_bills.values("bill_date", "consumption", "amount")
        )
        df = pd.DataFrame(historical_data + current_bills_data)

        if not df.empty:
            df["bill_date"] = pd.to_datetime(df["bill_date"])
            # Converter Decimal para float
            df["consumption"] = df["consumption"].astype(float)
            df["amount"] = df["amount"].astype(float)

            # Detectar anomalias
            anomalies_if, scores = detect_anomalies(df)

            # Preparar dados do período atual
            current_bills_info = []
            start_idx = len(historical_data)  # Índice onde começam as contas atuais
            for i, bill in enumerate(current_bills_data, start=start_idx):
                current_bills_info.append(
                    {
                        "bill_date": df["bill_date"].iloc[i].strftime("%Y-%m"),
                        "consumption": float(df["consumption"].iloc[i]),
                        "amount": float(df["amount"].iloc[i]),
                        "is_anomaly": bool(anomalies_if[i]),
                        "anomaly_score": float(scores[i]),
                    }
                )

            return Response(
                {
                    "property": property.name,
                    "period": period_name,
                    "bills": current_bills_info,
                }
            )

        return Response(
            {"error": "Dados históricos insuficientes para análise"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Property.DoesNotExist:
        return Response(
            {"error": "Unidade de consumo não encontrada"},
            status=status.HTTP_404_NOT_FOUND,
        )
    except Exception as e:
        return Response(
            {"error": f"Erro ao analisar anomalias: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@swagger_auto_schema(
    method="get",
    tags=["Análises"],
    operation_description="Analisa anomalias em contas de água usando Regressão para uma unidade de consumo em um determinado período.",
    manual_parameters=[
        openapi.Parameter(
            "property_id",
            openapi.IN_PATH,
            description="ID da unidade de consumo",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
        openapi.Parameter(
            "year",
            openapi.IN_PATH,
            description="Ano a ser analisado",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
        openapi.Parameter(
            "month",
            openapi.IN_PATH,
            description="Mês a ser analisado",
            type=openapi.TYPE_INTEGER,
            required=True,
        ),
    ],
    responses={
        200: openapi.Response(
            description="Análise de anomalias bem-sucedida",
            examples={
                "application/json": {
                    "message": "string",
                    "anomalies": ["string"],
                }
            },
        ),
        400: "Dados inválidos ou erro na análise",
        403: "Sem permissão para acessar",
        404: "Unidade de consumo não encontrada",
    },
)
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def analyze_regression(request, property_id, year, month):
    """
    Analisa anomalias usando Regressão para uma unidade de consumo em um determinado período.
    """
    try:
        # Verificar permissões
        property = Property.objects.get(pk=property_id)
        if (
            not request.user.has_perm("bills.view_all_properties")
            and property.owner != request.user
        ):
            return Response(
                {"error": "Você não tem permissão para ver esta unidade de consumo"},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Validar ano e mês
        is_valid, error_message = validate_year_month(year, month)
        if not is_valid:
            return Response(
                {"error": error_message},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Filtrar contas pelo período
        bills_query = WaterBill.objects.filter(
            property=property, bill_date__year=year, bill_date__month=month
        )
        period_name = f"{month}/{year}"

        bills = bills_query.order_by("bill_date")

        if not bills.exists():
            return Response(
                {"error": f"Nenhuma conta encontrada para o período {period_name}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Obter histórico de contas anteriores para comparação
        historical_bills = WaterBill.objects.filter(
            property=property, bill_date__lt=bills.first().bill_date
        ).order_by("-bill_date")[:12]  # 12 meses anteriores

        # Preparar dados para análise
        current_bills_data = list(bills.values("bill_date", "consumption", "amount"))
        historical_data = list(
            historical_bills.values("bill_date", "consumption", "amount")
        )
        df = pd.DataFrame(historical_data + current_bills_data)

        if not df.empty:
            df["bill_date"] = pd.to_datetime(df["bill_date"])
            # Converter Decimal para float
            df["consumption"] = df["consumption"].astype(float)
            df["amount"] = df["amount"].astype(float)

            # Detectar anomalias por regressão
            anomalies_reg, residuals, r2_score = detect_regression_anomalies(df)

            # Preparar dados do período atual
            current_bills_info = []
            start_idx = len(historical_data)  # Índice onde começam as contas atuais
            for i, bill in enumerate(current_bills_data, start=start_idx):
                current_bills_info.append(
                    {
                        "bill_date": df["bill_date"].iloc[i].strftime("%Y-%m"),
                        "consumption": float(df["consumption"].iloc[i]),
                        "amount": float(df["amount"].iloc[i]),
                        "is_anomaly": bool(anomalies_reg[i]),
                        "residual": float(residuals[i]) if len(residuals) > 0 else 0,
                    }
                )

            return Response(
                {
                    "property": property.name,
                    "period": period_name,
                    "r2_score": float(r2_score),
                    "bills": current_bills_info,
                }
            )

        return Response(
            {"error": "Dados históricos insuficientes para análise"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Property.DoesNotExist:
        return Response(
            {"error": "Unidade de consumo não encontrada"},
            status=status.HTTP_404_NOT_FOUND,
        )
    except Exception as e:
        return Response(
            {"error": f"Erro ao analisar anomalias: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
