from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from drf_yasg.utils import swagger_auto_schema
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

from bills.models import WaterBill

from ..serializers.water_bill import WaterBillSerializer


class WaterBillFilter(filters.FilterSet):
    property = filters.NumberFilter()
    year = filters.NumberFilter()
    month = filters.NumberFilter()

    class Meta:
        model = WaterBill
        fields = ["property", "year", "month"]


@swagger_auto_schema(tags=["Contas de Água"])
class WaterBillViewSet(viewsets.ModelViewSet):
    """
    API endpoint que permite visualizar e editar contas de água.
    """

    queryset = WaterBill.objects.all()
    serializer_class = WaterBillSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_class = WaterBillFilter

    def get_queryset(self):
        """
        Filtra as contas para mostrar apenas as das unidades do usuário atual,
        a menos que o usuário tenha permissão para ver todas.
        """
        if getattr(self, "swagger_fake_view", False):  # Detecta se é chamada do Swagger
            return WaterBill.objects.none()  # Retorna queryset vazio para o Swagger

        if self.request.user.is_staff or self.request.user.is_superuser:
            return WaterBill.objects.all()
        return WaterBill.objects.filter(property__owner=self.request.user)

    @swagger_auto_schema(tags=["Contas de Água"])
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Contas de Água"])
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Contas de Água"])
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Contas de Água"])
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Contas de Água"])
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Contas de Água"])
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
