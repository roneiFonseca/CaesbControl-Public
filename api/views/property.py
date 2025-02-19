from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from drf_yasg.utils import swagger_auto_schema
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

from bills.models import Property

from ..serializers.property import PropertySerializer


class PropertyFilter(filters.FilterSet):
    name = filters.CharFilter(lookup_expr="icontains")
    owner = filters.NumberFilter()

    class Meta:
        model = Property
        fields = ["name", "owner"]


@swagger_auto_schema(tags=["Unidades de Consumo"])
class PropertyViewSet(viewsets.ModelViewSet):
    """
    API endpoint que permite visualizar e editar unidades de consumo.
    """

    queryset = Property.objects.all()
    serializer_class = PropertySerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_class = PropertyFilter

    @swagger_auto_schema(tags=["Unidades de Consumo"])
    def get_queryset(self):
        """
        Filtra as unidades de consumo para mostrar apenas as do usuário atual,
        a menos que o usuário tenha permissão para ver todas.
        """
        if getattr(self, "swagger_fake_view", False):  # Detecta se é chamada do Swagger
            return Property.objects.none()  # Retorna queryset vazio para o Swagger

        if self.request.user.is_staff or self.request.user.is_superuser:
            return Property.objects.all()
        return Property.objects.filter(owner=self.request.user)

    @swagger_auto_schema(tags=["Unidades de Consumo"])
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Unidades de Consumo"])
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Unidades de Consumo"])
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Unidades de Consumo"])
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Unidades de Consumo"])
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    @swagger_auto_schema(tags=["Unidades de Consumo"])
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
