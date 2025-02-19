from django.urls import include, path
from rest_framework import routers

from .views.analysis import (
    analyze_isolation_forest,
    analyze_regression,
    explain_bill_anomalies,
)
from .views.property import PropertyViewSet
from .views.water_bill import WaterBillViewSet

router = routers.DefaultRouter()
router.register(r"properties", PropertyViewSet)
router.register(r"bills", WaterBillViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path(
        "explain-bill/v1/<int:property_id>/<int:year>/",
        explain_bill_anomalies,
        name="explain_bill_anomalies",
    ),
    path(
        "v1/properties/<int:property_id>/anomalies/isolation-forest/<int:year>/<int:month>/",
        analyze_isolation_forest,
        name="analyze_isolation_forest",
    ),
    path(
        "v1/properties/<int:property_id>/anomalies/regression/<int:year>/<int:month>/",
        analyze_regression,
        name="analyze_regression",
    ),
]
