from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("properties/", views.property_list, name="property_list"),
    path(
        "properties/upload-csv/",
        views.upload_properties_csv,
        name="upload_properties_csv",
    ),
    path("bills/upload-csv/", views.upload_bills_csv, name="upload_bills_csv"),
    path("properties/add/", views.property_create, name="property_create"),
    path("properties/<int:pk>/", views.property_detail, name="property_detail"),
    path("properties/<int:pk>/edit/", views.property_update, name="property_update"),
    path("bills/add/", views.bill_create, name="bill_create"),
    path(
        "bills/add/<int:property_pk>/",
        views.bill_create,
        name="bill_create_for_property",
    ),
    path("bills/<int:pk>/edit/", views.bill_update, name="bill_update"),
    path("bills/export_bills_csv/", views.export_bills_csv, name="export_bills_csv"),
    path("search-properties/", views.search_properties, name="search_properties"),
    path(
        "api/explain-anomalies/",
        views.explain_anomalies_api,
        name="explain_anomalies_api",
    ),
]
