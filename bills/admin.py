from django.contrib import admin
from .models import Property, WaterBill, Consumption


@admin.register(Property)
class PropertyAdmin(admin.ModelAdmin):
    list_display = ("name", "registration_number", "owner", "created_at")
    list_filter = ("owner",)
    search_fields = ("name", "registration_number")
    date_hierarchy = "created_at"


@admin.register(WaterBill)
class WaterBillAdmin(admin.ModelAdmin):
    list_display = (
        "property",
        "bill_date",
        "due_date",
        "consumption",
        "amount",
        "status",
    )
    list_filter = ("status", "property", "bill_date")
    search_fields = ("property__name", "property__registration_number")
    date_hierarchy = "bill_date"
    readonly_fields = ("created_at", "updated_at")


@admin.register(Consumption)
class ConsumptionAdmin(admin.ModelAdmin):
    list_display = ("property", "date", "consumption", "average_daily_consumption")
    list_filter = ("property", "date")
    search_fields = ("property__name",)
    date_hierarchy = "date"
