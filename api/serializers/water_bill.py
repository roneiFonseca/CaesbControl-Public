from rest_framework import serializers

from bills.models import WaterBill


class WaterBillSerializer(serializers.ModelSerializer):
    """
    Serializer para o modelo WaterBill.
    """

    class Meta:
        model = WaterBill
        fields = "__all__"
