from rest_framework import serializers

from bills.models import Property


class PropertySerializer(serializers.ModelSerializer):
    """
    Serializer para o modelo Property.
    """

    class Meta:
        model = Property
        fields = "__all__"
