from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from django.utils.translation import gettext_lazy as _


class Property(models.Model):
    """Model representing a property that receives water bills."""

    name = models.CharField(_("Nome"), max_length=200)
    registration_number = models.CharField(
        _("Número de Inscrição"), max_length=50, unique=True
    )
    hidrometer_number = models.CharField(
        _("Número do Hidrômetro"), max_length=50, unique=True
    )
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="properties")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _("Imóvel")
        verbose_name_plural = _("Imóveis")
        ordering = ["name"]
        permissions = [
            ("view_all_properties", "Can view all properties"),
            ("view_all_bills", "Can view all water bills"),
        ]

    def __str__(self):
        return f"{self.name} ({self.registration_number})"


class WaterBill(models.Model):
    """Model representing a water bill."""

    STATUS_CHOICES = [
        ("pending", _("Pendente")),
        ("paid", _("Pago")),
        ("overdue", _("Vencido")),
    ]

    property = models.ForeignKey(
        Property,
        on_delete=models.CASCADE,
        related_name="water_bills",
        verbose_name=_("Imóvel"),
    )
    bill_date = models.DateField(_("Data da Conta"))
    due_date = models.DateField(_("Data de Vencimento"))
    consumption = models.DecimalField(
        _("Consumo (m³)"),
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
    )
    amount = models.DecimalField(
        _("Valor (R$)"),
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
    )
    meter_reading = models.DecimalField(
        _("Leitura do Hidrômetro"),
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
    )
    status = models.CharField(
        _("Status"), max_length=10, choices=STATUS_CHOICES, default="pending"
    )
    bill_image = models.ImageField(
        _("Imagem da Conta"), upload_to="bills/%Y/%m/", null=True, blank=True
    )
    bill_pdf = models.FileField(
        _("PDF da Conta"), upload_to="bills/%Y/%m/", null=True, blank=True
    )
    notes = models.TextField(_("Observações"), blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _("Conta de Água")
        verbose_name_plural = _("Contas de Água")
        ordering = ["-bill_date"]

    def __str__(self):
        return f"{self.property.name} - {self.bill_date}"


class Consumption(models.Model):
    """Model for tracking historical consumption patterns."""

    property = models.ForeignKey(
        Property, on_delete=models.CASCADE, related_name="consumption_history"
    )
    date = models.DateField()
    consumption = models.DecimalField(
        max_digits=10, decimal_places=2, validators=[MinValueValidator(0)]
    )
    average_daily_consumption = models.DecimalField(
        max_digits=10, decimal_places=2, validators=[MinValueValidator(0)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date"]
        unique_together = ["property", "date"]

    def __str__(self):
        return f"{self.property.name} - {self.date}"
