from django import forms
from .models import WaterBill, Property


class WaterBillForm(forms.ModelForm):
    class Meta:
        model = WaterBill
        fields = [
            "property",
            "bill_date",
            "due_date",
            "consumption",
            "amount",
            "meter_reading",
            "status",
            "bill_image",
            "bill_pdf",
            "notes",
        ]
        widgets = {
            "bill_date": forms.DateInput(attrs={"type": "date"}),
            "due_date": forms.DateInput(attrs={"type": "date"}),
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        if user:
            self.fields["property"].queryset = Property.objects.filter(owner=user)


class PropertyForm(forms.ModelForm):
    class Meta:
        model = Property
        fields = ["name", "registration_number", "hidrometer_number"]


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Selecione o arquivo CSV",
        help_text="O arquivo deve conter as colunas: LOCAL, INSCRIÇÃO, HIDROMETRO",
    )


class BillsCSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Selecione o arquivo CSV",
        help_text="O arquivo deve conter as colunas: Ano, Mês, LOCAL, INSCRIÇÃO, HIDROMETRO, LEITURA ANTERIOR, LEITURA ATUAL, CONSUMO (m3), VALOR (R$)",
    )


class DashboardFilterForm(forms.Form):
    property_search = forms.CharField(
        required=False,
        label="Buscar Local",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Digite o nome do local..."}
        ),
    )
    year = forms.MultipleChoiceField(
        required=False,
        label="Anos",
        widget=forms.SelectMultiple(attrs={"class": "form-select"}),
        choices=[],
        help_text="Pressione Ctrl (Cmd no Mac) para selecionar múltiplos anos",
    )
    month = forms.IntegerField(
        required=False,
        label="Mês",
        widget=forms.Select(
            choices=[
                ("", "Todos"),
                (1, "Janeiro"),
                (2, "Fevereiro"),
                (3, "Março"),
                (4, "Abril"),
                (5, "Maio"),
                (6, "Junho"),
                (7, "Julho"),
                (8, "Agosto"),
                (9, "Setembro"),
                (10, "Outubro"),
                (11, "Novembro"),
                (12, "Dezembro"),
            ]
        ),
    )
    property = forms.ModelChoiceField(
        queryset=Property.objects.none(),
        required=False,
        label="Local",
        empty_label="Todos",
    )

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            # Configurar anos disponíveis
            years = (
                WaterBill.objects.filter(property__owner=user)
                .dates("bill_date", "year")
                .values_list("bill_date__year", flat=True)
            )
            self.fields["year"].choices = [(str(year), str(year)) for year in years]

            # Configurar propriedades do usuário
            self.fields["property"].queryset = Property.objects.filter(owner=user)
