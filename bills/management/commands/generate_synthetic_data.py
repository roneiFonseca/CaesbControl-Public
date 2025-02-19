from django.core.management.base import BaseCommand
from bills.models import Property, WaterBill
from django.contrib.auth.models import User
from datetime import datetime, timedelta
import random


class Command(BaseCommand):
    help = "Gera dados sintéticos para teste"

    def handle(self, *args, **kwargs):
        # Limpa dados existentes
        Property.objects.all().delete()
        WaterBill.objects.all().delete()

        # Criar usuário de teste se não existir
        user, created = User.objects.get_or_create(
            username="admin", defaults={"email": "admin@example.com", "is_staff": True}
        )
        if created:
            user.set_password("admin123")
            user.save()

        # Lista de propriedades fictícias
        properties_data = [
            {
                "name": "Quadra 10 Conjunto A, Casa 15",
                "registration_number": "123456",
                "hidrometer_number": "H123456",
            },
            {
                "name": "Quadra 15 Conjunto B, Casa 22",
                "registration_number": "234567",
                "hidrometer_number": "H234567",
            },
            {
                "name": "Quadra 20 Conjunto C, Casa 8",
                "registration_number": "345678",
                "hidrometer_number": "H345678",
            },
            {
                "name": "Quadra 25 Conjunto D, Casa 45",
                "registration_number": "456789",
                "hidrometer_number": "H456789",
            },
            {
                "name": "Quadra 30 Conjunto E, Casa 33",
                "registration_number": "567890",
                "hidrometer_number": "H567890",
            },
        ]

        # Criar propriedades
        properties = []
        for prop_data in properties_data:
            prop = Property.objects.create(
                name=prop_data["name"],
                registration_number=prop_data["registration_number"],
                hidrometer_number=prop_data["hidrometer_number"],
                owner=user,
            )
            properties.append(prop)
            self.stdout.write(f"Criada propriedade: {prop}")

        # Gerar contas para cada propriedade
        start_date = datetime(2007, 1, 1)
        end_date = datetime(2023, 12, 31)
        current_date = start_date

        while current_date <= end_date:
            for prop in properties:
                # Consumo base entre 10 e 20 m³
                base_consumption = random.uniform(10, 20)

                # Variação sazonal (mais consumo no verão)
                month = current_date.month
                if month in [12, 1, 2]:  # Verão
                    seasonal_factor = random.uniform(1.2, 1.4)
                elif month in [6, 7, 8]:  # Inverno
                    seasonal_factor = random.uniform(0.8, 0.9)
                else:
                    seasonal_factor = random.uniform(0.9, 1.1)

                consumption = base_consumption * seasonal_factor
                bill_date = current_date.date()
                due_date = bill_date + timedelta(days=10)
                payment_date = bill_date + timedelta(days=random.randint(5, 15))

                # Preço base por m³ (aumentando ao longo dos anos)
                years_passed = current_date.year - start_date.year
                base_price = 5.0 * (1.1**years_passed)  # 10% de aumento ao ano
                amount = round(consumption * base_price, 2)
                meter_reading = random.uniform(
                    1000, 5000
                )  # Leitura fictícia do hidrômetro

                bill = WaterBill.objects.create(
                    property=prop,
                    bill_date=bill_date,
                    due_date=due_date,
                    consumption=round(consumption, 2),
                    amount=amount,
                    meter_reading=meter_reading,
                    status="paid" if payment_date <= due_date else "overdue",
                )
                self.stdout.write(f"Criada conta: {bill}")

            # Avança para o próximo mês
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        self.stdout.write(self.style.SUCCESS("Dados sintéticos gerados com sucesso!"))
