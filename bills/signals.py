from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import WaterBill, Consumption


@receiver(post_save, sender=WaterBill)
def create_consumption_record(sender, instance, created, **kwargs):
    """Create or update consumption record when a water bill is saved."""
    if created or instance.consumption:
        # Calculate the billing period (assuming 30 days if not specified)
        billing_period = 30  # days

        # Calculate average daily consumption
        avg_daily_consumption = instance.consumption / billing_period

        # Create or update consumption record
        Consumption.objects.update_or_create(
            property=instance.property,
            date=instance.bill_date,
            defaults={
                "consumption": instance.consumption,
                "average_daily_consumption": avg_daily_consumption,
            },
        )
