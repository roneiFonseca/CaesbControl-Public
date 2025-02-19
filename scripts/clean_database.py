import os
import sys
import django

from django.contrib.auth.models import User
from django.db import transaction
from bills.models import Property, WaterBill, Consumption

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "caesb_control.settings")
django.setup()


def clean_database():
    """
    Remove all data from the database while preserving the database structure.
    """
    try:
        with transaction.atomic():
            # Delete all records from our models
            print("Cleaning database...")

            # Delete in order to respect foreign key constraints
            print("Removing Consumption records...")
            Consumption.objects.all().delete()

            print("Removing WaterBill records...")
            WaterBill.objects.all().delete()

            print("Removing Property records...")
            Property.objects.all().delete()

            # Optionally, remove all users except superusers
            print("Removing non-superuser accounts...")
            User.objects.filter(is_superuser=False).delete()

            print("Database cleaned successfully!")

    except Exception as e:
        print(f"An error occurred while cleaning the database: {e}")


if __name__ == "__main__":
    clean_database()
