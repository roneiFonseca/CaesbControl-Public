# Generated by Django 5.0.1 on 2025-01-31 19:42

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("bills", "0001_initial"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="property",
            options={
                "ordering": ["name"],
                "permissions": [
                    ("view_all_properties", "Can view all properties"),
                    ("view_all_bills", "Can view all water bills"),
                ],
                "verbose_name": "Imóvel",
                "verbose_name_plural": "Imóveis",
            },
        ),
    ]
