from django.apps import AppConfig


class BillsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "bills"
    verbose_name = "Contas de Água"

    def ready(self):
        import bills.signals  # noqa
