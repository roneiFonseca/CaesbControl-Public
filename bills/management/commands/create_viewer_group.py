from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from bills.models import Property, WaterBill


class Command(BaseCommand):
    help = "Create viewer group with permissions to view all properties and bills"

    def handle(self, *args, **options):
        # Create the viewer group
        viewer_group, created = Group.objects.get_or_create(name="Data Viewers")

        # Get the content type for our models
        property_ct = ContentType.objects.get_for_model(Property)
        waterbill_ct = ContentType.objects.get_for_model(WaterBill)

        # Get or create the custom permissions
        view_all_properties = Permission.objects.get(
            codename="view_all_properties",
            content_type=property_ct,
        )
        view_all_bills = Permission.objects.get(
            codename="view_all_bills",
            content_type=property_ct,
        )

        # Add the permissions to the group
        viewer_group.permissions.add(view_all_properties)
        viewer_group.permissions.add(view_all_bills)

        # Also add the basic view permissions
        view_property = Permission.objects.get(
            codename="view_property",
            content_type=property_ct,
        )
        view_waterbill = Permission.objects.get(
            codename="view_waterbill",
            content_type=waterbill_ct,
        )
        viewer_group.permissions.add(view_property)
        viewer_group.permissions.add(view_waterbill)

        self.stdout.write(self.style.SUCCESS("Successfully created Data Viewers group"))
