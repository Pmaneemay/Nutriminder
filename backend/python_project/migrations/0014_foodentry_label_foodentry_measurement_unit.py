# Generated by Django 5.1.6 on 2025-03-26 09:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('python_project', '0013_remove_foodentry_label_foodentry_serving_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='foodentry',
            name='label',
            field=models.CharField(choices=[('breakfast', 'Breakfast'), ('lunch', 'Lunch'), ('dinner', 'Dinner'), ('snack', 'Snack')], default='snack', max_length=10),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='foodentry',
            name='measurement_unit',
            field=models.CharField(choices=[('weight', 'Breakfast'), ('serving', 'Lunch')], default='serving', max_length=10),
            preserve_default=False,
        ),
    ]
