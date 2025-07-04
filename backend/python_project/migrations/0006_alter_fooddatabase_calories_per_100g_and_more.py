# Generated by Django 5.1.6 on 2025-03-23 19:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('python_project', '0005_alter_fooddatabase_sodium_per_100g'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fooddatabase',
            name='calories_per_100g',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='calories_per_unit',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='carbohydrate_per_100g',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='carbohydrate_per_unit',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='fat_per_100g',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='fat_per_unit',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='protein_per_100g',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='protein_per_unit',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='sugar_per_100g',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='fooddatabase',
            name='sugar_per_unit',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
    ]
