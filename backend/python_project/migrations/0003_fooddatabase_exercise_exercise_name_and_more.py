# Generated by Django 5.1.6 on 2025-03-23 18:57

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('python_project', '0002_exercise_workoutsetting_workoutweek_workout'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='FoodDatabase',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_name', models.CharField(max_length=255)),
                ('categories', models.CharField(max_length=30)),
                ('calories_per_100g', models.DecimalField(blank=True, decimal_places=2, max_digits=6, null=True)),
                ('protein_per_100g', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('fat_per_100g', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('carbohydrate_per_100g', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('sugar_per_100g', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('sodium_per_100g', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('calories_per_unit', models.DecimalField(blank=True, decimal_places=2, max_digits=6, null=True)),
                ('protein_per_unit', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('fat_per_unit', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('carbohydrate_per_unit', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('sugar_per_unit', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('sodium_per_unit', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('unit', models.CharField(max_length=60)),
                ('image_url', models.URLField(max_length=140)),
            ],
        ),
        migrations.AddField(
            model_name='exercise',
            name='exercise_name',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='workout',
            name='workout_week',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='workouts', to='python_project.workoutweek'),
        ),
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('birthday', models.DateField()),
                ('gender', models.CharField(choices=[('m', 'M'), ('f', 'F')], max_length=10)),
                ('average_meal_count', models.IntegerField()),
                ('height_cm', models.IntegerField()),
                ('current_weight_kg', models.IntegerField()),
                ('weight_goal', models.CharField(choices=[('maintain', 'Maintain Weight'), ('weight_loss', 'Weight Loss'), ('weight_gain', 'Weight Gain')], max_length=15)),
                ('total_target_kg', models.IntegerField()),
                ('target_weight_kg', models.IntegerField()),
                ('activity_level', models.CharField(choices=[('sedentary', 'Sedentary'), ('lightly_active', 'Lightly Active'), ('moderately_active', 'Moderately Active'), ('very_active', 'Very Active')], max_length=20)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
