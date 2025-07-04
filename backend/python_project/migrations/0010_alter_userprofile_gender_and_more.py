# Generated by Django 5.1.6 on 2025-03-23 22:54

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('python_project', '0009_usercalories'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userprofile',
            name='gender',
            field=models.CharField(choices=[('M', 'Male'), ('F', 'Female')], max_length=10),
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='weight_goal',
            field=models.CharField(choices=[('maintain', 'Maintain Weight'), ('Weight Loss', 'Weight Loss'), ('Weight Gain', 'Weight Gain')], max_length=15),
        ),
        migrations.CreateModel(
            name='DailyNutrition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(default=django.utils.timezone.now)),
                ('total_breakfast_cal', models.IntegerField(default=0)),
                ('total_lunch_cal', models.IntegerField(default=0)),
                ('total_dinner_cal', models.IntegerField(default=0)),
                ('total_snack_cal', models.IntegerField(default=0)),
                ('total_carbs', models.DecimalField(decimal_places=2, default=0.0, max_digits=6)),
                ('total_protein', models.DecimalField(decimal_places=2, default=0.0, max_digits=6)),
                ('total_fats', models.DecimalField(decimal_places=2, default=0.0, max_digits=6)),
                ('total_salts', models.DecimalField(decimal_places=2, default=0.0, max_digits=6)),
                ('total_sugar', models.DecimalField(decimal_places=2, default=0.0, max_digits=6)),
                ('total_calories', models.IntegerField(default=0)),
                ('target_calories', models.IntegerField(default=0)),
                ('user_profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='daily_nutrition', to='python_project.userprofile')),
            ],
            options={
                'unique_together': {('user_profile', 'date')},
            },
        ),
        migrations.CreateModel(
            name='FoodEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(default=django.utils.timezone.now)),
                ('label', models.CharField(choices=[('breakfast', 'Breakfast'), ('lunch', 'Lunch'), ('dinner', 'Dinner'), ('snack', 'Snack')], max_length=10)),
                ('calories', models.IntegerField()),
                ('carbs', models.DecimalField(decimal_places=2, max_digits=6)),
                ('protein', models.DecimalField(decimal_places=2, max_digits=6)),
                ('fats', models.DecimalField(decimal_places=2, max_digits=6)),
                ('salts', models.DecimalField(decimal_places=2, max_digits=6)),
                ('sugar', models.DecimalField(decimal_places=2, max_digits=6)),
                ('daily_nutrition', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='food_entries', to='python_project.dailynutrition')),
                ('food_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='python_project.fooddatabase')),
                ('user_profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='food_entries', to='python_project.userprofile')),
            ],
        ),
    ]
