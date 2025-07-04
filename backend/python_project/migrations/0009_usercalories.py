# Generated by Django 5.1.6 on 2025-03-23 21:20

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('python_project', '0008_alter_fooddatabase_calories_per_100g_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserCalories',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tdee', models.IntegerField()),
                ('calorie_adjustment', models.IntegerField()),
                ('target_calories', models.IntegerField()),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user_profile', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='calories', to='python_project.userprofile')),
            ],
        ),
    ]
