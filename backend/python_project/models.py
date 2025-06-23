from django.contrib.auth.models import User
from django.db import models
from django.forms.models import model_to_dict
import json
from django.db.models import Q, Count
from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver
from datetime import date
from django.utils.timezone import now
from decimal import Decimal 
from rest_framework import serializers
from collections import defaultdict

class Message(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    message_text = models.TextField()  
    is_user = models.BooleanField()  
    created_at = models.DateTimeField(auto_now_add=True)  
    llm_response_time_s = models.FloatField(null=True, blank=True)  
    function_called = models.JSONField(null=True, blank=True)  
    function_result = models.JSONField(null=True, blank=True)  
    token_used = models.IntegerField(null=True, blank=True)  
    metadata = models.JSONField(null=True, blank=True)  

    def __str__(self):
        return f"Message {self.id} by {'User' if self.is_user else 'LLM'}"

    def get_user_message(self):
        return {
            "id": self.id,
            "message_text": self.message_text,
            "is_user": self.is_user,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def get_messages_by_user(cls, user_id):
        messages = cls.objects.filter(user__id=user_id).order_by("created_at")  # Sort by oldest
        return json.dumps([msg.get_user_message() for msg in messages], indent=4)  # Convert to JSON
class Exercise(models.Model):
    exercise_name = models.TextField(null=True, blank=True)
    short_video = models.TextField(null=True, blank=True)
    long_video = models.TextField(null=True, blank=True)
    difficulty = models.TextField(null=True, blank=True)
    target_muscle = models.TextField(null=True, blank=True)
    primary_equipment = models.TextField(null=True, blank=True)
    secondary_equipment = models.TextField(null=True, blank=True)
    body_region = models.TextField(null=True, blank=True)
    mechanics = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.target_muscle} ({self.difficulty})"
    
    @classmethod
    def get_exercise_by_setting(cls, difficulty, exclude_equipment):
        # Ensure exclude_equipment is a list (avoid NoneType issues)
        if not exclude_equipment:
            exclude_equipment = []

        # Fetch exercises that match difficulty but do NOT use excluded equipment
        exercises = cls.objects.filter(difficulty=difficulty)

        if exclude_equipment:
            exercises = exercises.exclude(
                Q(primary_equipment__in=exclude_equipment) | Q(secondary_equipment__in=exclude_equipment)
            )

        # Group exercises by body region
        grouped_exercises = {}
        for ex in exercises:
            if ex.target_muscle not in grouped_exercises:
                grouped_exercises[ex.target_muscle ] = []
            grouped_exercises[ex.target_muscle].append({
                "id": ex.id,
                "exercise": str(ex),
                "difficulty": ex.difficulty,
                "primary_equipment": ex.primary_equipment,
                "secondary_equipment": ex.secondary_equipment
            })

        return grouped_exercises
        

class WorkoutWeek(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    week = models.IntegerField()
    start_date = models.DateField()
    end_date = models.DateField()

    def __str__(self):
        return f"Week {self.week} ({self.start_date} - {self.end_date})"

class Workout(models.Model):
    class StatusChoices(models.TextChoices):
        COMPLETED = "completed", "Completed"
        SKIPPED = "skipped", "Skipped"
        MISSED = "missed", "Missed"
        PENDING = "pending", "Pending"

    workout_week = models.ForeignKey(WorkoutWeek, on_delete=models.CASCADE,related_name="workouts")
    exercise = models.ForeignKey(Exercise, on_delete=models.CASCADE)
    duration_seconds = models.IntegerField(default=45)
    rest_seconds = models.IntegerField(default=20)
    in_progress = models.BooleanField(default=False)
    progress_data = models.JSONField(null=True, blank=True)
    last_paused_at = models.DateTimeField(null=True, blank=True)
    date = models.DateField()
    day = models.CharField(max_length=20)  
    status = models.CharField(
        max_length=10,
        choices=StatusChoices.choices,
        default=StatusChoices.PENDING,
    )

    def __str__(self):
        return f"{self.exercise} on {self.date} - {self.status}"
    
    def get_user_workout(self):
        workout_data = model_to_dict(self, fields=["id", "status"])
        workout_data["exercise_name"] = self.exercise.exercise_name
        workout_data["short_video"] = self.exercise.short_video
        workout_data["long_video"] = self.exercise.long_video
        return workout_data
    
    @classmethod
    def get_today_summary(cls, user_profile):
        today = now().date()

        workouts = cls.objects.filter(
            workout_week__user=user_profile.user,  
            date=today
        )

        summary = workouts.aggregate(
            total=Count('id'),
            completed=Count('id', filter=Q(status=cls.StatusChoices.COMPLETED)),
            skipped=Count('id', filter=Q(status=cls.StatusChoices.SKIPPED))
        )

        workout_list = [w.get_user_workout() for w in workouts]

        return {
            "date": today,
            "total": summary["total"],
            "completed": summary["completed"],
            "skipped": summary["skipped"],
            "workouts": workout_list,
        }
    
class WorkoutProgress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    workout = models.ForeignKey(Workout, on_delete=models.CASCADE)
    set_number = models.IntegerField(default=1)
    rep_number = models.IntegerField(default=1)
    seconds_left_in_rest = models.IntegerField(default=0)
    status = models.CharField(max_length=20, default="paused")  # paused/active/completed
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'workout')

class WorkoutSetting(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    generate_schedule = models.BooleanField(default=False)
    current_week = models.IntegerField(null=True, blank=True)
    current_week_start = models.DateField(null=True, blank=True)
    current_week_end = models.DateField(null=True, blank=True)
    workout_days = models.JSONField(null=True, blank=True)  # e.g., {"Monday": True, "Tuesday": False}
    difficulty = models.TextField(null=True, blank=True)
    exclude_equipment = models.JSONField(null=True, blank=True)  # e.g., ["Dumbbells", "Barbell"]

    def __str__(self):
        return f"Workout Settings for {self.user}"

class FoodDatabase(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    product_name = models.CharField(max_length=255)
    categories = models.CharField(max_length=30)
    calories_per_100g = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    protein_per_100g = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    fat_per_100g = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    carbohydrate_per_100g = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    sugar_per_100g = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    sodium_per_100g = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    calories_per_unit = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    protein_per_unit = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    fat_per_unit = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    carbohydrate_per_unit = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    sugar_per_unit = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    sodium_per_unit = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    unit = models.CharField(max_length=60)
    image_url = models.URLField(max_length=140, null=True,blank=True)

    def __str__(self):
        return self.product_name

class UserProfile(models.Model):
    class Gender(models.TextChoices):
        MALE = 'M', 'Male'
        FEMALE = 'F', 'Female'

    class Goal(models.TextChoices):
        MAINTAIN = 'maintain', 'Maintain Weight'
        WEIGHT_LOSS = 'Weight Loss', 'Weight Loss'
        WEIGHT_GAIN = 'Weight Gain', 'Weight Gain'

    class ActivityLevel(models.TextChoices):
        SEDENTARY = 'Sedentary', 'Sedentary'
        LIGHTLY_ACTIVE = "Lightly Active", "Lightly Active"
        MODERATELY_ACTIVE = "Moderately Active", "Moderately Active"
        VERY_ACTIVE = "Very Active", "Very Active"

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    birthday = models.DateField(null=False, blank=False)
    gender = models.CharField(max_length=10, choices=Gender.choices)
    average_meal_count = models.IntegerField(null=False, blank=False)
    height_cm = models.IntegerField(null=False, blank=False)
    current_weight_kg = models.IntegerField(null=False, blank=False)
    weight_goal = models.CharField(max_length=15, choices=Goal.choices)
    total_target_kg = models.IntegerField(null=False, blank=False)
    target_weight_kg = models.IntegerField(null=False, blank=False)
    activity_level = models.CharField(max_length=20, choices=ActivityLevel.choices)

    def __str__(self):
        return f"{self.user.username} - {self.weight_goal}"

    @classmethod
    def create_user_profile(cls, user, birthday, gender, meal_count, height, current_weight, weight_goal, total_target, activity_level):
        """
        Class method to create a UserProfile instance with calculated target_weight_kg.
        """
        if weight_goal == cls.Goal.WEIGHT_LOSS:
            target_weight = current_weight - total_target
        elif weight_goal == cls.Goal.WEIGHT_GAIN:
            target_weight = current_weight + total_target
        else:  # Maintain weight
            target_weight = current_weight

        return cls.objects.create(
            user=user,
            birthday=birthday,
            gender=gender,
            average_meal_count=meal_count,
            height_cm=height,
            current_weight_kg=current_weight,
            weight_goal=weight_goal,
            total_target_kg=total_target,
            target_weight_kg=target_weight,
            activity_level=activity_level
        )


class UserCalories(models.Model):
    user_profile = models.OneToOneField(UserProfile, on_delete=models.CASCADE, related_name="calories")
    tdee = models.IntegerField()
    calorie_adjustment = models.IntegerField()
    target_calories = models.IntegerField()
    updated_at = models.DateTimeField(auto_now=True)

    @classmethod
    def create_user_calories_if_not_exists(cls, user_profile):
        """Automatically create UserCalories record if it doesn't exist, using TDEE calculation."""
        if not cls.objects.filter(user_profile=user_profile).exists():
            tdee = cls.calculate_tdee(user_profile)
            calorie_adjustment = cls.get_calorie_adjustment(user_profile.weight_goal)
            target_calories = tdee + calorie_adjustment
            
            return cls.objects.create(
                user_profile=user_profile,
                tdee=tdee,
                calorie_adjustment=calorie_adjustment,
                target_calories=target_calories
                )

    @staticmethod
    def calculate_tdee(user_profile):
        """Calculate TDEE based on BMR and activity level."""
        try:

            weight = user_profile.current_weight_kg
            height = user_profile.height_cm
            age = date.today().year - user_profile.birthday.year
            gender = user_profile.gender

            # Calculate BMR (Mifflin-St Jeor Equation)
            if gender == UserProfile.Gender.MALE:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
            else:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

            activity_multipliers = {
                UserProfile.ActivityLevel.SEDENTARY: 1.2,
                UserProfile.ActivityLevel.LIGHTLY_ACTIVE: 1.375,
                UserProfile.ActivityLevel.MODERATELY_ACTIVE: 1.55,
                UserProfile.ActivityLevel.VERY_ACTIVE: 1.725
            }

            multiplier = activity_multipliers.get(user_profile.activity_level)
            result = int(bmr * multiplier)

            return result

        except Exception as e:
            print("Error in calculate_tdee:", e)
            raise

    @staticmethod
    def get_calorie_adjustment(weight_goal):
        """Determine calorie adjustment based on goal."""
        if weight_goal == UserProfile.Goal.WEIGHT_LOSS:
            return -500
        elif weight_goal == UserProfile.Goal.WEIGHT_GAIN:
            return 500
        return 0  # Maintain weight

    def save(self, *args, **kwargs):
        self.target_calories = self.tdee + self.calorie_adjustment
        super().save(*args, **kwargs)
class DailyNutrition(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="daily_nutrition")
    date = models.DateField(default=now)
    total_breakfast_cal = models.IntegerField(default=0)
    total_lunch_cal = models.IntegerField(default=0)
    total_dinner_cal = models.IntegerField(default=0)
    total_snack_cal = models.IntegerField(default=0)
    total_carbs = models.DecimalField(max_digits=6, decimal_places=2, default=0.0)  # in grams
    total_protein = models.DecimalField(max_digits=6, decimal_places=2, default=0.0)  # in grams
    total_fats = models.DecimalField(max_digits=6, decimal_places=2, default=0.0)  # in grams
    total_salts = models.DecimalField(max_digits=6, decimal_places=2, default=0.0)  # in grams
    total_sugar = models.DecimalField(max_digits=6, decimal_places=2, default=0.0)  # in grams
    
    total_calories = models.IntegerField(default=0)
    target_calories = models.IntegerField(default=0)  # Set from UserCalories
    
    class Meta:
        unique_together = ('user_profile', 'date')  # Ensure one record per user per day
    
    def save(self, *args, **kwargs):
        self.total_calories = (
            self.total_breakfast_cal + self.total_lunch_cal +
            self.total_dinner_cal + self.total_snack_cal
        )
        
        # Set target_calories from UserCalories
        user_calories = UserCalories.objects.filter(user_profile=self.user_profile).first()
        if user_calories:
            self.target_calories = user_calories.target_calories
        
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.user_profile.user.username} - {self.date}"

    def get_food_entries_grouped(self):
        """Fetch food entries and group them by meal label."""
        food_entries = self.food_entries.all().values(
            "id","label", "food_id__id", "food_id__product_name",
            "food_id__unit","measurement_unit","serving","weight", "calories", "carbs", "protein",
            "fats", "salts", "sugar"
        )
        
        grouped_entries = defaultdict(list)
        for entry in food_entries:
            grouped_entries[entry["label"]].append({
                "log_id": entry["id"],
                "meal_time":entry["label"],
                "food_id": entry["food_id__id"],
                "food_name": entry["food_id__product_name"],
                "food_unit" : entry["food_id__unit"],
                "measurement_unit": entry["measurement_unit"],
                "serving" : entry["serving"],
                "weight" : entry["weight"],
                "calories": entry["calories"],
                "carbs": float(entry["carbs"]),
                "protein": float(entry["protein"]),
                "fats": float(entry["fats"]),
                "salts": float(entry["salts"]),
                "sugar": float(entry["sugar"]),
            })
        
        return grouped_entries

    @classmethod
    def get_user_daily_nutrition(cls, user_profile):
        today = now().date()
        daily_nutrition, created = cls.objects.get_or_create(user_profile=user_profile, date=today)
        return daily_nutrition
    
    @classmethod
    def get_user_nutrition_records(cls, user_profile, start_date=None, end_date=None):
        """Ensure today's DailyNutrition entry exists, then retrieve all records within a date range."""
        today = now().date()
        cls.objects.get_or_create(user_profile=user_profile, date=today)  # Ensure today's entry exists

        # Fetch nutrition records
        nutrition_records = cls.objects.filter(user_profile=user_profile).order_by('date').prefetch_related('food_entries')

        if start_date and end_date:
            nutrition_records = nutrition_records.filter(date__range=[start_date, end_date])

        # Convert records into structured JSON format
        structured_records = []
        for record in nutrition_records:
            structured_records.append({
                "id": record.id,
                "date": record.date,
                "total_breakfast_cal": record.total_breakfast_cal,
                "total_lunch_cal": record.total_lunch_cal,
                "total_dinner_cal": record.total_dinner_cal,
                "total_snack_cal": record.total_snack_cal,
                "total_carbs": float(record.total_carbs),
                "total_protein": float(record.total_protein),
                "total_fats": float(record.total_fats),
                "total_salts": float(record.total_salts),
                "total_sugar": float(record.total_sugar),
                "total_calories": record.total_calories,
                "target_calories": record.target_calories,
                "food_entries": dict(record.get_food_entries_grouped()),
            })

        return structured_records

class FoodEntry(models.Model):
    
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="food_entries")
    daily_nutrition = models.ForeignKey(DailyNutrition, on_delete=models.CASCADE, related_name="food_entries")
    date = models.DateField(default=now)
    food_id = models.ForeignKey(FoodDatabase, on_delete=models.CASCADE) 
    label = models.CharField(max_length=10, choices=[
        ('breakfast', 'Breakfast'),
        ('lunch', 'Lunch'),
        ('dinner', 'Dinner'),
        ('snack', 'Snack')
    ])
    measurement_unit = models.CharField(max_length=10, choices=[
        ('weight', 'weight'),
        ('serving', 'serving'),
    ])
    serving = models.IntegerField(null=True)
    weight = models.DecimalField(max_digits=6, decimal_places=2, null=True) 
    calories = models.IntegerField()
    carbs = models.DecimalField(max_digits=6, decimal_places=2)  # in grams
    protein = models.DecimalField(max_digits=6, decimal_places=2)  # in grams
    fats = models.DecimalField(max_digits=6, decimal_places=2)  # in grams
    salts = models.DecimalField(max_digits=6, decimal_places=2)  # in grams
    sugar = models.DecimalField(max_digits=6, decimal_places=2)  # in grams
    
    def save(self, *args, **kwargs):
        is_update = self.pk is not None

        if not is_update:
            # New entry: fetch or create DailyNutrition
            daily_nutrition = DailyNutrition.get_user_daily_nutrition(self.user_profile)
            self.daily_nutrition = daily_nutrition
        else:
            # Existing entry: use current daily_nutrition
            daily_nutrition = self.daily_nutrition
        
        super().save(*args, **kwargs)
        
        # Update DailyNutrition totals
        if self.label == 'breakfast':
            daily_nutrition.total_breakfast_cal += self.calories
        elif self.label == 'lunch':
            daily_nutrition.total_lunch_cal += self.calories
        elif self.label == 'dinner':
            daily_nutrition.total_dinner_cal += self.calories
        elif self.label == 'snack':
            daily_nutrition.total_snack_cal += self.calories
        
        daily_nutrition.total_carbs += self.carbs
        daily_nutrition.total_protein += self.protein
        daily_nutrition.total_fats += self.fats
        daily_nutrition.total_salts += self.salts
        daily_nutrition.total_sugar += self.sugar
        
        daily_nutrition.save()
    
    def __str__(self):
        return f"{self.user_profile.user.username} - {self.food_id} ({self.label}) on {self.date}"

    @classmethod
    def Insert_new_entry(cls, user_id, daily_nutrition_id, meal_label, measurement_unit, food_id, serving, weight):
        try:
            user = User.objects.get(id=user_id)
            user_profile = UserProfile.objects.get(user=user)
            daily_nutrition = DailyNutrition.objects.get(id=daily_nutrition_id)  # Fix lookup
            food = FoodDatabase.objects.get(id=food_id)  # Fix lookup

            def safe_decimal(value, default=Decimal(0)):
                """Convert value to Decimal, return default if None."""
                return Decimal(value) if value is not None else default


            if measurement_unit == 'serving':
                serving = serving
                weight = None
                calories = safe_decimal(food.calories_per_unit) * safe_decimal(serving)
                protein = safe_decimal(food.protein_per_unit) * safe_decimal(serving)
                fats = safe_decimal(food.fat_per_unit) * safe_decimal(serving)
                carbs = safe_decimal(food.carbohydrate_per_unit) * safe_decimal(serving)
                sugar = safe_decimal(food.sugar_per_unit) * safe_decimal(serving)
                sodium = safe_decimal(food.sodium_per_unit) * safe_decimal(serving)
            else:
                serving = None
                weight = safe_decimal(weight)
                calories = safe_decimal(food.calories_per_100g) * (safe_decimal(weight) / Decimal(100))
                protein = safe_decimal(food.protein_per_100g) * (safe_decimal(weight) / Decimal(100))
                fats = safe_decimal(food.fat_per_100g) * (safe_decimal(weight) / Decimal(100))
                carbs = safe_decimal(food.carbohydrate_per_100g) * (safe_decimal(weight) / Decimal(100))
                sugar = safe_decimal(food.sugar_per_100g) * (safe_decimal(weight) / Decimal(100))
                sodium = safe_decimal(food.sodium_per_100g) * (safe_decimal(weight) / Decimal(100))


            new_entry = cls.objects.create(
                user_profile=user_profile, 
                date=daily_nutrition.date,
                daily_nutrition=daily_nutrition,
                food_id=food,
                label=meal_label,
                measurement_unit=measurement_unit,
                serving=serving,
                weight=weight,
                calories=calories,
                carbs=carbs,
                protein=protein,
                sugar=sugar,
                salts=sodium,
                fats=fats,
            )

            return new_entry  # Ensure an object is returned

        except Exception as e:
            raise ValueError(f"Error inserting new entry: {str(e)}")  # Handle errors properly

    @classmethod
    def edit_entry(cls, log_id, meal_label, measurement_unit, food_id, serving, weight):
        try:
            entry = cls.objects.get(id=log_id)
            old_label = entry.label
            old_calories = entry.calories
            old_carbs = entry.carbs
            old_protein = entry.protein
            old_fats = entry.fats
            old_salts = entry.salts
            old_sugar = entry.sugar

            daily_nutrition = entry.daily_nutrition
            
            cls.subtract_nutrition_from_daily( 
                daily_nutrition, old_label, old_calories,
                old_carbs,old_protein, old_fats, 
                old_salts, old_sugar)

            # Update the food reference
            food = FoodDatabase.objects.get(id=food_id)

            def safe_decimal(value, default=Decimal(0)):
                return Decimal(value) if value is not None else default

            if measurement_unit == 'serving':
                calories = safe_decimal(food.calories_per_unit) * safe_decimal(serving)
                protein = safe_decimal(food.protein_per_unit) * safe_decimal(serving)
                fats = safe_decimal(food.fat_per_unit) * safe_decimal(serving)
                carbs = safe_decimal(food.carbohydrate_per_unit) * safe_decimal(serving)
                sugar = safe_decimal(food.sugar_per_unit) * safe_decimal(serving)
                sodium = safe_decimal(food.sodium_per_unit) * safe_decimal(serving)
                weight = None  # ensure weight is cleared
            else:
                weight = safe_decimal(weight)
                factor = weight / Decimal(100)
                calories = safe_decimal(food.calories_per_100g) * factor
                protein = safe_decimal(food.protein_per_100g) * factor
                fats = safe_decimal(food.fat_per_100g) * factor
                carbs = safe_decimal(food.carbohydrate_per_100g) * factor
                sugar = safe_decimal(food.sugar_per_100g) * factor
                sodium = safe_decimal(food.sodium_per_100g) * factor
                serving = None  # ensure serving is cleared

            # Update the entry fields
            entry.label = meal_label
            entry.measurement_unit = measurement_unit
            entry.food_id = food
            entry.serving = serving
            entry.weight = weight
            entry.calories = calories
            entry.carbs = carbs
            entry.protein = protein
            entry.fats = fats
            entry.sugar = sugar
            entry.salts = sodium

            # Save to re-add new values via custom save()
            entry.save()

            return entry

        except cls.DoesNotExist:
            raise ValueError(f"FoodEntry with ID {log_id} does not exist.")
        except FoodDatabase.DoesNotExist:
            raise ValueError(f"Food with ID {food_id} does not exist.")
        except Exception as e:
            raise ValueError(f"Error editing entry: {str(e)}")
        
    @classmethod
    def delete_entry(cls, log_id):
        try:
            entry = cls.objects.get(id=log_id)

            # Store the existing nutrition values
            old_label = entry.label
            old_calories = entry.calories
            old_carbs = entry.carbs
            old_protein = entry.protein
            old_fats = entry.fats
            old_salts = entry.salts
            old_sugar = entry.sugar

            daily_nutrition = entry.daily_nutrition

            # Subtract from daily totals
            cls.subtract_nutrition_from_daily(
                daily_nutrition,
                old_label,
                old_calories,
                old_carbs,
                old_protein,
                old_fats,
                old_salts,
                old_sugar
            )

            entry.delete()

        except cls.DoesNotExist:
            raise ValueError(f"FoodEntry with ID {log_id} does not exist.")
        except Exception as e:
            raise ValueError(f"Error deleting entry: {str(e)}")
        

    @staticmethod
    def subtract_nutrition_from_daily(daily_nutrition, label, calories, carbs, protein, fats, salts, sugar):
        if label == 'breakfast':
            daily_nutrition.total_breakfast_cal -= calories
        elif label == 'lunch':
            daily_nutrition.total_lunch_cal -= calories
        elif label == 'dinner':
            daily_nutrition.total_dinner_cal -= calories
        elif label == 'snack':
            daily_nutrition.total_snack_cal -= calories

        daily_nutrition.total_carbs -= carbs
        daily_nutrition.total_protein -= protein
        daily_nutrition.total_fats -= fats
        daily_nutrition.total_salts -= salts
        daily_nutrition.total_sugar -= sugar
        daily_nutrition.save()



@receiver(post_save, sender=UserProfile)
def auto_create_user_calories(sender, instance, created, **kwargs):
    if created:
        UserCalories.create_user_calories_if_not_exists(instance)

class FoodEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = FoodEntry
        fields = '__all__'

class DailyNutritionSerializer(serializers.ModelSerializer):
    food_entries = FoodEntrySerializer(many=True, read_only=True)  # Nested food entries

    class Meta:
        model = DailyNutrition
        fields = '__all__'

class UserRecipe(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    ingredients = models.JSONField()  # List of strings
    food_ref = models.ForeignKey(FoodDatabase, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} by {self.user.username}"

# In your models.py, add this if you don't have a WeightLog model:
class WeightLog(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='weight_logs')
    date = models.DateField()
    weight = models.FloatField()
    class Meta:
        ordering = ["-date"]