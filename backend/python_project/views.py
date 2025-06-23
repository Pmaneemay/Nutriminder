from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt
from django.middleware.csrf import get_token
from django.http import JsonResponse
from datetime import datetime, date, timedelta
from django.contrib.sessions.models import Session
from django.contrib.auth.models import User
from django.utils import timezone
from .models import Message, WorkoutSetting, Exercise, Workout,WorkoutProgress, WorkoutWeek, UserProfile,UserCalories, DailyNutrition, FoodDatabase, FoodEntry, UserRecipe, WeightLog
from groq import Groq
from collections import defaultdict
from django.db import transaction
import json, os, pulp, logging, itertools, random, traceback
from datetime import date
from django.utils.timezone import now
from django.core.mail import send_mail
import random
from decimal import Decimal
from django.db.models import Q
from django.core.cache import cache


logger = logging.getLogger(__name__)


# Create your views here.
def myApp(request):
    return render(request,'main.html', {'name': 'mayratdaa'})

@api_view(['GET','POST'])
def getData(request):
    product_data = data.products
    return Response(product_data)

@csrf_exempt
def user_login(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        print("Received login data:", data)  
        username = data.get('username')
        password = data.get('password')
        
        user = authenticate(username=username, password=password)

        if user:
            login(request, user)
            session_id = request.session.session_key  
            csrf_token = get_token(request)
            return JsonResponse({"message": "Login successful", "sessionID": session_id, "csrfToken": csrf_token}, status=200)

        else:
            print("Invalid credentials");
            return JsonResponse({"error": "Invalid credentials"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)

@csrf_exempt
@api_view(['POST'])
def verify_username_email_and_send_code(request):
    try:
        data = json.loads(request.body)
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()

        response_data = {
            "username_exists": False,
            "email_exists": False,
        }

        # Check for duplicates
        if User.objects.filter(username=username).exists():
            response_data["username_exists"] = True

        if User.objects.filter(email=email).exists():
            response_data["email_exists"] = True

        if response_data["username_exists"] or response_data["email_exists"]:
            return JsonResponse(response_data, status=200)

        # If valid, generate verification code
        verification_code = random.randint(100000, 999999)

        # Send verification email
        send_mail(
            subject="NutriMinder Email Verification",
            message=f"Your NutriMinder verification code is: {verification_code}",
            from_email="noreply@nutriminder.com",
            recipient_list=[email],
            fail_silently=False,
        )

        # Optionally return code for debugging (REMOVE in production)
        response_data["verification_code"] = verification_code

        return JsonResponse(response_data, status=200)

    except Exception as e:
        print("Error verifying signup info:", e)
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def user_signup(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received signup data:", data)

            # Convert date format
            birthday = datetime.strptime(data['birthday'], "%d/%m/%Y").date()

            # Convert numeric fields
            height = int(data['height'])
            weight = int(data['weight'])
            meal_count = int(data['mealcount'])
            weight_goal = int(data['weightgoal'])

            # Ensure weight goal is valid
            valid_goals = [UserProfile.Goal.MAINTAIN, UserProfile.Goal.WEIGHT_LOSS, UserProfile.Goal.WEIGHT_GAIN]
            if data['goal'] not in valid_goals:
                return JsonResponse({"error": "Invalid weight goal"}, status=400)

            # Create user
            user_instance = User.objects.create_user(
                username=data['username'],
                email=data['email'],
                password=data['password']
            )

            # Create user profile
            user_profile = UserProfile.create_user_profile(
                user=user_instance,
                birthday=birthday,
                gender=data['gender'],
                meal_count=meal_count,
                height=height,
                current_weight=weight,
                weight_goal=data['goal'],
                total_target=weight_goal,
                activity_level=data['activity_level']
            )

            # Ensure UserCalories exists
            user_calories, created = UserCalories.objects.get_or_create(user_profile=user_profile)

            set_workoutsetting(user_instance=user_instance, data=data)

            WorkoutSetting_instance = WorkoutSetting.objects.filter(user=user_instance).first();

            schedule = json.loads(generate_schedule(workoutSetting_instance=WorkoutSetting_instance));

            map_dates_to_schedule(user=user_instance,workoutSetting_instance=WorkoutSetting_instance,workout_schedule=schedule);

            return JsonResponse({"message": "Signup successful"}, status=200 )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except KeyError as e:
            return JsonResponse({"error": f"Missing key: {str(e)}"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def meal_history(request):
    if request.method == 'GET':
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')

        try:
            user_profile = UserProfile.objects.get(user__id=user_id)
            nutrition_records = DailyNutrition.get_user_nutrition_records(user_profile)

            print(nutrition_records)

            return JsonResponse({
                    "nutrition_records": nutrition_records
                    }, status=200)

        except UserProfile.DoesNotExist:
            return JsonResponse({"error": "User profile not found"}, status=404)

    return JsonResponse({"error": "Invalid request method"}, status=400)

def get_duration_and_rest(difficulty):
    if difficulty == 'beginner':
        return 30, 30
    elif difficulty == 'intermediate':
        return 45, 20
    elif difficulty == 'advanced':
        return 60, 15
    return 45, 20

def map_dates_to_schedule(user, workoutSetting_instance, workout_schedule):
     # Add this to get difficulty
    difficulty = workoutSetting_instance.difficulty if hasattr(workoutSetting_instance, 'difficulty') else 'intermediate'
    duration, rest = get_duration_and_rest(difficulty)
    start_date = workoutSetting_instance.current_week_start
    end_date = workoutSetting_instance.current_week_end
    week_number = workoutSetting_instance.current_week

    workout_week = WorkoutWeek.objects.create(
        user=user,
        week=week_number,
        start_date=start_date,
        end_date=end_date
    )

    workout_objects = []
    current_date = start_date

    while current_date <= end_date:
        day_name = current_date.strftime("%A")
        day_key = next((key for key in workout_schedule.keys() if key.lower() == day_name.lower()), None)

        if day_key:
            exercise_ids = workout_schedule[day_key]  # Get exercise list for the day

            for exercise_id in exercise_ids:
                exercise_instance = Exercise.objects.filter(id=exercise_id).first()
                workout_objects.append(Workout(
                    workout_week=workout_week,
                    exercise=exercise_instance,
                    date=current_date,
                    day=day_name,
                    status=Workout.StatusChoices.PENDING,
                    duration_seconds=duration,
                    rest_seconds=rest,
                ))

        current_date += timedelta(days=1)

    if workout_objects:
        with transaction.atomic():
            Workout.objects.bulk_create(workout_objects)


def set_workoutsetting(user_instance, data):
    exclude_equipment = data.get('exclude_equipment', [])
    exclude_equipment = None if not exclude_equipment else exclude_equipment

    # Create Workout Settings
    if data.get('generate_workout')==False:
        workout_setting = WorkoutSetting.objects.create(
            user=user_instance,
            generate_schedule=False,
        )
    else:
        workout_setting = WorkoutSetting.objects.create(
            user=user_instance,
            generate_schedule=True,
            current_week=1,
            current_week_start=date.today(),
            current_week_end=date.today() + timedelta(days=6),
            workout_days=data.get('workout_days', {}),
            difficulty=data.get('workout_difficulty', ""),
            exclude_equipment=exclude_equipment,
        )

def generate_schedule(workoutSetting_instance):
    try:
        # Load user workout settings
        exclude_equipment = json.loads(workoutSetting_instance.exclude_equipment) if workoutSetting_instance.exclude_equipment else []
        workout_days = json.loads(workoutSetting_instance.workout_days) if isinstance(workoutSetting_instance.workout_days, str) else workoutSetting_instance.workout_days
        difficulty = workoutSetting_instance.difficulty

        # Extract active workout days
        workout_days = [day for day, is_active in workout_days.items() if is_active]

        # Ensure correct order of days
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        workout_days.sort(key=lambda x: day_order.index(x))

        logger.info(f"Workout Days: {workout_days}")
        logger.info(f"Exclude Equipment: {exclude_equipment}")

        # Fetch exercises grouped by muscle
        exercises = Exercise.get_exercise_by_setting(difficulty, exclude_equipment)

        # Define muscle group priority
        muscle_priority = [
            "Abdominals", "Glutes", "Chest", "Shoulders", "Back",
            "Adductors", "Biceps", "Quadriceps", "Hamstrings", 
            "Abductors", "Trapezius", "Triceps", "Forearms",
            "Calves", "Shins", "Hip Flexors"
        ]

        # Sort muscles by priority
        sorted_muscles = sorted(exercises.keys(), key=lambda x: muscle_priority.index(x) if x in muscle_priority else len(muscle_priority))

        # Step 1: Assign each muscle group at least once before repeating
        day_muscle_count = {day: [] for day in workout_days}
        assigned_muscles = set()

        day_index = 0
        for muscle in sorted_muscles:
            if muscle in exercises:
                chosen_day = workout_days[day_index % len(workout_days)]
                day_muscle_count[chosen_day].append(muscle)
                assigned_muscles.add(muscle)
                day_index += 1  # Move to the next day

        # Step 2: Fill remaining slots (up to 5 per day) with additional muscles
        while any(len(day_muscle_count[d]) < 5 for d in workout_days):
            for muscle in sorted_muscles:
                available_days = [d for d in workout_days if len(day_muscle_count[d]) < 5]
                if available_days:
                    chosen_day = available_days[0]
                    if muscle not in day_muscle_count[chosen_day]:  # Avoid duplicates
                        day_muscle_count[chosen_day].append(muscle)

        # Step 3: Assign exercises to each muscle group
        prob = pulp.LpProblem("Workout_Schedule", pulp.LpMaximize)
        exercise_vars = {}

        for day in workout_days:
            for muscle in day_muscle_count[day]:
                if muscle in exercises:
                    selected_exercises = random.sample(exercises[muscle], min(len(exercises[muscle]), 2))
                    for ex in selected_exercises:
                        exercise_vars[(ex['id'], day)] = pulp.LpVariable(f"ex_{ex['id']}_d{day}", cat='Binary')

        # Constraints: At least 1 exercise per muscle per day, max 5 exercises per day
        for day in workout_days:
            relevant_exercises = [exercise_vars[(ex_id, day)] for (ex_id, day_) in exercise_vars if day_ == day]

            if relevant_exercises:
                prob += pulp.lpSum(relevant_exercises) >= min(len(day_muscle_count[day]), 5)  # At least 1 per muscle
                prob += pulp.lpSum(relevant_exercises) <= 6  # Allow up to 6 per day for flexibility

        # Solve the MILP problem
        prob.solve()

        # Debug solver status
        print("Solver Status:", pulp.LpStatus[prob.status])

        # Extract results
        schedule = {day: [] for day in workout_days}
        for (ex_id, day), var in exercise_vars.items():
            if var.varValue == 1:
                schedule[day].append(ex_id)

        return json.dumps(schedule, indent=4)

    except Exception as e:
        logger.error(f"Error in generate_schedule: {str(e)}", exc_info=True)
        return {"error": str(e)}

def user_logout (request) :
    if request.method == 'POST':
        try:
            #get session from cookies and logout user
            session = Session.objects.get(session_key=request.COOKIES['sessionid'])
            session.delete()

            return JsonResponse({"Message":"Log out successfully"},status=200)
        
        except Exception as e:
            print(str(e))
            return JsonResponse({"error": str(e)}, status=500)


def verify_session(request):
    if request.method == 'POST':
        #get session from cookies and logout user
        try:
            session = Session.objects.get(session_key=request.COOKIES['sessionid'])

            if session.expire_date >= timezone.now() : 
                return JsonResponse({"Message":"Session Valid"},status=200)
            else :
                return JsonResponse({"error":"Session expired"}, status=403)
        except Exception as e : 
            print(str(e))
            return JsonResponse({"error":str(e)},status=500)


def get_message_history(request):
    if request.method == 'GET' :
        try:
            #get session from cookies and get user id 
            session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
            user_id = session['_auth_user_id']

            user = User.objects.get(pk=user_id)

            #get all Messages with user_id in JSON format
            messages = Message.get_messages_by_user(user_id)

            return JsonResponse({"messages": json.loads(messages)}, status=200)

        except Exception as e:
            print(str(e))
            return JsonResponse({"error": str(e)}, status=500)

def get_user_workout_data(request):
    if request.method == 'GET':
        try:
            # Get session from cookies and extract user ID
            session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
            user_id = session['_auth_user_id']

            user_instance = User.objects.get(pk=user_id)

            # Fetch all workout weeks with related workouts and exercises
            workout_weeks = WorkoutWeek.objects.filter(user=user_instance).prefetch_related("workouts__exercise")

            user_workouts = []

            for week in workout_weeks:
                week_data = {
                    "week": week.week,
                    "workout_days": defaultdict(list)  # Group workouts by day
                }

                for workout in week.workouts.all():  # Use 'workouts' instead of 'workout_set'
                    week_data["workout_days"][workout.day].append({
                        "id" : workout.id,
                        "exercise_name": workout.exercise.exercise_name,
                        "status": workout.status,
                        "date": str(workout.date),  # Convert DateField to string for JSON
                        "short_video": workout.exercise.short_video,
                        "long_video": workout.exercise.long_video,
                        "duration_seconds": workout.duration_seconds,  # Add this
                        "rest_seconds": workout.rest_seconds, 
                    })

                # Convert defaultdict to normal dict before appending
                week_data["workout_days"] = dict(week_data["workout_days"])
                user_workouts.append(week_data)

            return JsonResponse({"workouts": user_workouts}, status=200)
        
        except Exception as e:
            print(str(e))
            return JsonResponse({"error": str(e)}, status=500)
        
def update_workout_status(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            workout_id = data.get("workout_id")
            new_status = data.get("status")

            if not workout_id or not new_status:
                return JsonResponse({"error": "Missing 'workout_id' or 'status'."}, status=400)

            workout = Workout.objects.get(pk=workout_id)
            workout.status = new_status
            workout.save()

            return JsonResponse({"message": "Workout status updated successfully."}, status=200)

        except Workout.DoesNotExist:
            return JsonResponse({"error": "Workout not found."}, status=404)
        except Exception as e:
            logger.error(f"Error in update workout status: {traceback.format_exc()}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=400)

@api_view(["GET"])
def get_today_workouts(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session.get('_auth_user_id')
        user_profile = UserProfile.objects.get(user__id=user_id)
        today = now().date()
        workouts = Workout.objects.filter(workout_week__user=user_profile.user, date=today).select_related('exercise')

        workout_list = []
        for w in workouts:
            workout_list.append({
                "id": w.id,
                "exercise_name": w.exercise.exercise_name,
                "short_video": w.exercise.short_video,
                "long_video": w.exercise.long_video,
                "duration_seconds": w.duration_seconds,
                "rest_seconds": w.rest_seconds,
                "status": w.status,
            })

        return JsonResponse({"workouts": workout_list})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
def save_workout_progress(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)

        data = request.data if hasattr(request, "data") else json.loads(request.body)
        workout_id = data.get('workout_id')
        set_number = data.get('set_number', 1)
        rep_number = data.get('rep_number', 1)
        seconds_left_in_rest = data.get('seconds_left_in_rest', 0)
        status = data.get('status', 'paused')

        if not workout_id:
            return Response({'error': 'Missing workout_id'}, status=400)
        
        progress, _ = WorkoutProgress.objects.update_or_create(
            user=user, workout_id=workout_id,
            defaults=dict(
                set_number=set_number,
                rep_number=rep_number,
                seconds_left_in_rest=seconds_left_in_rest,
                status=status
            )
        )
        return Response({'success': True}, status=200)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['GET'])
def get_workout_progress(request, workout_id):
    try:
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        progress = WorkoutProgress.objects.filter(user=user, workout_id=workout_id).first()
        if not progress:
            return Response({'progress': None})
        return Response({
            "progress": {
                "workout_id": progress.workout_id.id,
                "set_number": progress.set_number,
                "rep_number": progress.rep_number,
                "seconds_left_in_rest": progress.seconds_left_in_rest,
                "status": progress.status,
            }
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
def complete_workout_progress(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        data = request.data if hasattr(request, "data") else json.loads(request.body)
        workout_id = data.get('workout_id')

        WorkoutProgress.objects.filter(user=user, workout_id=workout_id).delete()
        # Optionally update Workout status to completed
        workout = Workout.objects.filter(id=workout_id).first()
        if workout:
            workout.status = "completed"
            workout.save()
        return Response({'success': True}, status=200)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
def user_message(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        #get session from cookies and get user id 
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session['_auth_user_id']

        user = User.objects.get(pk=user_id)

        #create message and save 
        message = Message.objects.create(user=user, message_text=data['message_text'], is_user=data['is_user'], created_at=data['created_at'])
        message.save()

        print("Received message", message)

        #call function to call llm here  
        llm_reply = call_llm(message.message_text)

        #save llm model to new message 
        llm_message = Message.objects.create(user=user, message_text=llm_reply, is_user=False, created_at=datetime.now().isoformat())
        llm_message.save()

        #return reply here 
        reply = {
            "id" : llm_message.id,
            "message_text" : llm_message.message_text,
            "is_user" : llm_message.is_user,
            "created_at": llm_message.created_at,
        }


    return JsonResponse({"reply" : reply,}, status = 200)


def call_llm(user_message:str):
    client = Groq(
        api_key=os.environ.get("GROQ_API_TOKEN"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        model="llama3-70b-8192",
        max_completion_tokens=200,
    )

    return chat_completion.choices[0].message.content

def  get_food_list(request):
    if request.method == 'GET':
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        food = FoodDatabase.objects.filter(Q(user=None) | Q(user=user)).values()

        return JsonResponse({"foodlist": list(food)},status = 200)

def add_meal(request): 
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
            user_id = session.get('_auth_user_id')

            logger.info(f"Received data: {data}, User ID: {user_id}")

            newEntry = FoodEntry.Insert_new_entry(
                user_id=user_id, 
                daily_nutrition_id=data.get('log_id'),
                meal_label=data.get('meal_type'),
                measurement_unit=data.get('measurement_unit'),
                food_id=data.get('food_id'),
                serving=data.get('serving'),
                weight=data.get('weight'),
            )

            return JsonResponse({"data": str(newEntry)}, status=200)
        except Exception as e:
            logger.error(f"Error in add_meal: {traceback.format_exc()}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


def edit_meal(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
            user_id = session.get('_auth_user_id')

            logger.info(f"Received data: {data}, User ID: {user_id}")

            editEntry = FoodEntry.edit_entry(
                log_id=data.get('log_id'), 
                meal_label=data.get('meal_label'),
                measurement_unit=data.get('measurement_unit'),
                food_id=data.get('food_id'),
                serving=data.get('serving'),
                weight=data.get('weight'),
            )

            return JsonResponse({"data": str(editEntry)}, status=200)
        except Exception as e:
            logger.error(f"Error in add_meal: {traceback.format_exc()}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


def delete_meal(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
            user_id = session.get('_auth_user_id')

            log_id = data.get('log_id')
            logger.info(f"Received data: {data}, User ID: {user_id}")

            # Get entry and food before deleting
            entry = FoodEntry.objects.get(id=log_id)
            food_obj = entry.food_id

            # Check manual before deletion
            is_manual = getattr(food_obj, 'unit', '') == 'manual' or getattr(food_obj, 'categories', '') == 'Manual'

            # Use your custom method to delete (this handles DailyNutrition subtraction!)
            FoodEntry.delete_entry(log_id=log_id)

            # Now safe to delete food if manual
            if is_manual:
                food_obj.delete()

            return JsonResponse({"success": True}, status=200)
        except FoodEntry.DoesNotExist:
            return JsonResponse({"error": "FoodEntry does not exist"}, status=404)
        except Exception as e:
            logger.error(f"Error in delete_meal: {traceback.format_exc()}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)

def dashboard_data(request):
    if request.method == 'GET':
        try:
            today = now().date()
            session_key = request.COOKIES.get('sessionid')
            session = Session.objects.get(session_key=session_key).get_decoded()
            user_id = session.get('_auth_user_id')

            user = User.objects.get(id=user_id)
            user_profile = UserProfile.objects.get(user=user)

            # Get today's DailyNutrition
            today_nutrition = DailyNutrition.get_user_daily_nutrition(user_profile)
            workout_summary = Workout.get_today_summary(user_profile)

            response_data = {
                "date": str(today_nutrition.date),
                "target_calories": today_nutrition.target_calories,
                "total_calories": today_nutrition.total_calories,
                "total_breakfast_cal": today_nutrition.total_breakfast_cal,
                "total_lunch_cal": today_nutrition.total_lunch_cal,
                "total_dinner_cal": today_nutrition.total_dinner_cal,
                "total_snack_cal": today_nutrition.total_snack_cal,
                "workout_total": workout_summary["total"],
                "workout_completed": workout_summary["completed"],
                "workout_skipped": workout_summary["skipped"],
            }

            return JsonResponse({"data": response_data}, status=200)

        except Exception as e:
            logger.error(f"Error in dashboard_data: {traceback.format_exc()}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)

@api_view(['POST'])
def add_new_food(request):
    try:
        data = request.data
        # Required fields
        required_fields = [
            "product_name", "unit",
            "calories_per_100g", "protein_per_100g", "fat_per_100g",
            "carbohydrate_per_100g", "sugar_per_100g", "sodium_per_100g",
            "calories_per_unit", "protein_per_unit", "fat_per_unit",
            "carbohydrate_per_unit", "sugar_per_unit", "sodium_per_unit"
        ]
        for f in required_fields:
            if f not in data or data[f] in [None, ""]:
                return Response({"error": f"{f} is required"}, status=400)

        # Default image if not sent by user
        image_url = data.get("image_url") or None

        food = FoodDatabase.objects.create(
            product_name = data["product_name"],
            categories = "User Added",
            unit = data["unit"],
            calories_per_100g = data["calories_per_100g"],
            protein_per_100g = data["protein_per_100g"],
            fat_per_100g = data["fat_per_100g"],
            carbohydrate_per_100g = data["carbohydrate_per_100g"],
            sugar_per_100g = data["sugar_per_100g"],
            sodium_per_100g = data["sodium_per_100g"],
            calories_per_unit = data["calories_per_unit"],
            protein_per_unit = data["protein_per_unit"],
            fat_per_unit = data["fat_per_unit"],
            carbohydrate_per_unit = data["carbohydrate_per_unit"],
            sugar_per_unit = data["sugar_per_unit"],
            sodium_per_unit = data["sodium_per_unit"],
            image_url = image_url,
        )

        # Return the created food as dict/JSON for frontend update
        food_dict = {
            "id": food.id,
            "product_name": food.product_name,
            "categories": food.categories,
            "unit": food.unit,
            "calories_per_100g": str(food.calories_per_100g),
            "protein_per_100g": str(food.protein_per_100g),
            "fat_per_100g": str(food.fat_per_100g),
            "carbohydrate_per_100g": str(food.carbohydrate_per_100g),
            "sugar_per_100g": str(food.sugar_per_100g),
            "sodium_per_100g": str(food.sodium_per_100g),
            "calories_per_unit": str(food.calories_per_unit),
            "protein_per_unit": str(food.protein_per_unit),
            "fat_per_unit": str(food.fat_per_unit),
            "carbohydrate_per_unit": str(food.carbohydrate_per_unit),
            "sugar_per_unit": str(food.sugar_per_unit),
            "sodium_per_unit": str(food.sodium_per_unit),
            "image_url": food.image_url,
            "is_user_added": True,
        }
        return Response({"success": True, "food": food_dict}, status=201)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

def add_manual_entry(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
        session = request.COOKIES.get('sessionid')
        if not session:
            return JsonResponse({'error': 'No session found'}, status=401)

        # Example: Get user ID from session (adjust as needed)
        from django.contrib.sessions.models import Session
        s = Session.objects.get(session_key=session).get_decoded()
        user_id = s.get('_auth_user_id')

        user = User.objects.get(id=user_id)
        user_profile = UserProfile.objects.get(user=user)
        daily_nutrition = DailyNutrition.objects.get(id=data['log_id'])

        # Required fields from frontend:
        description = data.get('description')   # e.g. "Quick egg sandwich"
        meal_type   = data.get('meal_type')     # e.g. "lunch"
        calories    = data.get('calories')
        # Optional macros:
        protein = data.get('protein') or 0
        fats    = data.get('fats') or 0
        carbs   = data.get('carbs') or 0
        sugar   = data.get('sugar') or 0
        salts   = data.get('salts') or 0

        # --- 1. Create a FoodDatabase entry (minimal required fields) ---
        # You can set a special category so you can filter "Manual Entries" later if needed.
        food_obj = FoodDatabase.objects.create(
            product_name=description,
            categories="Manual",
            calories_per_unit=Decimal(calories),
            protein_per_unit=Decimal(protein),
            fat_per_unit=Decimal(fats),
            carbohydrate_per_unit=Decimal(carbs),
            sugar_per_unit=Decimal(sugar),
            sodium_per_unit=Decimal(salts),
            unit="manual",       # You can customize this label
        )

        # --- 2. Create FoodEntry referencing that new food_id ---
        food_entry = FoodEntry.objects.create(
            user_profile=user_profile,
            daily_nutrition=daily_nutrition,
            date=daily_nutrition.date,
            food_id=food_obj,
            label=meal_type,
            measurement_unit="serving",     # Since all values are per "unit" (manual input)
            serving=1,                      # Always 1 for a quick entry
            weight=None,                    # Not used for manual entry
            calories=Decimal(calories),
            carbs=Decimal(carbs),
            protein=Decimal(protein),
            fats=Decimal(fats),
            sugar=Decimal(sugar),
            salts=Decimal(salts)
        )

        return JsonResponse({
            "success": True,
            "food_entry_id": food_entry.id,
            "food_id": food_obj.id,
            "food_name": food_obj.product_name,
        }, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def edit_manual_entry(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    try:
        data = json.loads(request.body)

        log_id = data.get('log_id')
        manual_description = data.get('manual_description')
        meal_type = data.get('meal_type')
        calories = Decimal(str(data.get('calories', 0)))
        protein = Decimal(str(data.get('protein', 0)))
        fats = Decimal(str(data.get('fats', 0)))
        carbs = Decimal(str(data.get('carbs', 0)))
        sugar = Decimal(str(data.get('sugar', 0)))
        salts = Decimal(str(data.get('salts', 0)))

        if not log_id or not manual_description or not meal_type or calories is None:
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        entry = FoodEntry.objects.get(id=log_id)
        daily_nutrition = entry.daily_nutrition
        food_obj = entry.food_id

        # --- 1. Subtract old nutrition from daily totals ---
        FoodEntry.subtract_nutrition_from_daily(
            daily_nutrition,
            entry.label,
            entry.calories,
            entry.carbs,
            entry.protein,
            entry.fats,
            entry.salts,
            entry.sugar
        )

        # --- 2. Update FoodDatabase (manual entry) ---
        food_obj.product_name = manual_description
        food_obj.calories_per_unit = calories
        food_obj.protein_per_unit = protein
        food_obj.fat_per_unit = fats
        food_obj.carbohydrate_per_unit = carbs
        food_obj.sugar_per_unit = sugar
        food_obj.sodium_per_unit = salts
        food_obj.save()

        # --- 3. Update FoodEntry with new values ---
        entry.label = meal_type
        entry.food_id = food_obj
        entry.calories = calories
        entry.protein = protein
        entry.fats = fats
        entry.carbs = carbs
        entry.sugar = sugar
        entry.salts = salts

        entry.save()  # This will add the new nutrition back to daily totals

        return JsonResponse({'success': True}, status=200)
    except FoodEntry.DoesNotExist:
        return JsonResponse({'error': 'Food entry not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
@api_view(['POST'])
def analyze_custom_recipe(request):
    try:
        # Get authenticated user from session
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)

        # Extract request data
        data = request.data
        recipe_name = data.get("recipe_name")
        ingredients = data.get("ingredients")  # Should be a list

        if not recipe_name or not ingredients:
            return JsonResponse({"error": "Missing recipe name or ingredients"}, status=400)

        # --- Build Prompt for LLM ---
        prompt = (
            f"You are a nutrition expert. Given a list of ingredients for a recipe, "
            f"the recipe name is: \"{recipe_name}\". "
            "Analyze the recipe and return the following JSON format:\n\n"
            "{\n"
            "  \"categories\": \"Custom Recipe\",\n"
            "  \"unit\": str,  # Estimated serving label (e.g., 'cup', 'slice', 'bowl')\n"
            "  \"calories_per_100g\": float,\n"
            "  \"protein_per_100g\": float,\n"
            "  \"fat_per_100g\": float,\n"
            "  \"carbohydrate_per_100g\": float,\n"
            "  \"sugar_per_100g\": float,\n"
            "  \"sodium_per_100g\": float,\n"
            "  \"calories_per_unit\": float,\n"
            "  \"protein_per_unit\": float,\n"
            "  \"fat_per_unit\": float,\n"
            "  \"carbohydrate_per_unit\": float,\n"
            "  \"sugar_per_unit\": float,\n"
            "  \"sodium_per_unit\": float\n"
            "}\n\n"
            "Only return the JSON. No comments or explanation.\n"
            "Example units: 'slice', 'plate', 'cup', 'sandwich', 'bowl'\n\n"
            "Ingredients:\n" + "\n".join(ingredients)
        )

        # --- Call Groq LLM ---
        client = Groq(api_key=os.environ.get("GROQ_API_TOKEN"))
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=400,
        )
        response_content = chat_completion.choices[0].message.content.strip()
        print("LLM RAW RESPONSE:", response_content)  # DEBUG

        # --- Parse LLM JSON ---
        try:
            nutrition_data = json.loads(response_content)
        except json.JSONDecodeError:
            print("LLM response could not be parsed. Raw:", response_content)  # DEBUG
            return JsonResponse({"error": "LLM response could not be parsed. Raw: " + response_content}, status=500)

        print("nutrition_data dict:", nutrition_data)  # DEBUG

        # --- Save to FoodDatabase ---
        food = FoodDatabase.objects.create(
            user=user,
            product_name=str(recipe_name),  # ensure string, no comma!
            categories=nutrition_data.get("categories", "Custom Recipe"),
            unit=nutrition_data.get("unit", ""),
            calories_per_100g=float(nutrition_data.get("calories_per_100g", 0) or 0),
            protein_per_100g=float(nutrition_data.get("protein_per_100g", 0) or 0),
            fat_per_100g=float(nutrition_data.get("fat_per_100g", 0) or 0),
            carbohydrate_per_100g=float(nutrition_data.get("carbohydrate_per_100g", 0) or 0),
            sugar_per_100g=float(nutrition_data.get("sugar_per_100g", 0) or 0),
            sodium_per_100g=float(nutrition_data.get("sodium_per_100g", 0) or 0),
            calories_per_unit=float(nutrition_data.get("calories_per_unit", 0) or 0),
            protein_per_unit=float(nutrition_data.get("protein_per_unit", 0) or 0),
            fat_per_unit=float(nutrition_data.get("fat_per_unit", 0) or 0),
            carbohydrate_per_unit=float(nutrition_data.get("carbohydrate_per_unit", 0) or 0),
            sugar_per_unit=float(nutrition_data.get("sugar_per_unit", 0) or 0),
            sodium_per_unit=float(nutrition_data.get("sodium_per_unit", 0) or 0),
        )

        # --- Link to UserRecipe ---
        from .models import UserRecipe
        UserRecipe.objects.create(
            user=user,
            name=str(recipe_name),
            ingredients=ingredients,
            food_ref=food
        )

        print("SAVED FOOD TO DB:", food.__dict__)  # DEBUG

        return JsonResponse({
            "success": True,
            "food": {
                "id": food.id,
                "product_name": food.product_name,
                "unit": food.unit,
                "calories_per_unit": str(food.calories_per_unit),
                "protein_per_unit": str(food.protein_per_unit),
                "fat_per_unit": str(food.fat_per_unit),
                "carbohydrate_per_unit": str(food.carbohydrate_per_unit),
                "sugar_per_unit": str(food.sugar_per_unit),
                "sodium_per_unit": str(food.sodium_per_unit),
                "calories_per_100g": str(food.calories_per_100g),
                "protein_per_100g": str(food.protein_per_100g),
                "fat_per_100g": str(food.fat_per_100g),
                "carbohydrate_per_100g": str(food.carbohydrate_per_100g),
                "sugar_per_100g": str(food.sugar_per_100g),
                "sodium_per_100g": str(food.sodium_per_100g),
            }
        }, status=200)

    except Exception as e:
        import traceback
        print("Error in analyze_custom_recipe:", traceback.format_exc())  # DEBUG
        return JsonResponse({"error": str(e)}, status=500)

@api_view(['PUT'])
def edit_custom_recipe(request, recipe_id):
    try:
        # 1. Get authenticated user from session
        session = Session.objects.get(session_key=request.COOKIES['sessionid']).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)

        # 2. Look up recipe (owned by user)
        try:
            recipe = UserRecipe.objects.get(id=recipe_id, user=user)
        except UserRecipe.DoesNotExist:
            return JsonResponse({"error": "Recipe not found."}, status=404)

        # 3. Parse incoming data
        data = request.data
        recipe_name = data.get("recipe_name")
        ingredients = data.get("ingredients")  # Should be a list

        if not recipe_name or not ingredients:
            return JsonResponse({"error": "Missing recipe name or ingredients."}, status=400)

        # --- Build Prompt for LLM ---
        prompt = (
            f"You are a nutrition expert. Given a list of ingredients for a recipe, "
            f"the recipe name is: \"{recipe_name}\". "
            "Analyze the recipe and return the following JSON format:\n\n"
            "{\n"
            "  \"categories\": \"Custom Recipe\",\n"
            "  \"unit\": str,  # Estimated serving label (e.g., 'cup', 'slice', 'bowl')\n"
            "  \"calories_per_100g\": float,\n"
            "  \"protein_per_100g\": float,\n"
            "  \"fat_per_100g\": float,\n"
            "  \"carbohydrate_per_100g\": float,\n"
            "  \"sugar_per_100g\": float,\n"
            "  \"sodium_per_100g\": float,\n"
            "  \"calories_per_unit\": float,\n"
            "  \"protein_per_unit\": float,\n"
            "  \"fat_per_unit\": float,\n"
            "  \"carbohydrate_per_unit\": float,\n"
            "  \"sugar_per_unit\": float,\n"
            "  \"sodium_per_unit\": float\n"
            "}\n\n"
            "Only return the JSON. No comments or explanation.\n"
            "Example units: 'slice', 'plate', 'cup', 'sandwich', 'bowl'\n\n"
            "Ingredients:\n" + "\n".join(ingredients)
        )

        # --- Call Groq LLM ---
        client = Groq(api_key=os.environ.get("GROQ_API_TOKEN"))
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=400,
        )
        response_content = chat_completion.choices[0].message.content.strip()
        print("LLM RAW RESPONSE (EDIT):", response_content)  # DEBUG

        # --- Parse LLM JSON ---
        try:
            nutrition_data = json.loads(response_content)
        except json.JSONDecodeError:
            print("LLM response could not be parsed. Raw:", response_content)  # DEBUG
            return JsonResponse({"error": "LLM response could not be parsed. Raw: " + response_content}, status=500)

        print("nutrition_data dict (EDIT):", nutrition_data)  # DEBUG

        # --- Update FoodDatabase linked to this recipe
        food = recipe.food_ref

        food.product_name = str(recipe_name)  # ensure string, no comma!
        food.categories = nutrition_data.get("categories", "Custom Recipe")
        food.unit = nutrition_data.get("unit", "")
        food.calories_per_100g = float(nutrition_data.get("calories_per_100g", 0) or 0)
        food.protein_per_100g = float(nutrition_data.get("protein_per_100g", 0) or 0)
        food.fat_per_100g = float(nutrition_data.get("fat_per_100g", 0) or 0)
        food.carbohydrate_per_100g = float(nutrition_data.get("carbohydrate_per_100g", 0) or 0)
        food.sugar_per_100g = float(nutrition_data.get("sugar_per_100g", 0) or 0)
        food.sodium_per_100g = float(nutrition_data.get("sodium_per_100g", 0) or 0)
        food.calories_per_unit = float(nutrition_data.get("calories_per_unit", 0) or 0)
        food.protein_per_unit = float(nutrition_data.get("protein_per_unit", 0) or 0)
        food.fat_per_unit = float(nutrition_data.get("fat_per_unit", 0) or 0)
        food.carbohydrate_per_unit = float(nutrition_data.get("carbohydrate_per_unit", 0) or 0)
        food.sugar_per_unit = float(nutrition_data.get("sugar_per_unit", 0) or 0)
        food.sodium_per_unit = float(nutrition_data.get("sodium_per_unit", 0) or 0)
        food.save()
        print("UPDATED FOOD IN DB:", food.__dict__)  # DEBUG

        # --- Update UserRecipe (name & ingredients)
        recipe.name = str(recipe_name)
        recipe.ingredients = ingredients
        recipe.save()

        # --- Return the updated recipe + nutrition
        return JsonResponse({
            "success": True,
            "recipe": {
                "id": recipe.id,
                "name": recipe.name,
                "ingredients": recipe.ingredients,
                "food_ref_id": food.id,
                "unit": food.unit,
                "calories_per_100g": str(food.calories_per_100g),
                "protein_per_100g": str(food.protein_per_100g),
                "fat_per_100g": str(food.fat_per_100g),
                "carbohydrate_per_100g": str(food.carbohydrate_per_100g),
                "sugar_per_100g": str(food.sugar_per_100g),
                "sodium_per_100g": str(food.sodium_per_100g),
                "calories_per_unit": str(food.calories_per_unit),
                "protein_per_unit": str(food.protein_per_unit),
                "fat_per_unit": str(food.fat_per_unit),
                "carbohydrate_per_unit": str(food.carbohydrate_per_unit),
                "sugar_per_unit": str(food.sugar_per_unit),
                "sodium_per_unit": str(food.sodium_per_unit),
            }
        }, status=200)

    except Exception as e:
        import traceback
        print("Error in edit_custom_recipe:", traceback.format_exc())  # DEBUG
        return JsonResponse({"error": str(e)}, status=500)

    
@api_view(["GET"])
def get_user_recipes(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)

        recipes = UserRecipe.objects.filter(user=user)

        data = []
        for r in recipes:
            data.append({
                "id": r.id,
                "name": r.name,
                "ingredients": r.ingredients,
                "food_ref_id": r.food_ref.id,
                "calories_per_unit": str(r.food_ref.calories_per_unit),
                "unit": r.food_ref.unit,
                # per 100g fields (all as string for safety with decimals)
                "calories_per_100g": str(r.food_ref.calories_per_100g) if r.food_ref.calories_per_100g is not None else "",
                "protein_per_100g": str(r.food_ref.protein_per_100g) if r.food_ref.protein_per_100g is not None else "",
                "fat_per_100g": str(r.food_ref.fat_per_100g) if r.food_ref.fat_per_100g is not None else "",
                "carbohydrate_per_100g": str(r.food_ref.carbohydrate_per_100g) if r.food_ref.carbohydrate_per_100g is not None else "",
                "sugar_per_100g": str(r.food_ref.sugar_per_100g) if r.food_ref.sugar_per_100g is not None else "",
                "sodium_per_100g": str(r.food_ref.sodium_per_100g) if r.food_ref.sodium_per_100g is not None else "",
            })


        return JsonResponse({"recipes": data})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
@api_view(["POST"])
def delete_user_recipe(request):
    try:
        session = request.COOKIES.get('sessionid')
        if not session:
            return JsonResponse({"error": "Not authenticated."}, status=401)
        session = Session.objects.get(session_key=session).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)

        recipe_id = request.data.get("recipe_id")
        if not recipe_id:
            return JsonResponse({"error": "Missing recipe_id"}, status=400)
        
        recipe = UserRecipe.objects.get(id=recipe_id, user=user)
        food_ref = recipe.food_ref

        # Delete the recipe and its linked food entry
        recipe.delete()
        if food_ref:
            food_ref.delete()
        return JsonResponse({"success": True})
    except UserRecipe.DoesNotExist:
        return JsonResponse({"error": "Recipe not found."}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@api_view(['GET'])
def check_meal_logged(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        user_profile = UserProfile.objects.get(user=user)
        today = now().date()
        meal_type = request.GET.get('meal_type', '').lower()
        if not meal_type or meal_type not in ['breakfast', 'lunch', 'dinner', 'snack']:
            return JsonResponse({'error': 'Invalid meal_type'}, status=400)
        daily = DailyNutrition.get_user_daily_nutrition(user_profile)
        has_logged = daily.food_entries.filter(label=meal_type).exists()
        return JsonResponse({'logged': has_logged})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@api_view(['POST'])
def request_change_email(request):
    try:
        data = request.data
        current_email = data.get('current_email')
        new_email = data.get('new_email')
        password = data.get('password')

        # Add this check:
        if current_email == new_email:
            return Response({"error": "New email must be different from your current email."}, status=400)

        user = User.objects.filter(email=current_email).first()
        if not user or not user.check_password(password):
            return Response({"error": "Incorrect password or user not found"}, status=400)

        if User.objects.filter(email=new_email).exists():
            return Response({"error": "Email already in use"}, status=400)

        code = random.randint(100000, 999999)
        cache.set(f'change_email_{user.id}', (new_email, code), timeout=600) # 10 min

        send_mail(
            subject="NutriMinder: Change Email Verification",
            message=f"Your verification code is: {code}",
            from_email="noreply@nutriminder.com",
            recipient_list=[new_email],
        )

        return Response({"message": "Verification code sent."})
    except Exception as e:
        import traceback
        print("Exception occurred:", e)
        print(traceback.format_exc())
        return Response({"error": str(e)}, status=500)
    

@csrf_exempt
@api_view(['POST'])
def confirm_change_email(request):
    try:
        data = request.data
        new_email = data.get('new_email')
        code = int(data.get('code'))
        user = request.user

        cache_key = f'change_email_{user.id}'
        stored = cache.get(cache_key)
        if not stored:
            return Response({"error": "No verification in progress"}, status=400)
        stored_email, stored_code = stored

        if stored_email != new_email or int(stored_code) != int(code):
            return Response({"error": "Incorrect code or email"}, status=400)

        # All OK: Update email
        user.email = new_email
        user.save()
        cache.delete(cache_key)
        return Response({"message": "Email updated"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@csrf_exempt
@api_view(['POST'])
def change_password(request):
    try:
        data = request.data
        old_password = data.get('old_password')
        new_password = data.get('new_password')
        user = request.user
        if not user.check_password(old_password):
            return Response({"error": "Current password is incorrect"}, status=400)
        user.set_password(new_password)
        user.save()
        return Response({"message": "Password changed"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

def user_profile(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        return JsonResponse({
            "username": user.username,
            "email": user.email,
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
@api_view(['GET'])
def user_weight_logs(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        user_profile = UserProfile.objects.get(user=user)

        # Check if logs exist
        logs_qs = WeightLog.objects.filter(user_profile=user_profile)
        if logs_qs.count() == 0:
            # Get or create UserCalories (for updated_at date)
            user_calories, _ = UserCalories.objects.get_or_create(user_profile=user_profile)
            initial_log_date = user_calories.updated_at.date() if user_calories.updated_at else date.today()
            # Add the initial log
            WeightLog.objects.create(
                user_profile=user_profile,
                date=initial_log_date,
                weight=user_profile.current_weight_kg
            )

        logs = WeightLog.objects.filter(user_profile=user_profile).order_by('-date')[:5]
        logs_data = [{
            "id": log.id,
            "date": log.date.strftime("%d/%m/%Y"),
            "weight": float(log.weight)
        } for log in reversed(logs)]  # Return oldest first for chart

        return Response({"logs": logs_data}, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@api_view(['POST'])
def add_weight_log(request):
    try:
        # --- user/session code as before ---
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        user_profile = UserProfile.objects.get(user=user)

        data = request.data
        date_str = data.get('date')
        weight_str = data.get('weight')

        if not date_str or not weight_str:
            return Response({"error": "Missing date or weight"}, status=400)

        # Handle flexible date formats
        try:
            try:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
            except ValueError:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return Response({"error": "Date format must be dd/MM/yyyy or yyyy-MM-dd"}, status=400)

        try:
            weight = float(weight_str)
        except Exception:
            return Response({"error": "Weight must be a number"}, status=400)
      
        # Add the new log for the selected date
        log, created = WeightLog.objects.get_or_create(
            user_profile=user_profile,
            date=date_obj,
            defaults={"weight": weight}
        )
        if not created:
            log.weight = weight
            log.save()
            msg = "Log updated."
        else:
            msg = "Log created."

        # Check if this is the latest date, then update UserProfile and calories
        latest_log = WeightLog.objects.filter(user_profile=user_profile).order_by('-date').first()
        msg = None
        calories_updated = False
        if latest_log and latest_log.date == date_obj:
            user_profile.current_weight_kg = weight
            user_profile.save()
            # Recalculate TDEE and UserCalories
            user_calories, created = UserCalories.objects.get_or_create(user_profile=user_profile)
            new_tdee = UserCalories.calculate_tdee(user_profile)
            user_calories.tdee = new_tdee
            user_calories.target_calories = new_tdee + user_calories.calorie_adjustment
            user_calories.save()
            calories_updated = True
            msg = "🎉 Weight updated! Calories recalculated!"
        else:
            msg = "Weight log added."

        # Get user's goal and target
        goal = user_profile.weight_goal
        target_weight = user_profile.target_weight_kg
        goal_met = False
        encouraging_msg = None

        # Compare with previous log for progress message
        prev_log = WeightLog.objects.filter(user_profile=user_profile, date__lt=date_obj).order_by('-date').first()
        progress_msg = None

        # Proper string matching for goal
        if prev_log:
            if weight < prev_log.weight:
                if goal == UserProfile.Goal.WEIGHT_LOSS:
                    encouraging_msg = "🔥 Amazing! You're making progress towards your goal weight!"
                elif goal == UserProfile.Goal.MAINTAIN:
                    encouraging_msg = "You lost some weight. Make sure this is intended!"
                elif goal == UserProfile.Goal.WEIGHT_GAIN:
                    encouraging_msg = "Heads up! You lost weight, but your goal is to gain."
                progress_msg = "Great job! You lost weight since your last log."
            elif weight > prev_log.weight:
                if goal == UserProfile.Goal.WEIGHT_GAIN:
                    encouraging_msg = "🔥 Great work! You're gaining towards your goal!"
                elif goal == UserProfile.Goal.MAINTAIN:
                    encouraging_msg = "You gained some weight. Make sure this is intended!"
                elif goal == UserProfile.Goal.WEIGHT_LOSS:
                    encouraging_msg = "Heads up! You gained weight, but your goal is to lose."
                progress_msg = "Heads up! You gained some weight since last time."
            else:
                progress_msg = "Weight unchanged from last log."
        else:
            progress_msg = "First log!"

        # If user meets/exceeds their goal, switch to maintain
        if goal == UserProfile.Goal.WEIGHT_LOSS and weight <= target_weight:
            user_profile.weight_goal = UserProfile.Goal.MAINTAIN
            user_calories, _ = UserCalories.objects.get_or_create(user_profile=user_profile)
            user_calories.calorie_adjustment = 0
            new_tdee = UserCalories.calculate_tdee(user_profile)
            user_calories.tdee = new_tdee
            user_calories.target_calories = new_tdee
            user_profile.save()
            user_calories.save()
            goal_met = True
            encouraging_msg = "🎯 Goal achieved! You're now set to maintain your weight."
        elif goal == UserProfile.Goal.WEIGHT_GAIN and weight >= target_weight:
            user_profile.weight_goal = UserProfile.Goal.MAINTAIN
            user_calories, _ = UserCalories.objects.get_or_create(user_profile=user_profile)
            user_calories.calorie_adjustment = 0
            new_tdee = UserCalories.calculate_tdee(user_profile)
            user_calories.tdee = new_tdee
            user_calories.target_calories = new_tdee
            user_profile.save()
            user_calories.save()
            goal_met = True
            encouraging_msg = "🎯 Goal achieved! You're now set to maintain your weight."

        return Response({
            "success": True,
            "message": msg,
            "progress_message": progress_msg,
            "calories_updated": calories_updated,
            "encouraging_message": encouraging_msg,
            "goal_met": goal_met,
        }, status=201)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return Response({"error": str(e)}, status=500)


@api_view(['GET', 'PATCH'])
def user_profile_stats(request):
    try:
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        profile = UserProfile.objects.get(user=user)
        
        if request.method == 'GET':
            height = float(profile.height_cm)
            weight = float(profile.current_weight_kg)
            target = float(profile.target_weight_kg)
            bmi = round(weight / ((height / 100) ** 2), 2) if height > 0 else None
            goal = profile.weight_goal
            return Response({
                "height_cm": height,
                "current_weight_kg": weight,
                "target_weight_kg": target,
                "current_bmi": bmi,
                "goal": goal,
            })
        
        if request.method == 'PATCH':
            data = request.data
            updated_weight = None

            if 'height_cm' in data:
                profile.height_cm = float(data['height_cm'])
            if 'current_weight_kg' in data:
                profile.current_weight_kg = float(data['current_weight_kg'])
                updated_weight = profile.current_weight_kg
            if 'target_weight_kg' in data:
                profile.target_weight_kg = float(data['target_weight_kg'])
            if 'goal' in data:
                profile.weight_goal = data['goal']
                # Always align target weight with current weight if "maintain"
                if data['goal'] == 'maintain':
                    profile.target_weight_kg = profile.current_weight_kg

            profile.save()

            # If weight updated, update today's log
            if updated_weight is not None:
                today = date.today()
                log, created = WeightLog.objects.update_or_create(
                    user_profile=profile,
                    date=today,
                    defaults={'weight': updated_weight}
                )

            return Response({"success": True})

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def weekly_progress(request):
    try:
        # --- User/session logic ---
        session = Session.objects.get(session_key=request.COOKIES.get('sessionid')).get_decoded()
        user_id = session.get('_auth_user_id')
        user = User.objects.get(id=user_id)
        profile = UserProfile.objects.filter(user=user).first()
        if not profile:
            return Response({"error": "Profile not found"}, status=404)

        week_start_str = request.query_params.get('week_start')
        if not week_start_str:
            return Response({"error": "week_start param required"}, status=400)
        try:
            week_start = datetime.strptime(week_start_str, "%Y-%m-%d").date()
        except Exception:
            return Response({"error": "Invalid date format for week_start (YYYY-MM-DD)"}, status=400)

        week_dates = [week_start + timedelta(days=i) for i in range(7)]
        week_end = week_dates[-1]

        # Weight logs for week
        weight_logs = {
            wl.date: wl.weight
            for wl in profile.weight_logs.filter(date__range=(week_start, week_end))
        }

        # Nutrition logs for week
        nutrition_logs = {
            dn.date: dn for dn in profile.daily_nutrition.filter(date__range=(week_start, week_end))
        }
        try:
            default_target_calories = profile.calories.target_calories
        except Exception:
            default_target_calories = 2000

        # Compose data
        weight_result = []
        cal_result = []
        for d in week_dates:
            weight_result.append({
                "date": d.strftime("%Y-%m-%d"),
                "weight": float(weight_logs[d]) if d in weight_logs else None,
            })
            daily_nut = nutrition_logs.get(d)
            cal_result.append({
                "date": d.strftime("%Y-%m-%d"),
                "total_calories": daily_nut.total_calories if daily_nut else 0,
                "target_calories": daily_nut.target_calories if daily_nut and daily_nut.target_calories else default_target_calories,
            })
        return Response({
            "weight": weight_result,
            "calories": cal_result,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return Response({"error": str(e)}, status=500)