from django.urls import path 
from . import views

urlpatterns = [
    path('', views.myApp),
    path('products/api/', views.getData),
    path('login/api/', views.user_login),
    path('logout/api/',views.user_logout),
    path('signup/verify/', views.verify_username_email_and_send_code),
    path('signup/api/',views.user_signup),
    path('verify_session/api/',views.verify_session),
    path('message_history/api/', views.get_message_history),
    path('workout_data/api/',views.get_user_workout_data),
    path('today_workouts/', views.get_today_workouts,),
    path('workout_update_status/api/', views.update_workout_status),
    path('save_workout_progress/', views.save_workout_progress,),
    path('get_workout_progress/<int:workout_id>/', views.get_workout_progress,),
    path('complete_workout/', views.complete_workout_progress,),
    path('message/api/', views.user_message),
    path('meal_history/api/', views.meal_history),
    path('food_list/api/',views.get_food_list),
    path('meal_entry/api/',views.add_meal),
    path('meal_edit/api/',views.edit_meal),
    path('meal_delete/api/', views.delete_meal),
    path("food_list/api/add/", views.add_new_food),
    path('manual_meal_entry/api/', views.add_manual_entry),
    path('manual_meal_edit/api/', views.edit_manual_entry),
    path('custom_recipe/analyze/', views.analyze_custom_recipe),
    path('custom_recipe/edit/<int:recipe_id>/', views.edit_custom_recipe,),
    path("custom_recipe/list/", views.get_user_recipes),
    path("custom_recipe/delete/", views.delete_user_recipe,),
    path('user_dashboard/api/', views.dashboard_data),
    path('check_meal_logged/', views.check_meal_logged),

    path('api/user_profile/', views.user_profile,),
    path('auth/request_change_email/', views.request_change_email),
    path('auth/confirm_change_email/', views.confirm_change_email),
    path('auth/change_password/', views.change_password),

    path('user_weight_logs/api/', views.user_weight_logs),
    path('add_weight_log/api/', views.add_weight_log),

    path('profile_stats/', views.user_profile_stats),
    path('progress/weekly/', views.weekly_progress),

]
