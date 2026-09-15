from django.urls import path
from . import views

urlpatterns = [
    # เมื่อเข้าหน้าแรก ให้เรียกฟังก์ชัน recommend_pc ใน views.py
    path('', views.recommend_pc, name='home'),
    path('custom-builder/', views.custom_builder_view, name='custom_builder'),
    path('api/analyze-build/', views.api_analyze_build, name='api_analyze_build'),
]