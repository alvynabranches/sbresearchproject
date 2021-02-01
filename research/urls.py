from django.urls import path
from .views import index, branch_prediction, college_prediction, placement_prediction

urlpatterns = [
    path('', index), 
    path('branch_prediction', branch_prediction), 
    path('college_prediction', college_prediction), 
    path('placement_prediction', placement_prediction)
]
