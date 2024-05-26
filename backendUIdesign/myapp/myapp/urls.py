"""
URL configuration for myapp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]


# urls.py
from django.urls import path
from .views import get_analysis_results

urlpatterns = [
    path('api/analysis-results/', get_analysis_results, name='analysis-results'),
]



from django.urls import path
from .views import market_data_visualization
#<!DOCTYPE html>
#<html lang="en">
#<head>
   # <meta charset="UTF-8">
    #<title>Market Data Visualization</title>
    #{{ script|safe }}
#</head>
#<body>
  #  <h1>Market Data Visualization</h1>
   # {{ div|safe }}
#</body>
#</html>

urlpatterns = [
    path('market-data-visualization/', market_data_visualization, name='market_data_visualization'),
    # Add other URL patterns as needed
]
