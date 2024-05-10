
from django.db import models
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from django.db.models import Count
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as ps



class MarketData(models.Model):
    category = models.CharField(max_length=100)
    value = models.FloatField()

# Query the database to retrieve data
data = MarketData.objects.all()

# Extract categories and values
categories = [entry.category for entry in data]
values = [entry.value for entry in data]

# Create Bokeh figure
p = figure(x_range=categories, title='Market Volatility', x_axis_label='Category', y_axis_label='Values')

# Plot the bar graph
p.vbar(x=categories, top=values, width=0.9)

# Display the graph
show(p)
market = ['rise ', 'fall']
# Function to generate dynamic data based on machine learning analysis
def generate_data():
    # Replace this with your machine learning analysis to get dynamic data
    # For demonstration, let's generate random probabilities for 10 classes
    probabilities = np.random.rand(10)
    # Normalize probabilities to sum to 1
    probabilities /= probabilities.sum()
    # Scale probabilities to be out of 100 (percentage)
    percentages = probabilities * 100
    return percentages

# Function to update the plot
def update_plot():
    # Clear the current plot
    plt.clf()
    # Generate new percentages based on machine learning analysis
    percentages = generate_data()
    # Plot the data
    sns.barplot(x=range(len(percentages)), y=percentages)
    # Annotate each bar with its percentage of occurrence
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')
    # Show the plot
    plt.show()

# Continuously update the plot
while True:
    update_plot()