#views.pys
from django.shortcuts import render
from .model import MarketData
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import components

def market_data_visualization(request):
    # Query the database to retrieve data
    data = MarketData.objects.all()

    # Extract categories and values
    categories = [entry.category for entry in data]
    values = [entry.value for entry in data]

    # Create Bokeh figure
    p = figure(x_range=categories, title='Market Volatility', x_axis_label='Category', y_axis_label='Values')

    # Plot the bar graph
    p.vbar(x=categories, top=values, width=0.9)

    # Generate Bokeh script and div
    script, div = components(p, CDN)

    return render(request, 'market_data_visualization.html', {'script': script, 'div': div})
