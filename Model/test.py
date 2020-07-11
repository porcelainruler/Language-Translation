import os
import warnings

# For ignoring warnings
warnings.filterwarnings("ignore")

# Plotting Libraries
# Bokeh Imports
# from bokeh.io import curdoc
# from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_file
# from functools import partial
# from tornado import gen

# Data Visualization Info
epochs = []
trainlosses = []
vallosses = []


# For Plotting Train and Valid Loss Plot
def plot():
    output_file("GEC_Visualize.html")

    source = ColumnDataSource(data={'epochs': epochs, 'trainlosses': trainlosses, 'vallosses': vallosses})

    plot = figure()
    plot.line(x='epochs', y='trainlosses',
              color='green', alpha=0.8, legend='Train loss', line_width=2,
              source=source)
    plot.line(x='epochs', y='vallosses',
              color='red', alpha=0.8, legend='Val loss', line_width=2,
              source=source)

    show(plot)