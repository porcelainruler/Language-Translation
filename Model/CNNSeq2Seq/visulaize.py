import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_file

import warnings

# For ignoring warnings
warnings.filterwarnings("ignore")

# Data Visualization Info
epochs = []
trainlosses = []
vallosses = []


# For Plotting Train and Valid Loss Plot
def plot(epochs: list, trainlosses: list, vallosses: list):
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


# For Attention Visualization
def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(0).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()