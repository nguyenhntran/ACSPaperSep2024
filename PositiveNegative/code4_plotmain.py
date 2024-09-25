# Script to plot main figure

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# Load data
data2 = np.load('PositiveNegative_SensitivityPairsParetos.npz')
for key, value in data2.items():
    globals()[key] = value

# Create a 5x3 grid for subplots
fig, axs = plt.subplots(2, 2, figsize=(6, 6))  # Adjust figsize as needed for clarity
fig.tight_layout(pad=2.5)  # Adjust padding between subplots

# List of pareto sets and labels for each plot, now with modulus (absolute value) signs around labels
pareto_sets = [
    paretoset_SensPair1, paretoset_SensPair2,  
    paretoset_SensPair4, paretoset_SensPair5
]

labels = [
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{\beta_x}(y_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{\beta_y}(x_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{n}(y_{ss})|$')
]

# Plot each pair in the respective subplot
for i, ax in enumerate(axs.flat):
    ax.scatter(pareto_sets[i][:, 0], pareto_sets[i][:, 1], s=5)
    ax.set_xlabel(labels[i][0])
    ax.set_ylabel(labels[i][1])

    # Set axis formatting to scientific if values need more than 3 decimal points
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((-2, 2))
    
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))


# Save the complete figure with all subplots
plt.savefig("posneg_main.pdf", dpi=600, bbox_inches='tight')
plt.show()
