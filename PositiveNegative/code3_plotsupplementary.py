# Script to plot supplementary figure

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Load data
data = np.load('PositiveNegative_SensitivityPairsParetos.npz')
for key, value in data.items():
    globals()[key] = value

# Create a 5x3 grid for subplots
fig, axs = plt.subplots(4, 3, figsize=(8,10))  # Adjust figsize as needed for clarity
fig.suptitle('Positive-negative: all sensitivity combinations')
fig.tight_layout(pad=2)  # Adjust padding between subplots

# List of pareto sets and labels for each plot, now with modulus (absolute value) signs around labels
pareto_sets = [
    paretoset_SensPair1, paretoset_SensPair2, paretoset_SensPair4, 
    paretoset_SensPair5, paretoset_SensPair6, paretoset_SensPair8, 
    paretoset_SensPair9, paretoset_SensPair11, paretoset_SensPair12, 
    paretoset_SensPair15
]

labels = [
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{\beta_x}(y_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{\beta_y}(x_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{n}(y_{ss})|$'),
    (r'$|S_{\beta_x}(y_{ss})|$', r'$|S_{\beta_y}(x_{ss})|$'),
    (r'$|S_{\beta_x}(y_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_x}(y_{ss})|$', r'$|S_{n}(y_{ss})|$'),
    (r'$|S_{\beta_y}(x_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_y}(x_{ss})|$', r'$|S_{n}(y_{ss})|$'),
    (r'$|S_{n}(x_{ss})|$', r'$|S_{n}(y_{ss})|$')
]

# Plot each pair in the respective subplot
for i, ax in enumerate(axs.flat[2:]):
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

    # Adjust xlim and ylim for specific subplots
    if i == 7-1:  # Subplots 9-12 correspond to index 6 and above in `pareto_sets`
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)


# Save the complete figure with all subplots
plt.savefig("posneg_supp.pdf", dpi=600, bbox_inches='tight')
plt.show()
