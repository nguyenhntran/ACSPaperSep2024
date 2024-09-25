# Script to plot supplementary figure

import numpy as np
import matplotlib.pyplot as plt

# Load data
data2 = np.load('PositivePositive_SensitivityPairsParetos.npz')
for key, value in data2.items():
    globals()[key] = value

# Create a 3x5 grid for subplots
fig, axs = plt.subplots(4, 3, figsize=(8,10))  # Adjust figsize as needed for clarity
fig.suptitle('Double positive: all sensitivity combinations')
fig.tight_layout(pad=2.5)  # Adjust padding between subplots

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
    ax.scatter(pareto_sets[i][:, 0], pareto_sets[i][:, 1], s=10)
    ax.set_xlabel(labels[i][0])
    ax.set_ylabel(labels[i][1])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

# Save the complete figure with all subplots
plt.savefig("pospos_supp.pdf", dpi=600, bbox_inches='tight')
plt.show()