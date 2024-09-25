# Plots supplementary figure associated with steady state 1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# Load data
data = np.load('Toggle_Paretos_Sens1_final.npz')
for key, value in data.items():
    globals()[key] = value

# Create a 5x3 grid for subplots
fig, axs = plt.subplots(4, 3, figsize=(8,10))  # Adjust figsize as needed for clarity
fig.suptitle('Double negative: all sensitivity combinations of equilibrium 1')
fig.tight_layout(pad=2)  # Adjust padding between subplots

# List of pareto sets and labels for each plot, now with modulus (absolute value) signs around labels
pareto_sets = [
    paretoset_Sens1Pair1_final, paretoset_Sens1Pair2_final, paretoset_Sens1Pair4_final, 
    paretoset_Sens1Pair5_final, paretoset_Sens1Pair6_final, paretoset_Sens1Pair8_final, 
    paretoset_Sens1Pair9_final, paretoset_Sens1Pair11_final, paretoset_Sens1Pair12_final, 
    paretoset_Sens1Pair15_final
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
    if i == 7-1:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
    if i == 8-1:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)


# Save the complete figure with all subplots
plt.savefig("toggle_ss1_supp.pdf", dpi=600, bbox_inches='tight')
plt.show()
