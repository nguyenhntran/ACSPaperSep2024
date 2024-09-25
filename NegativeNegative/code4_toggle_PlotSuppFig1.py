# Plots supplementary figure for steady state 1

import numpy as np
import matplotlib.pyplot as plt

# Load data
data2 = np.load('Toggle_Paretos_Sens1_final.npz')
for key, value in data2.items():
    globals()[key] = value

# Create a 5x3 grid for subplots
fig, axs = plt.subplots(5, 3, figsize=(10, 16))  # Adjust figsize as needed for clarity
fig.suptitle('Toggle switch: first steady state sensitivities')
fig.tight_layout(pad=2.5)  # Adjust padding between subplots

# List of pareto sets and labels for each plot, now with modulus (absolute value) signs around labels
pareto_sets = [
    paretoset_Sens1Pair1_final, paretoset_Sens1Pair2_final, paretoset_Sens1Pair3_final, 
    paretoset_Sens1Pair4_final, paretoset_Sens1Pair5_final, paretoset_Sens1Pair6_final, 
    paretoset_Sens1Pair7_final, paretoset_Sens1Pair8_final, paretoset_Sens1Pair9_final, 
    paretoset_Sens1Pair10_final, paretoset_Sens1Pair11_final, paretoset_Sens1Pair12_final, 
    paretoset_Sens1Pair13_final, paretoset_Sens1Pair14_final, paretoset_Sens1Pair15_final
]

labels = [
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{\beta_x}(y_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{\beta_y}(x_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{\beta_y}(y_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_x}(x_{ss})|$', r'$|S_{n}(y_{ss})|$'),
    (r'$|S_{\beta_x}(y_{ss})|$', r'$|S_{\beta_y}(x_{ss})|$'),
    (r'$|S_{\beta_x}(y_{ss})|$', r'$|S_{\beta_y}(y_{ss})|$'),
    (r'$|S_{\beta_x}(y_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_x}(y_{ss})|$', r'$|S_{n}(y_{ss})|$'),
    (r'$|S_{\beta_y}(x_{ss})|$', r'$|S_{\beta_y}(y_{ss})|$'),
    (r'$|S_{\beta_y}(x_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_y}(x_{ss})|$', r'$|S_{n}(y_{ss})|$'),
    (r'$|S_{\beta_y}(y_{ss})|$', r'$|S_{n}(x_{ss})|$'),
    (r'$|S_{\beta_y}(y_{ss})|$', r'$|S_{n}(y_{ss})|$'),
    (r'$|S_{n}(x_{ss})|$', r'$|S_{n}(y_{ss})|$')
]

# Plot each pair in the respective subplot
for i, ax in enumerate(axs.flat):
    ax.scatter(pareto_sets[i][:, 0], pareto_sets[i][:, 1], s=10)
    ax.set_xlabel(labels[i][0])
    ax.set_ylabel(labels[i][1])

# Save the complete figure with all subplots
plt.savefig("Toggle_ss1_supp.pdf", dpi=300, bbox_inches='tight')
plt.show()
