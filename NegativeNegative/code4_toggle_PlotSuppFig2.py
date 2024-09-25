# Plots supplementary figure for steady state 2

import numpy as np
import matplotlib.pyplot as plt

# Load data
data2 = np.load('Toggle_Paretos_Sens2_final.npz')
for key, value in data2.items():
    globals()[key] = value

# Create a 5x3 grid for subplots
fig, axs = plt.subplots(5, 3, figsize=(10, 16))  # Adjust figsize as needed for clarity
fig.suptitle('Toggle switch: second steady state sensitivities')
fig.tight_layout(pad=2.5)  # Adjust padding between subplots

# List of pareto sets and labels for each plot, now with modulus (absolute value) signs around labels
pareto_sets = [
    paretoset_Sens2Pair1_final, paretoset_Sens2Pair2_final, paretoset_Sens2Pair3_final, 
    paretoset_Sens2Pair4_final, paretoset_Sens2Pair5_final, paretoset_Sens2Pair6_final, 
    paretoset_Sens2Pair7_final, paretoset_Sens2Pair8_final, paretoset_Sens2Pair9_final, 
    paretoset_Sens2Pair10_final, paretoset_Sens2Pair11_final, paretoset_Sens2Pair12_final, 
    paretoset_Sens2Pair13_final, paretoset_Sens2Pair14_final, paretoset_Sens2Pair15_final
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
plt.savefig("Toggle_ss2_supp.pdf", bbox_inches='tight')
plt.show()
