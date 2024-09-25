# Plots equilibria coordinates of the two steady states both pre- and post- bifurcation

import numpy as np
import matplotlib.pyplot as plt

# Load data

data0 = np.load('Toggle_SteadyStates_final.npz', allow_pickle=True)
for key, value in data0.items():
    globals()[key] = value

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   xss1_final,   yss1_final,   s=10)
axes[0].set_xlabel(r'$x_{ss}$')
axes[0].set_ylabel(r'$y_{ss}$')
axes[0].set_title('Steady state 1')

axes[1].scatter(   xss2_final,   yss2_final,   s=10)
axes[1].set_xlabel(r'$x_{ss}$')
axes[1].set_ylabel(r'$y_{ss}$')
axes[1].set_title('Steady state 2')

plt.tight_layout()
plt.show()
