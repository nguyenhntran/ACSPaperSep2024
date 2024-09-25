# For second steady state, plots Pareto fronts for each sensitivity pair and their points in parameter space

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Load data


data1 = np.load('Toggle_Paretos_Sens2_final.npz', allow_pickle=True)
for key, value in data1.items():
    globals()[key] = value

data2 = np.load('Toggle_Paretos_Param2_final.npz', allow_pickle=True)
for key, value in data2.items():
    globals()[key] = value





fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair1_final[:,0],   paretoset_Sens2Pair1_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask1_Param_final[:,0],    paretoset_Sens2mask1_Param_final[:,1],    paretoset_Sens2mask1_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

#-------------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair2_final[:,0],   paretoset_Sens2Pair2_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask2_Param_final[:,0],    paretoset_Sens2mask2_Param_final[:,1],    paretoset_Sens2mask2_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair3_final[:,0],   paretoset_Sens2Pair3_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask3_Param_final[:,0],    paretoset_Sens2mask3_Param_final[:,1],    paretoset_Sens2mask3_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair4_final[:,0],   paretoset_Sens2Pair4_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask4_Param_final[:,0],    paretoset_Sens2mask4_Param_final[:,1],    paretoset_Sens2mask4_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair5_final[:,0],   paretoset_Sens2Pair5_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask5_Param_final[:,0],    paretoset_Sens2mask5_Param_final[:,1],    paretoset_Sens2mask5_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair6_final[:,0],   paretoset_Sens2Pair6_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask6_Param_final[:,0],    paretoset_Sens2mask6_Param_final[:,1],    paretoset_Sens2mask6_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair7_final[:,0],   paretoset_Sens2Pair7_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask7_Param_final[:,0],    paretoset_Sens2mask7_Param_final[:,1],    paretoset_Sens2mask7_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair8_final[:,0],   paretoset_Sens2Pair8_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask8_Param_final[:,0],    paretoset_Sens2mask8_Param_final[:,1],    paretoset_Sens2mask8_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair9_final[:,0],   paretoset_Sens2Pair9_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask9_Param_final[:,0],    paretoset_Sens2mask9_Param_final[:,1],    paretoset_Sens2mask9_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair10_final[:,0],   paretoset_Sens2Pair10_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask10_Param_final[:,0],    paretoset_Sens2mask10_Param_final[:,1],    paretoset_Sens2mask10_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair11_final[:,0],   paretoset_Sens2Pair11_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask11_Param_final[:,0],    paretoset_Sens2mask11_Param_final[:,1],    paretoset_Sens2mask11_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair12_final[:,0],   paretoset_Sens2Pair12_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask12_Param_final[:,0],    paretoset_Sens2mask12_Param_final[:,1],    paretoset_Sens2mask12_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair13_final[:,0],   paretoset_Sens2Pair13_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask13_Param_final[:,0],    paretoset_Sens2mask13_Param_final[:,1],    paretoset_Sens2mask13_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair14_final[:,0],   paretoset_Sens2Pair14_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask14_Param_final[:,0],    paretoset_Sens2mask14_Param_final[:,1],    paretoset_Sens2mask14_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens2Pair15_final[:,0],   paretoset_Sens2Pair15_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{n}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')
axes[0].set_yscale('log')
axes[0].set_xscale('log')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens2mask15_Param_final[:,0],    paretoset_Sens2mask15_Param_final[:,1],    paretoset_Sens2mask15_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()
