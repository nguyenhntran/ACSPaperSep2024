# For first steady state, plots Pareto fronts for each sensitivity pair and their points in parameter space
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Load data

data0 = np.load('Toggle_Paretos_Sens1_final.npz', allow_pickle=True)
for key, value in data0.items():
    globals()[key] = value

#data1 = np.load('Toggle_Paretos_Sens2_final.npz', allow_pickle=True)
#for key, value in data1.items():
#    globals()[key] = value
    
data2 = np.load('Toggle_Paretos_Param1_final.npz', allow_pickle=True)
for key, value in data2.items():
    globals()[key] = value

#data3 = np.load('Toggle_Paretos_Param2_final.npz', allow_pickle=True)
#for key, value in data3.items():
#    globals()[key] = value





fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair1_final[:,0],   paretoset_Sens1Pair1_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_x}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask1_Param_final[:,0],    paretoset_Sens1mask1_Param_final[:,1],    paretoset_Sens1mask1_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

#-------------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair2_final[:,0],   paretoset_Sens1Pair2_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask2_Param_final[:,0],    paretoset_Sens1mask2_Param_final[:,1],    paretoset_Sens1mask2_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair3_final[:,0],   paretoset_Sens1Pair3_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask3_Param_final[:,0],    paretoset_Sens1mask3_Param_final[:,1],    paretoset_Sens1mask3_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair4_final[:,0],   paretoset_Sens1Pair4_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask4_Param_final[:,0],    paretoset_Sens1mask4_Param_final[:,1],    paretoset_Sens1mask4_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair5_final[:,0],   paretoset_Sens1Pair5_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask5_Param_final[:,0],    paretoset_Sens1mask5_Param_final[:,1],    paretoset_Sens1mask5_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair6_final[:,0],   paretoset_Sens1Pair6_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask6_Param_final[:,0],    paretoset_Sens1mask6_Param_final[:,1],    paretoset_Sens1mask6_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair7_final[:,0],   paretoset_Sens1Pair7_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask7_Param_final[:,0],    paretoset_Sens1mask7_Param_final[:,1],    paretoset_Sens1mask7_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair8_final[:,0],   paretoset_Sens1Pair8_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask8_Param_final[:,0],    paretoset_Sens1mask8_Param_final[:,1],    paretoset_Sens1mask8_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair9_final[:,0],   paretoset_Sens1Pair9_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask9_Param_final[:,0],    paretoset_Sens1mask9_Param_final[:,1],    paretoset_Sens1mask9_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair10_final[:,0],   paretoset_Sens1Pair10_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask10_Param_final[:,0],    paretoset_Sens1mask10_Param_final[:,1],    paretoset_Sens1mask10_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair11_final[:,0],   paretoset_Sens1Pair11_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask11_Param_final[:,0],    paretoset_Sens1mask11_Param_final[:,1],    paretoset_Sens1mask11_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair12_final[:,0],   paretoset_Sens1Pair12_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask12_Param_final[:,0],    paretoset_Sens1mask12_Param_final[:,1],    paretoset_Sens1mask12_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# -----

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair13_final[:,0],   paretoset_Sens1Pair13_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask13_Param_final[:,0],    paretoset_Sens1mask13_Param_final[:,1],    paretoset_Sens1mask13_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair14_final[:,0],   paretoset_Sens1Pair14_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask14_Param_final[:,0],    paretoset_Sens1mask14_Param_final[:,1],    paretoset_Sens1mask14_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

# --------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].scatter(   paretoset_Sens1Pair15_final[:,0],   paretoset_Sens1Pair15_final[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{n}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    paretoset_Sens1mask15_Param_final[:,0],    paretoset_Sens1mask15_Param_final[:,1],    paretoset_Sens1mask15_Param_final[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()

