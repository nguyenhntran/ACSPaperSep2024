# Script to plot Pareto fronts in sensitivity space and corresponding locations in parameter space

# Python preliminary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Load data

# SenPolyhedrons = np.load('PositivePositive_SensitivityPolyhedrons.npy')

data2 = np.load('PositivePositive_SensitivityPairsParetos.npz')
for key, value in data2.items():
    globals()[key] = value
    
# ParamPolyhedron = np.load('PositivePositive_ParamPolyhedron.npy', allow_pickle=True)
    
data4 = np.load('PositivePositive_ParamParetoArrays.npz', allow_pickle=True)
for key, value in data4.items():
    globals()[key] = value


# _____________ Plot sensitivity pairs and Pareto front of each pair _____________

# S_beta_x_xss = SenPolyhedrons[:,0]
# S_beta_x_yss = SenPolyhedrons[:,1]
# S_beta_y_xss = SenPolyhedrons[:,2]
# S_beta_y_yss = SenPolyhedrons[:,3]
# S_n_xss      = SenPolyhedrons[:,4]
# S_n_yss      = SenPolyhedrons[:,5]

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_xss,               S_beta_x_yss,   alpha= 0.5 , s=10)
axes[0].scatter(   paretoset_SensPair1[:,0],   paretoset_SensPair1[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_x}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto1_params[:,0],    pareto1_params[:,1],    pareto1_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 1')

plt.tight_layout()
fig.savefig("fig1_S_beta_x_xss_AND_S_beta_x_yss.png", dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_xss,               S_beta_y_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair2[:,0],   paretoset_SensPair2[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto2_params[:,0],    pareto2_params[:,1],    pareto2_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 2')

plt.tight_layout()
fig.savefig("fig3_S_beta_x_xss_AND_S_beta_y_xss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_xss,               S_beta_y_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair3[:,0],   paretoset_SensPair3[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto3_params[:,0],    pareto3_params[:,1],    pareto3_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 3')

plt.tight_layout()
fig.savefig("fig3_S_beta_x_xss_AND_S_beta_y_yss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_xss,                    S_n_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair4[:,0],   paretoset_SensPair4[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto4_params[:,0],    pareto4_params[:,1],    pareto4_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 4')

plt.tight_layout()
fig.savefig("fig4_S_beta_x_xss_AND_S_n_xss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_xss,                    S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair5[:,0],   paretoset_SensPair5[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto5_params[:,0],    pareto5_params[:,1],    pareto5_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 5')

plt.tight_layout()
fig.savefig("fig5_S_beta_x_xss_AND_S_n_yss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_yss,               S_beta_y_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair6[:,0],   paretoset_SensPair6[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto6_params[:,0],    pareto6_params[:,1],    pareto6_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 6')

plt.tight_layout()
fig.savefig("fig6_S_beta_x_yss_AND_S_beta_y_xss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_yss,               S_beta_y_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair7[:,0],   paretoset_SensPair7[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto7_params[:,0],    pareto7_params[:,1],    pareto7_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 7')

plt.tight_layout()
fig.savefig("fig7_S_beta_x_yss_AND_S_beta_y_yss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_yss,                    S_n_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair8[:,0],   paretoset_SensPair8[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto8_params[:,0],    pareto8_params[:,1],    pareto8_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 8')

plt.tight_layout()
fig.savefig("fig8_S_beta_x_yss_AND_S_n_xss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_x_yss,                    S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair9[:,0],   paretoset_SensPair9[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto9_params[:,0],    pareto9_params[:,1],    pareto9_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 9')

plt.tight_layout()
fig.savefig("fig9_S_beta_x_yss_AND_S_n_yss.png", dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(               S_beta_y_xss,                 S_beta_y_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair10[:,0],   paretoset_SensPair10[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto10_params[:,0],    pareto10_params[:,1],    pareto10_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 10')

plt.tight_layout()
fig.savefig("fig10_S_beta_y_xss_AND_S_beta_y_yss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(                S_beta_y_xss,                     S_n_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair11[:,0],   paretoset_SensPair11[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(   pareto11_params[:,0],   pareto11_params[:,1],   pareto11_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 11')

plt.tight_layout()
fig.savefig("fig11_S_beta_y_xss_AND_S_n_xss.png", dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(                S_beta_y_xss,                     S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair12[:,0],   paretoset_SensPair12[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(   pareto12_params[:,0],   pareto12_params[:,1],   pareto12_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 12')

plt.tight_layout()
fig.savefig("fig12_S_beta_y_xss_AND_S_n_yss.png", dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(                S_beta_y_yss,                     S_n_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair13[:,0],   paretoset_SensPair13[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto13_params[:,0],    pareto13_params[:,1],    pareto13_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 13')

plt.tight_layout()
fig.savefig("fig13_S_beta_y_yss_AND_S_n_xss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(                S_beta_y_yss,                     S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair14[:,0],   paretoset_SensPair14[:,1],   s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(    pareto14_params[:,0],    pareto14_params[:,1],    pareto14_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 14')

plt.tight_layout()
fig.savefig("fig14_S_beta_y_yss_AND_S_n_yss.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# axes[0].scatter(                     S_n_xss,                     S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair15[:,0],   paretoset_SensPair15[:,1],   s=10)
axes[0].set_xlabel(r'$S_{n}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(   pareto15_params[:,0],   pareto15_params[:,1],   pareto15_params[:,2],   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

fig.suptitle('Double-positive feedback: Pareto front and paramters 15')

plt.tight_layout()
fig.savefig("fig15_S_n_xss_AND_S_n_yss.png", dpi=300, bbox_inches='tight')
plt.show()

