
#  ---------------------------------------------------------------------
# |                                                                     |
# |          PART 3: PLOTTING ZOOMED UNIQUE SENSITIVITY PAIRS           |
# |                     AND PARETO IN PARAMETER SPACE                   |
# |                                                                     |
#  ---------------------------------------------------------------------

# Python preliminary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Load data

#data1 = np.load('PositiveNegative_SensitivityPolyhedrons.npz')
#for key, value in data1.items():
#    globals()[key] = value

data2 = np.load('PositiveNegative_SensitivityPairsParetos.npz')
for key, value in data2.items():
    globals()[key] = value
    
#data3 = np.load('PositiveNegative_ParamPolyhedron.npz', allow_pickle=True)
#for key, value in data3.items():
#    globals()[key] = value
    
data4 = np.load('PositiveNegative_ParamParetoArrays.npz', allow_pickle=True)
for key, value in data4.items():
    globals()[key] = value


# _____________ Plot sensitivity pairs and Pareto front of each pair _____________

#S_beta_x_xss = SenPolyhedrons[:,0]
#S_beta_x_yss = SenPolyhedrons[:,1]
#S_beta_y_xss = SenPolyhedrons[:,2]
#S_beta_y_yss = SenPolyhedrons[:,3]
#S_n_xss      = SenPolyhedrons[:,4]
#S_n_yss      = SenPolyhedrons[:,5]

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(               S_beta_x_xss,               S_beta_x_yss,   alpha= 0.5 , s=10)
axes[0].scatter(   paretoset_SensPair1[:,0],   paretoset_SensPair1[:,1],   color='red', s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_x}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(    pareto1_params[:,0],    pareto1_params[:,1],    pareto1_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel(r'$\beta_x$')
ax_3d.set_ylabel(r'$\beta_y$')
ax_3d.set_zlabel(r'$n$')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(               S_beta_x_xss,               S_beta_y_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair2[:,0],   paretoset_SensPair2[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(    pareto2_params[:,0],    pareto2_params[:,1],    pareto2_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(               S_beta_x_xss,                    S_n_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair4[:,0],   paretoset_SensPair4[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(    pareto4_params[:,0],    pareto4_params[:,1],    pareto4_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(               S_beta_x_xss,                    S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair5[:,0],   paretoset_SensPair5[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(    pareto5_params[:,0],    pareto5_params[:,1],    pareto5_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(               S_beta_x_yss,               S_beta_y_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair6[:,0],   paretoset_SensPair6[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{\beta_y}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(    pareto6_params[:,0],    pareto6_params[:,1],    pareto6_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(               S_beta_x_yss,                    S_n_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair8[:,0],   paretoset_SensPair8[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(    pareto8_params[:,0],    pareto8_params[:,1],    pareto8_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(               S_beta_x_yss,                    S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair9[:,0],   paretoset_SensPair9[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_x}(y_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(    pareto9_params[:,0],    pareto9_params[:,1],    pareto9_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(                S_beta_y_xss,                     S_n_xss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair11[:,0],   paretoset_SensPair11[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(x_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(   pareto11_params[:,0],   pareto11_params[:,1],   pareto11_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(                S_beta_y_xss,                     S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair12[:,0],   paretoset_SensPair12[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{\beta_y}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(   pareto12_params[:,0],   pareto12_params[:,1],   pareto12_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#axes[0].scatter(                     S_n_xss,                     S_n_yss,   alpha= 0.5 ,   s=10)
axes[0].scatter(   paretoset_SensPair15[:,0],   paretoset_SensPair15[:,1],   color='red',   s=10)
axes[0].set_xlabel(r'$S_{n}(x_{ss})$')
axes[0].set_ylabel(r'$S_{n}(y_{ss})$')

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#ax_3d.scatter(   ParamPolyhedron[:,0],   ParamPolyhedron[:,1],   ParamPolyhedron[:,2],   alpha= 0.5 ,   s=10)
ax_3d.scatter(   pareto15_params[:,0],   pareto15_params[:,1],   pareto15_params[:,2],   color='red',   s=10)
ax_3d.set_xlabel('beta_x')
ax_3d.set_ylabel('beta_y')
ax_3d.set_zlabel('n')
ax_3d.grid(True)

plt.tight_layout()
plt.show()
