
# Plotting Pareto fronts in sensitivity space only

# Python preliminary

import numpy as np
import matplotlib.pyplot as plt

# Load data

# SenPolyhedrons = np.load('PositivePositive_SensitivityPolyhedrons.npy')

data2 = np.load('PositivePositive_SensitivityPairsParetos.npz')
for key, value in data2.items():
    globals()[key] = value
    
# ParamPolyhedron = np.load('PositivePositive_ParamPolyhedron.npy', allow_pickle=True)
    
# data4 = np.load('PositivePositive_ParamParetoArrays.npz', allow_pickle=True)
# for key, value in data4.items():
#     globals()[key] = value


# _____________ Plot sensitivity pairs and Pareto front of each pair _____________

# S_beta_x_xss = SenPolyhedrons[:,0]
# S_beta_x_yss = SenPolyhedrons[:,1]
# S_beta_y_xss = SenPolyhedrons[:,2]
# S_beta_y_yss = SenPolyhedrons[:,3]
# S_n_xss      = SenPolyhedrons[:,4]
# S_n_yss      = SenPolyhedrons[:,5]

'''
# ---------------------------------------------------------------

plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_xss,               S_beta_x_yss,   alpha= 0.5 ,   s=10)
plt.scatter(paretoset_SensPair1[:, 0], paretoset_SensPair1[:, 1], s=10)
plt.xlabel(r'$S_{\beta_x}(x_{ss})$')  # X-axis label
plt.ylabel(r'$S_{\beta_x}(y_{ss})$')  # Y-axis label
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig1_S_beta_x_xss_AND_S_beta_x_yss.png", dpi=600, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_xss,               S_beta_y_xss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair2[:,0],   paretoset_SensPair2[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(x_{ss})$')
plt.ylabel(r'$S_{\beta_y}(x_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig2_S_beta_x_xss_AND_S_beta_y_xss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_xss,               S_beta_y_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair3[:,0],   paretoset_SensPair3[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(x_{ss})$')
plt.ylabel(r'$S_{\beta_y}(y_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig3_S_beta_x_xss_AND_S_beta_y_yss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------

plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_xss,                    S_n_xss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair4[:,0],   paretoset_SensPair4[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(x_{ss})$')
plt.ylabel(r'$S_{n}(x_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig4_S_beta_x_xss_AND_S_n_xss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_xss,                    S_n_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair5[:,0],   paretoset_SensPair5[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(x_{ss})$')
plt.ylabel(r'$S_{n}(y_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig5_S_beta_x_xss_AND_S_n_yss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_yss,               S_beta_y_xss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair6[:,0],   paretoset_SensPair6[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(y_{ss})$')
plt.ylabel(r'$S_{\beta_y}(x_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig6_S_beta_x_yss_AND_S_beta_y_xss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_yss,               S_beta_y_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair7[:,0],   paretoset_SensPair7[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(y_{ss})$')
plt.ylabel(r'$S_{\beta_y}(y_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig7_S_beta_x_yss_AND_S_beta_y_yss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_yss,                    S_n_xss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair8[:,0],   paretoset_SensPair8[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(y_{ss})$')
plt.ylabel(r'$S_{n}(x_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig8_S_beta_x_yss_AND_S_n_xss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_x_yss,                    S_n_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair9[:,0],   paretoset_SensPair9[:,1],   s=10)
plt.xlabel(r'$S_{\beta_x}(y_{ss})$')
plt.ylabel(r'$S_{n}(y_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig9_S_beta_x_yss_AND_S_n_yss.png", dpi=600, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(               S_beta_y_xss,                 S_beta_y_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair10[:,0],   paretoset_SensPair10[:,1],   s=10)
plt.xlabel(r'$S_{\beta_y}(x_{ss})$')
plt.ylabel(r'$S_{\beta_y}(y_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig10_S_beta_y_xss_AND_S_beta_y_yss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(                S_beta_y_xss,                     S_n_xss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair11[:,0],   paretoset_SensPair11[:,1],   s=10)
plt.xlabel(r'$S_{\beta_y}(x_{ss})$')
plt.ylabel(r'$S_{n}(x_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig11_S_beta_y_xss_AND_S_n_xss.png", dpi=600, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(                S_beta_y_xss,                     S_n_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair12[:,0],   paretoset_SensPair12[:,1],   s=10)
plt.xlabel(r'$S_{\beta_y}(x_{ss})$')
plt.ylabel(r'$S_{n}(y_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig12_S_beta_y_xss_AND_S_n_yss.png", dpi=600, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(                S_beta_y_yss,                     S_n_xss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair13[:,0],   paretoset_SensPair13[:,1],   s=10)
plt.xlabel(r'$S_{\beta_y}(y_{ss})$')
plt.ylabel(r'$S_{n}(x_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig13_S_beta_y_yss_AND_S_n_xss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------


plt.figure(figsize=(3, 3))

# plt.scatter(                S_beta_y_yss,                     S_n_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair14[:,0],   paretoset_SensPair14[:,1],   s=10)
plt.xlabel(r'$S_{\beta_y}(y_{ss})$')
plt.ylabel(r'$S_{n}(y_{ss})$')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig14_S_beta_y_yss_AND_S_n_yss.png", dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------
'''

plt.figure(figsize=(3, 3))

# plt.scatter(                     S_n_xss,                     S_n_yss,   alpha= 0.5 ,   s=10)
plt.scatter(   paretoset_SensPair15[:,0],   paretoset_SensPair15[:,1],   s=10)
plt.xlabel(r'$|S_{n}(x_{ss})|$')
plt.ylabel(r'$|S_{n}(y_{ss})|$')
plt.title('Double positive:\nPareto front of cooperativity sensitivities')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.tight_layout()
plt.savefig("fig15_S_n_xss_AND_S_n_yss.pdf", dpi=600, bbox_inches='tight')
plt.show()

