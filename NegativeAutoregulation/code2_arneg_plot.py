
#  ---------------------------------------------------------------------
# |                                                                     |
# |                   PLOTTING UNIQUE SENSITIVITY PAIRS                 |
# |                     AND PARETO IN PARAMETER SPACE                   |
# |                                                                     |
#  ---------------------------------------------------------------------

# Python preliminary
import numpy as np
import matplotlib.pyplot as plt

# Load data
SenPolyhedrons = np.load('ARneg_SensitivityPolyhedrons.npy')
pareto_Sens = np.load('ARneg_SensitivityPareto.npy')
ParamPolyhedron = np.load('ARneg_ParamPolyhedron.npy', allow_pickle=True)
pareto_Params = np.load('ARneg_ParamPareto.npy', allow_pickle=True)


# _____________ Plot sensitivity pairs and Pareto front of each pair _____________

S_a_xss = SenPolyhedrons[:,0]
S_n_xss = SenPolyhedrons[:,1]

# ---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(8,4))
'''
#axes[0].scatter(            S_n_xss,            S_a_xss,   alpha= 0.5 , s=10)
axes[0].scatter(   pareto_Sens[:,1],   pareto_Sens[:,0],   color='red', s=10)
axes[0].set_xlabel(r'$S_{n}(x_{ss})$')
axes[0].set_ylabel(r'$S_{a}(x_{ss})$')

axes[1].scatter(   pareto_Params[:,1],   pareto_Params[:,0],   color='red', s=10)
axes[1].set_xlabel(r'$n$')
axes[1].set_ylabel(r'$a$')

plt.tight_layout()
plt.show()

'''
plt.figure(figsize=(3, 3))
plt.scatter(   pareto_Sens[:,0],   pareto_Sens[:,1], s=10)
plt.xlabel(r'$|S_{n}(x_{ss})|$',fontsize=10)
plt.ylabel(r'$|S_{a}(x_{ss})|$',fontsize=10)
plt.title('Negative autoregulation: Pareto front',fontsize=10)
plt.tight_layout()
plt.savefig("arneg_main.pdf", dpi=600, bbox_inches='tight')
plt.show()

