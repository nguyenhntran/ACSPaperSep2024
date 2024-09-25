#  ---------------------------------------------------------------------
# |                                                                     |
# |                   PART 1: CREATING VALUES TO PLOT                   |
# |                                                                     |
#  ---------------------------------------------------------------------


# _________ Python preliminary _________
                                       #|
import numpy as np                     #|
from tqdm import tqdm                  #|
from paretoset import paretoset        #|
from scipy.optimize import fsolve      #|
from joblib import Parallel, delayed   #|
#______________________________________#|


# ____________________________ DEFINE DYNAMICAL SYSTEM _______________________________
#                               (CAN EDIT THIS CELL)                                 #|
                                                                                     #|
# x-nullcline                                                                        #|
def Equ1(x, y, beta_x, n):                                                           #|
    return beta_x * y**n / (1 + y**n) - x                                            #|
                                                                                     #|
# y-nullcline                                                                        #|
def Equ2(x, y, beta_y, n):                                                           #|
    return beta_y * x**n / (1 + x**n) - y                                            #|
                                                                                     #|
# ____________________________________________________________________________________


# ________________ FUNCTION THAT EVALUATES VECTOR FIELD AT A POINT ___________________
                                                                                     #|
# Function that takes in coordinate   P = [x value, y value],                        #|
#                        initial time t = time value,                                #|
#                        parameters   params = [param 1 value, param 2 value, ...]   #|
# and returns corresponding value of dx/dt and dy/dt in array of form [dx/dt, dy/dt] #|
def Equs(P, t, params):                                                              #|
                                                                                     #|
    x = P[0]                                                                         #|
    y = P[1]                                                                         #|
                                                                                     #|
    beta_x = params[0]                                                               #|
    beta_y = params[1]                                                               #|
    n      = params[2]                                                               #|
                                                                                     #|
    val0 = Equ1(x, y, beta_x, n)                                                     #|
    val1 = Equ2(x, y, beta_y, n)                                                     #|
                                                                                     #|
    return np.array([val0, val1])                                                    #|
                                                                                     #|
# ____________________________________________________________________________________


# _______________ DEFINE GRID-SEARCHED POLYHEDRON IN PARAMETER SPACE _______________
#                               CAN EDIT THIS CELL                                 #|
                                                                                   #|
# Parameter range                                                                  #|
                                                                                   #|
beta_x_min  = 0.01                                                                 #|
beta_x_max  = 50                                                                   #|
beta_x_no   = 100                                                                  #|
beta_x_vals = np.linspace(beta_x_min,beta_x_max,beta_x_no)                         #|
                                                                                   #|
beta_y_min  = 0.01                                                                 #|
beta_y_max  = 50                                                                   #|
beta_y_no   = 100                                                                  #|
beta_y_vals = np.linspace(beta_y_min,beta_y_max,beta_y_no)                         #|
                                                                                   #|
n_min  = 0.01                                                                      #|
n_max  = 10                                                                        #|
n_no   = 100                                                                       #|
n_vals = np.linspace(n_min,n_max,n_no)                                             #|
                                                                                   #|
# __________________________________________________________________________________


# _____________ PARALLELISE TABULATING PARAMETER SPACE COORDINATES ______________
                                                                                #|
# Define the function to create a single row of the ParamPolyhedron             #|
def create_param_row(beta_x_val, beta_y_val, n_val, rownum):                    #|
    return (rownum, np.array([beta_x_val, beta_y_val, n_val]))                  #|
                                                                                #|
# Create the full parameter space as a list of tasks to distribute              #|
def generate_param_space():                                                     #|
    param_space = []                                                            #|
    rownum = 0                                                                  #|
    for beta_x_val in beta_x_vals:                                              #|
        for beta_y_val in beta_y_vals:                                          #|
            for n_val in n_vals:                                                #|
                param_space.append((beta_x_val, beta_y_val, n_val, rownum))     #|
                rownum += 1                                                     #|
    return param_space                                                          #|
                                                                                #|
# Parallelize the generation of ParamPolyhedron using joblib                    #|
def generate_param_polyhedron():                                                #|
    param_space = generate_param_space()                                        #|
                                                                                #|
    # Create an empty array to hold the results                                 #|
    ParamPolyhedron = np.empty((beta_x_no * beta_y_no * n_no, 3))               #|
                                                                                #|
    # Use joblib's Parallel to generate the rows of ParamPolyhedron in parallel #|
    results = Parallel(n_jobs=-1)(                                              #|
        delayed(create_param_row)(beta_x_val, beta_y_val, n_val, rownum)        #|
        for beta_x_val, beta_y_val, n_val, rownum in tqdm(param_space))         #|
                                                                                #|
    # Collect results into ParamPolyhedron                                      #|
    for rownum, param_values in results:                                        #|
        ParamPolyhedron[rownum, :] = param_values                               #|
                                                                                #|
    # Save parameter choices                                                    #|
    np.save('PositivePositive_ParamPolyhedron.npy', ParamPolyhedron)            #|
                                                                                #|
# _______________________________________________________________________________



generate_param_polyhedron()



# ________________________________________ TABLE OF X AND Y STEADY STATES ________________________________________
#                                                                                                                 #|
# We want to get the following two arrays                                                                         #|
#                  ---------                                    ---------                                         #|
#                 |   xss   |                                  |   yss   |                                        #|
#                 |    #    |                                  |    #    |                                        #|
# xssPolyhedron = |    #    |        and       yssPolyhedron = |    #    |                                        #|
#                 |    #    |                                  |    #    |                                        #|
#                 |    #    |                                  |    #    |                                        #|
#                  ---------                                    ---------                                         #|
                                                                                                                  #|
# Load data                                                                                                       #|
ParamPolyhedron = np.load('PositivePositive_ParamPolyhedron.npy', allow_pickle=True)                              #|
                                                                                                                  #|
# Get number of rows in the ParamPolyhedron                                                                       #|
rows = ParamPolyhedron.shape[0]                                                                                   #|
                                                                                                                  #|
# Create empty arrays to store x and y steady states                                                              #|
xssPolyhedron = np.empty((rows, 1))                                                                               #|
yssPolyhedron = np.empty((rows, 1))                                                                               #|
                                                                                                                  #|
# Define function to solve for steady states                                                                      #|
def solve_steady_state(rownum, ParamPolyhedron):                                                     #|
    params = ParamPolyhedron[rownum]                                                                              #|
    beta_x_val = params[0]
    beta_y_val = params[1]
    InitGuesses = [np.array([beta_x_val, beta_y_val]),
                     np.array([beta_x_val/2, beta_y_val/2]),
                     np.array([beta_x_val, 0]),
                     np.array([beta_x_val/2, 0]),
                     np.array([0, beta_y_val]),
                     np.array([0, beta_y_val/2]),
                     np.array([1,1]),
                     np.array([0.1,0.1])]
    t = 0.0                                                                                                       #|
                                                                                                                  #|
    for InitGuess in InitGuesses:                                                                                 #|
        output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)    #|
        xss, yss = output                                                                                         #|
        fvec = infodict['fvec']                                                                                   #|

        delta = 1e-8
        dEqudx = (Equs([xss+delta,yss], t, params)-Equs([xss,yss], t, params))/delta
        dEqudy = (Equs([xss,yss+delta], t, params)-Equs([xss,yss], t, params))/delta
        jac = np.transpose(np.vstack((dEqudx,dEqudy)))
        eig = np.linalg.eig(jac)[0]
        instablility = np.any(np.real(eig) >= 0)
                                                                                                                  #|
        # Check for valid solution                                                                                #|
        if xss > 0.04 and yss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:   #|
            return xss, yss, rownum                                                                               #|
                                                                                                                  #|
    return float('nan'), float('nan'), rownum                                                                     #|
                                                                                                                  #|
                                                                                                                  #|
# Parallel processing to solve steady states                                                                      #|
results = Parallel(n_jobs=-1)(                                                                                    #|
    delayed(solve_steady_state)(rownum, ParamPolyhedron)                                             #|
    for rownum in range(rows))                                                                                    #|
                                                                                                                  #|
# Process results and store them in the polyhedron arrays                                                         #|
for xss, yss, rownum in results:                                                                                  #|
    xssPolyhedron[rownum] = xss                                                                                   #|
    yssPolyhedron[rownum] = yss                                                                                   #|
                                                                                                                  #|
# Save arrays                                                                                                     #|
np.savez('PositivePositive_EquilibriumPolyhedrons.npz', xssPolyhedron=xssPolyhedron, yssPolyhedron=yssPolyhedron) #|
                                                                                                                  #|
#__________________________________________________________________________________________________________________



# ___________________________________ DEFINE SENSITIVITY FUNCTIONS ___________________________________
#                                          CAN EDIT THIS CELL                                        #|
                                                                                                     #|
# Setting up progress printing of loop                                                               #|
pbar = tqdm(total=6, desc="Defining sensitivity functions")                                          #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betax(xss)                                                      #|
def S_betax_xss_analytic(xss, yss, n, beta_x, beta_y):                                               #|
    hill = beta_y * xss**n / (1 + xss**n)
    numer = (1 + xss**n) * (1 + (hill)**n)
    denom = 1 - n**2 + (hill)**n + xss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betax(yss)                                                      #|
def S_betax_yss_analytic(xss, yss, n, beta_x, beta_y):                                               #|
    hill = beta_x * yss**n / (1 + yss**n)
    numer = n * (1 + yss**n)
    denom = 1 - n**2 + (hill)**n + yss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betay(xss)                                                      #|
def S_betay_xss_analytic(xss, yss, n, beta_x, beta_y):                                               #|
    hill = beta_y * xss**n / (1 + xss**n)
    numer = n * (1 + xss**n)
    denom = 1 - n**2 + (hill)**n + xss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betay(yss)                                                      #|
def S_betay_yss_analytic(xss, yss, n, beta_x, beta_y):                                               #|
    hill = beta_x * yss**n / (1 + yss**n)
    numer = (1 + yss**n) * (1 + (hill)**n)
    denom = 1 - n**2 + (hill)**n + yss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_n(xss)                                                          #|
def S_n_xss_analytic(xss, yss, n, beta_x, beta_y):                                                   #|
    hill = beta_y * xss**n / (1 + xss**n)
    numer = n * (n * np.log(xss) + np.log(hill) + np.log(hill) * xss**n)
    denom = 1 - n**2 + (hill)**n + xss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_n(yss)                                                          #|
def S_n_yss_analytic(xss, yss, n, beta_x, beta_y):                                                   #|
    hill = beta_x * yss**n / (1 + yss**n)
    numer = n * (n * np.log(yss) + np.log(hill) + np.log(hill) * yss**n)
    denom = 1 - n**2 + (hill)**n + yss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
# Close the progress bar                                                                             #|
pbar.close()                                                                                         #|
                                                                                                     #|
#_____________________________________________________________________________________________________



# __________________________________ OBTAIN TABLE OF SENSITIVITIES ____________________________________
#                                                                                                     #|
# We want to get the following array                                                                  #|
#  -------------------------------------------------------------------------------------------------  #|
# | S_{beta_x}(xss) | S_{beta_x}(yss) | S_{beta_y}(xss) | S_{beta_y}(yss) | S_{n}(xss) | S_{n}(yss) | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
#  -------------------------------------------------------------------------------------------------  #|
                                                                                                      #|
                                                                                                      #|
def compute_sensitivities(rownum, ParamPolyhedron, xssPolyhedron, yssPolyhedron):                     #|
    beta_x_val = ParamPolyhedron[rownum, 0]                                                           #|
    beta_y_val = ParamPolyhedron[rownum, 1]                                                           #|
    n_val = ParamPolyhedron[rownum, 2]                                                                #|
                                                                                                      #|
    xss_val = xssPolyhedron[rownum]                                                                   #|
    yss_val = yssPolyhedron[rownum]                                                                   #|
                                                                                                      #|
    return np.array([                                                                                 #|
        S_betax_xss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val),                        #|
        S_betax_yss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val),                        #|
        S_betay_xss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val),                        #|
        S_betay_yss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val),                        #|
        S_n_xss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val),                            #|
        S_n_yss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val)])                           #|
                                                                                                      #|
                                                                                                      #|
# Parallel processing for sensitivity calculations                                                    #|
sensitivity_results = Parallel(n_jobs=-1)(                                                            #|
    delayed(compute_sensitivities)(rownum, ParamPolyhedron, xssPolyhedron, yssPolyhedron)             #|
    for rownum in range(rows))                                                                        #|
                                                                                                      #|
# Collect results                                                                                     #|
SenPolyhedrons = np.array(sensitivity_results).squeeze()                                              #|
                                                                                                      #|
# Save table                                                                                          #|
np.save('PositivePositive_SensitivityPolyhedrons.npy', SenPolyhedrons)                               #|
                                                                                                      #|
#______________________________________________________________________________________________________



# Pareto minimisation will think NaNs are minimum. Replace NaNs with infinities.
SenPolyhedrons = np.where(np.isnan(SenPolyhedrons), np.inf, SenPolyhedrons)



# _____________ PARALLELISE OBTAINING TABLES OF UNIQUE SENSITIVITY PAIRS _____________
                                                                                     #|
# Function to extract a sensitivity pair                                             #|
def extract_sensitivity_pair(pair_indices, SenPolyhedrons):                          #|
    return SenPolyhedrons[:, pair_indices]                                           #|
                                                                                     #|
# List of index pairs for sensitivity columns                                        #|
sensitivity_pairs = [                                                                #|
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],                                          #|
    [1, 2], [1, 3], [1, 4], [1, 5],                                                  #|
    [2, 3], [2, 4], [2, 5],                                                          #|
    [3, 4], [3, 5],                                                                  #|
    [4, 5]]                                                                          #|
                                                                                     #|
# Parallel extraction of sensitivity pairs                                           #|
sensitivitypairs = Parallel(n_jobs=-1)(                                              #|
    delayed(extract_sensitivity_pair)(pair_indices, SenPolyhedrons)                  #|
    for pair_indices in tqdm(sensitivity_pairs, desc="Creating sensitivity pairs"))  #|
                                                                                     #|
# Assign the results to variables                                                    #|
(SensPair1, SensPair2, SensPair3, SensPair4, SensPair5,                              #|
 SensPair6, SensPair7, SensPair8, SensPair9, SensPair10,                             #|
 SensPair11, SensPair12, SensPair13, SensPair14, SensPair15) = sensitivitypairs      #|
                                                                                     #|
#_____________________________________________________________________________________



#____________________ OBTAIN PARETO FRONTS IN EACH SENSITIVITY SUBSPACE ____________________
                                                                                           #|
def compute_pareto(mask_fn, SensPair):                                                     #|
    mask = paretoset(SensPair, sense=["min", "min"])                                       #|
    paretopoints_sens = SensPair[mask]                                                     #|
    return mask, paretopoints_sens                                                         #|
                                                                                           #|
pairs = [SensPair1, SensPair2, SensPair3, SensPair4, SensPair5,                            #|
         SensPair6, SensPair7, SensPair8, SensPair9, SensPair10,                           #|
         SensPair11, SensPair12, SensPair13, SensPair14, SensPair15]                       #|
                                                                                           #|
pareto_results = Parallel(n_jobs=-1)(                                                      #|
    delayed(compute_pareto)(paretoset, pair)                                               #|
    for pair in pairs)                                                                     #|
                                                                                           #|
pareto_masks = {}                                                                          #|
pareto_points = {}                                                                         #|
                                                                                           #|
for idx, (mask, paretopoints_sens) in enumerate(pareto_results):                           #|
    pareto_masks[f'mask{idx+1}'] = mask                                                    #|
    pareto_points[f'paretoset_SensPair{idx+1}'] = paretopoints_sens                        #|
                                                                                           #|
# Save all Pareto masks in one file                                                        #|
np.savez('PositivePositive_ParetoMasks.npz', **pareto_masks)                               #|
                                                                                           #|
# Save all Pareto points in one file                                                       #|
np.savez('PositivePositive_SensitivityPairsParetos.npz', **pareto_points)                  #|
                                                                                           #|
#___________________________________________________________________________________________



# ___________ CORRESPONDING PARETO FRONTS IN PARAMETER SPACE ____________
                                                                        #|
data1 = np.load('PositivePositive_ParetoMasks.npz')                     #|
for key, value in data1.items():                                        #|
    globals()[key] = value                                              #|
                                                                        #|
# Extract Pareto masks and get corresponding fronts in parameter space  #|
print("Obtaining Pareto points in parameter space... in progress")      #|
pareto_params = {}                                                      #|
for i in range(1, 16):                                                  #|
    mask = data1[f'mask{i}']                                            #|
    pareto_params[f'pareto{i}_params'] = ParamPolyhedron[mask]          #|
print("Obtaining Pareto points in parameter space... complete")         #|
                                                                        #|
# Save the Pareto fronts in parameter space                             #|
print("Saving Pareto fronts in parameter space... in progress")         #|
np.savez('PositivePositive_ParamParetoArrays.npz', **pareto_params)     #|
print("Saving Pareto fronts in parameter space... complete")            #|
                                                                        #|
#________________________________________________________________________

