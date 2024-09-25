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
#______________________________________#|


# ____________________________ DEFINE DYNAMICAL SYSTEM _______________________________
#                               (CAN EDIT THIS CELL)                                 #|                             
                                                                                     #|
# x-nullcline                                                                        #|
def Equ1(x, y, beta_x, n):                                                           #|
    return beta_x / (1 + y**n) - x                                                   #|
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


# ________________________ OUTER CUBE OF PARAMETER SPACE ___________________________
#                               CAN EDIT THIS CELL                                 #|
                                                                                   #|
# Parameter range                                                                  #|
                                                                                   #|
print("Initialising outer parameter cube parameters... in progress")               #|
                                                                                   #|
beta_x_min  = 0.01                                                                 #|
beta_x_max  = 50                                                                   #|
beta_x_no   = 1000                                                                  #|
beta_x_vals = np.linspace(beta_x_min,beta_x_max,beta_x_no)                         #|
                                                                                   #|
beta_y_min  = 0.01                                                                 #|
beta_y_max  = 50                                                                   #|
beta_y_no   = 1000                                                                  #|
beta_y_vals = np.linspace(beta_y_min,beta_y_max,beta_y_no)                         #|
                                                                                   #|
n_min  = 0.01                                                                      #|
n_max  = 10                                                                        #|
n_no   = 1000                                                                       #|
n_vals = np.linspace(n_min,n_max,n_no)                                             #|
                                                                                   #|
print("Initialising outer parameter cube parameters... complete")                  #|
                                                                                   #|
# __________________________________________________________________________________


# _____ TABULATE OUTER CUBE WHILE FILTERING OUT REGIONS OF CUBE WE DO NOT WANT _____
#               IE. CREATING PARAMTER POLYHEDRON SUBSPACE OF INTEREST              #|
                                                                                   #|
#  ---------------------------------------                                         #|
# | beta_x value | beta_y value | n value |                                        #|
# |       #      |       #      |    #    | <--- row 0                             #|
# |       #      |       #      |    #    |                                        #|
# |       #      |       #      |    #    |                                        #|
# |       #      |       #      |    #    | <--- row (beta_x_no)*(beta_y_no)*(n_no)#|
#  ---------------------------------------                                         #|
                                                                                   #|
# Prompt                                                                           #|
print("Creating and saving parameter subspace... in progress")                     #|
                                                                                   #|
# Initialise memory corresponding to largest case scenario                         #|
# ie. outer cube that encompasses polyhedron of interest                           #|
ParamPolyhedron = np.full( (beta_x_no*beta_y_no*n_no , 3) , None)                  #|
                                                                                   #|
# Dummy counter to track current row in table                                      #|
currentrow = 0                                                                     #|
                                                                                   #|
# For each position in parameter space cube                                        #|
for beta_x_val in beta_x_vals:                                                     #|
  for beta_y_val in beta_y_vals:                                                   #|
    for n_val in n_vals:                                                           #|
                                                                                   #|
#      # Check if coordinate lies within region of interest                         #|
#      if not(2<beta_x_val<48 and 2<beta_y_val<48 and 1<n_val<9):                   #| <--- Filtering condition that gives polyhedron
                                                                                   #|                     (can edit)
        # Add paramter values to parameter space data frame                        #|
        ParamPolyhedron[currentrow,:] = np.array([beta_x_val, beta_y_val, n_val])  #|
                                                                                   #|
        # Update current row                                                       #|
        currentrow += 1                                                            #|
                                                                                   #|
# Remove unused rows of None's                                                     #|
ParamPolyhedron = ParamPolyhedron[~np.all(ParamPolyhedron == None, axis=1)]        #|
                                                                                   #|
# Save parameter choices                                                           #|
np.savez('PositiveNegative_ParamPolyhedron.npz', ParamPolyhedron = ParamPolyhedron)#|
                                                                                   #|
# Prompt                                                                           #|
print("Creating and saving parameter subspace... complete")                        #|
                                                                                   #|
# __________________________________________________________________________________



# _____________________________________ TABLE OF X AND Y STEADY STATES ______________________________________
#                                  CAN EDIT INITGUESSES ARRAY IN THIS CELL                                  #|
                                                                                                            #|
#              ---------                                ---------                                           #|
#             |   xss   |                              |   yss   |                                          #|
#             |    #    |                              |    #    |                                          #|
# xss_Shell = |    #    |        and       yss_Shell = |    #    |                                          #|
#             |    #    |                              |    #    |                                          #|
#             |    #    |                              |    #    |                                          #|
#              ---------                                ---------                                           #|
                                                                                                            #|
# Get number of rows in table                                                                               #|
rows = ParamPolyhedron.shape[0]                                                                             #|
                                                                                                            #|
# Create empty cubes to store data                                                                          #|
xssPolyhedron = np.empty((rows,1))                                                                          #|
yssPolyhedron = np.empty((rows,1))                                                                          #|
                                                                                                            #|
# Setting up progress printing of loop                                                                      #|
pbar = tqdm(total=rows, desc="Solving for steady states")                                                   #|
                                                                                                            #|
# For each position in parameter polyhedron                                                                 #|
for rownum in range(rows):                                                                                  #|
                                                                                                            #|
  # Update parameter set                                                                                    #|
  params = ParamPolyhedron[rownum]                                                                          #|
                                                                                                            #|
  # Define initial guesses                                                                                  #|
  # Have placed InitGuesses array here within loop, in case need to make in terms of paramerters:           #|
  # beta_x = params[0], beta_y = params[1], n = params[2]                                                   #|
  InitGuesses = [np.array([1,1]),                                                                           #|
                 np.array([0,0]),                                                                           #|
                 np.array([0.3,12.5]),                                                                      #|
                 np.array([50.5,0.9]),                                                                      #|
                 np.array([67.6,0.9]),                                                                      #|
                 np.array([21,0.9])]                                                                        #|
                                                                                                            #|
  # For each initial guess until one of them is valid                                                       #|
  found_valid = False                                                                                       #|
  for InitGuess in InitGuesses:                                                                             #|
                                                                                                            #|
    # Get output, info dictionary containing fvec, integer flag, message of termination                     #|
    t = 0.0                                                                                                 #|
    output, infodict, intflag, mesg = fsolve(Equs, InitGuess, args=(t,params), xtol=1e-12, full_output=True)#|
                                                                                                            #|
    # Get the steady states xss and yss                                                                     #|
    xss, yss = output                                                                                       #|
                                                                                                            #|
    # Get fvec                                                                                              #|
    fvec = infodict['fvec']                                                                                 #|
                                                                                                            #|
    # Check if steady state is valid and store                                                              #|
    if xss>0.04 and yss>0.04 and np.linalg.norm(fvec)<1e-10 and intflag==1:                                 #|
      xssPolyhedron[rownum] = xss                                                                           #|
      yssPolyhedron[rownum] = yss                                                                           #|
      found_valid = True                                                                                    #|
                                                                                                            #|
      # Update progress bar and break loop                                                                  #|
      pbar.update(1)                                                                                        #|
      break                                                                                                 #|
                                                                                                            #|
  # If not valid, set to nan                                                                                #|
  if not found_valid:                                                                                       #|
    xssPolyhedron[rownum] = (float('nan'))                                                                  #|
    yssPolyhedron[rownum] = (float('nan'))                                                                  #|
                                                                                                            #|
    # Update progress bar                                                                                   #|
    pbar.update(1)                                                                                          #|
                                                                                                            #|
# Close the progress bar                                                                                    #|
pbar.close()                                                                                                #|
                                                                                                            #|
# _________________                                                                                         #|
                                                                                                            #|
# Save arrays                                                                                               #|
print("Saving x steady state and y steady state polyhedrons... in progress")                                #|
np.savez('PositiveNegative_EquilibriumPolyhedrons.npz', xssPolyhedron = xssPolyhedron,                      #|
                                                        yssPolyhedron = yssPolyhedron)                      #|
print("Saving x steady state and y steady state polyhedrons... completed")                                  #|
                                                                                                            #|
#____________________________________________________________________________________________________________


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
    numer = beta_x * (1 + xss**n) * (1 + yss**n)                                                     #|
    denom = n**2 * beta_x * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1 + yss**n)**2            #|
    sensitivity = numer/denom                                                                        #|
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betax(yss)                                                      #|
def S_betax_yss_analytic(xss, yss, n, beta_x, beta_y):                                               #|
    numer = n * beta_x * beta_y * xss**(n-1)                                                         #|
    denom = (1 + xss**n)**2 * yss + n**2 * beta_y * xss**n * yss**n + (1 + xss**n)**2 * yss**(1+n)   #|
    sensitivity = numer/denom                                                                        #|
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betay(xss)                                                      #|
def S_betay_xss_analytic(xss, yss, n, beta_x, beta_y):                                               #|
    numer = n * beta_x * (1 + xss**n) * yss**n                                                       #|
    denom = n**2 * beta_x * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1 + yss**n)**2            #|
    sensitivity = - numer/denom                                                                      #|
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betay(yss)                                                      #|
def S_betay_yss_analytic(xss, yss, n, beta_x, beta_y):                                               #|
    numer = beta_x * beta_y * xss**(n-1) * (1 + xss**n)                                              #|
    denom = (1 + xss**n)**2 * yss + n**2 * beta_y * xss**n * yss**n + (1 + xss**n)**2 * yss**(1+n)   #|
    sensitivity = numer/denom                                                                        #|
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_n(xss)                                                          #|
def S_n_xss_analytic(xss, yss, n, beta_x, beta_y):                                                   #|
    numer = n * beta_x * (n * np.log(xss) + np.log(yss) + np.log(yss) * xss**n) * yss**n             #|
    denom = n**2 * beta_x * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1 + yss**n)**2            #|
    sensitivity = - numer/denom                                                                      #|
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_n(yss)                                                          #|
def S_n_yss_analytic(xss, yss, n, beta_x, beta_y):                                                   #|
    numer = n * beta_y * xss**n * (np.log(xss) + (np.log(xss) - n * np.log(yss)) * yss**n)           #|
    denom = (1 + xss**n)**2 * yss + n**2 * beta_y * xss**n * yss**n + (1 + xss**n)**2 * yss**(1+n)   #|
    sensitivity = numer/denom                                                                        #|
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Update progress bar                                                                                #|
pbar.update(1)                                                                                       #|
# Close the progress bar                                                                             #|
pbar.close()                                                                                         #|
                                                                                                     #|
#_____________________________________________________________________________________________________



# ______________________________________ TABLE OF SENSITIVITIES _______________________________________
                                                                                                      #|
                                                                                                      #|
# Create empty sensitivity space numpy array of shape                                                 #|
#  -------------------------------------------------------------------------------------------------  #|
# | S_{beta_x}(xss) | S_{beta_x}(yss) | S_{beta_y}(xss) | S_{beta_y}(yss) | S_{n}(xss) | S_{n}(yss) | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
# |         #       |         #       |         #       |         #       |      #     |      #     | #|
#  -------------------------------------------------------------------------------------------------  #|
SenPolyhedrons = np.empty([rows, 6])                                                                  #|
                                                                                                      #|
# Setting up progress bar                                                                             #|
pbar = tqdm(total=rows, desc="Calculating sensitivities")                                             #|
                                                                                                      #|
# For each position in parameter polyhedron                                                           #|
for rownum in range(rows):                                                                            #|
                                                                                                      #|
      # get the corresponding parameter values                                                        #|
      beta_x_val = ParamPolyhedron[rownum,0]                                                          #|
      beta_y_val = ParamPolyhedron[rownum,1]                                                          #|
      n_val      = ParamPolyhedron[rownum,2]                                                          #|
                                                                                                      #|
      # get the corresponding steady state values                                                     #|
      xss_val = xssPolyhedron[rownum]                                                                 #|
      yss_val = yssPolyhedron[rownum]                                                                 #|
                                                                                                      #|
      # compute the corresponding sensitivity values                                                  #|
      S_beta_x_xss_val = S_betax_xss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val)        #|
      S_beta_x_yss_val = S_betax_yss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val)        #|
      S_beta_y_xss_val = S_betay_xss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val)        #|
      S_beta_y_yss_val = S_betay_yss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val)        #|
      S_n_xss_val      =     S_n_xss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val)        #|
      S_n_yss_val      =     S_n_yss_analytic(xss_val, yss_val, n_val, beta_x_val, beta_y_val)        #|
                                                                                                      #|
      # Add sensitivity values to sensitivity table                                                   #|
      SenPolyhedrons[rownum,:] = np.array([S_beta_x_xss_val,                                          #|
                                           S_beta_x_yss_val,                                          #|
                                           S_beta_y_xss_val,                                          #|
                                           S_beta_y_yss_val,                                          #|
                                           S_n_xss_val     ,                                          #|
                                           S_n_yss_val    ]).flatten()                                #|
                                                                                                      #|
      # Update progress bar                                                                           #|
      pbar.update(1)                                                                                  #|
                                                                                                      #|
# Close the progress bar                                                                              #|
pbar.close()                                                                                          #|
                                                                                                      #|
# Save table                                                                                          #|
print("Saving sensitivity polyhedrons... in progress")                                                #|
np.savez('PositiveNegative_SensitivityPolyhedrons.npz', SenPolyhedrons = SenPolyhedrons)              #|
print("Saving sensitivity polyhedrons... completed")                                                  #|
                                                                                                      #|
#______________________________________________________________________________________________________



# _______________________ PARETO FRONTS FOR EACH PAIR OF SENSITIVITY POLYHEDRONS _________________________
                                                                                                         #|
# There may be NaNs in the array. Pareto minimisation will think NaNs                                    #|
# are minimum. We don't want this. Let's replace NaNs with infinities.                                   #|
print("Replacing NaN's with Inf's... in progress")                                                       #|
SenPolyhedrons = np.where(np.isnan(SenPolyhedrons), np.inf, SenPolyhedrons)                              #|
print("Replacing NaN's with Inf's... completed")                                                         #|
                                                                                                         #|
# _______________________                                                                                #|
                                                                                                         #|
# Initialise loading bar                                                                                 #|
pbar = tqdm(total=15, desc="Creating sensitivity pair tables")                                           #|
                                                                                                         #|
# Create a new table for each unique pair of sensitivites                                                #|
SensPair1  = SenPolyhedrons[:, [0, 1]]      #Columns: 'S_beta_x_xss', 'S_beta_x_yss'                     #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair2  = SenPolyhedrons[:, [0, 2]]      #Columns: 'S_beta_x_xss', 'S_beta_y_xss'                     #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair3  = SenPolyhedrons[:, [0, 3]]      #Columns: 'S_beta_x_xss', 'S_beta_y_yss'                     #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair4  = SenPolyhedrons[:, [0, 4]]      #Columns: 'S_beta_x_xss', 'S_n_xss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair5  = SenPolyhedrons[:, [0, 5]]      #Columns: 'S_beta_x_xss', 'S_n_yss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair6  = SenPolyhedrons[:, [1, 2]]      #Columns: 'S_beta_x_yss', 'S_beta_y_xss'                     #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair7  = SenPolyhedrons[:, [1, 3]]      #Columns: 'S_beta_x_yss', 'S_beta_y_yss'                     #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair8  = SenPolyhedrons[:, [1, 4]]      #Columns: 'S_beta_x_yss', 'S_n_xss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair9  = SenPolyhedrons[:, [1, 5]]      #Columns: 'S_beta_x_yss', 'S_n_yss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair10 = SenPolyhedrons[:, [2, 3]]      #Columns: 'S_beta_y_xss', 'S_beta_y_yss'                     #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair11 = SenPolyhedrons[:, [2, 4]]      #Columns: 'S_beta_y_xss', 'S_n_xss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair12 = SenPolyhedrons[:, [2, 5]]      #Columns: 'S_beta_y_xss', 'S_n_yss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair13 = SenPolyhedrons[:, [3, 4]]      #Columns: 'S_beta_y_yss', 'S_n_xss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair14 = SenPolyhedrons[:, [3, 5]]      #Columns: 'S_beta_y_yss', 'S_n_yss'                          #|
pbar.update(1) #Update progress bar                                                                      #|
SensPair15 = SenPolyhedrons[:, [4, 5]]      #Columns: 'S_n_xss', 'S_n_yss'                               #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
# Close the progress bar                                                                                 #|
pbar.close()                                                                                             #|
                                                                                                         #|
# _______________________                                                                                #|
                                                                                                         #|
# Initialise loading bar                                                                                 #|
pbar = tqdm(total=15, desc="Obtaining Pareto fronts")                                                    #|
                                                                                                         #|
# Run Pareto tool with minimisation setting to get a mask.                                               #|
# Each mask is an array of the form [True, False, True, ...].                                            #|
# Indexing another array with this mask will remove rows corresponding to Falses.                        #|
# Eg:  dummy       = [1   , 2   , 3    ]                                                                 #|
#      mask        = [True, True, False]                                                                 #|
#      dummy[mask] = [1   , 2          ]                                                                 #|
                                                                                                         #|
mask1 = paretoset(SensPair1, sense=["min", "min"])                                                       #|
paretoset_SensPair1 = SensPair1[mask1]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask2 = paretoset(SensPair2, sense=["min", "min"])                                                       #|
paretoset_SensPair2 = SensPair2[mask2]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask3 = paretoset(SensPair3, sense=["min", "min"])                                                       #|
paretoset_SensPair3 = SensPair3[mask3]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask4 = paretoset(SensPair4, sense=["min", "min"])                                                       #|
paretoset_SensPair4 = SensPair4[mask4]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask5 = paretoset(SensPair5, sense=["min", "min"])                                                       #|
paretoset_SensPair5 = SensPair5[mask5]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask6 = paretoset(SensPair6, sense=["min", "min"])                                                       #|
paretoset_SensPair6 = SensPair6[mask6]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask7 = paretoset(SensPair7, sense=["min", "min"])                                                       #|
paretoset_SensPair7 = SensPair7[mask7]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask8 = paretoset(SensPair8, sense=["min", "min"])                                                       #|
paretoset_SensPair8 = SensPair8[mask8]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask9 = paretoset(SensPair9, sense=["min", "min"])                                                       #|
paretoset_SensPair9 = SensPair9[mask9]                                                                   #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask10 = paretoset(SensPair10, sense=["min", "min"])                                                     #|
paretoset_SensPair10 = SensPair10[mask10]                                                                #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask11 = paretoset(SensPair11, sense=["min", "min"])                                                     #|
paretoset_SensPair11 = SensPair11[mask11]                                                                #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask12 = paretoset(SensPair12, sense=["min", "min"])                                                     #|
paretoset_SensPair12 = SensPair12[mask12]                                                                #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask13 = paretoset(SensPair13, sense=["min", "min"])                                                     #|
paretoset_SensPair13 = SensPair13[mask13]                                                                #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask14 = paretoset(SensPair14, sense=["min", "min"])                                                     #|
paretoset_SensPair14 = SensPair14[mask14]                                                                #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
mask15 = paretoset(SensPair15, sense=["min", "min"])                                                     #|
paretoset_SensPair15 = SensPair15[mask15]                                                                #|
pbar.update(1) #Update progress bar                                                                      #|
                                                                                                         #|
# Close the progress bar                                                                                 #|
pbar.close()                                                                                             #|
                                                                                                         #|
# Save masks                                                                                             #|
print("Saving Pareto masks... in progress")                                                              #|
np.savez('PositiveNegative_ParetoMasks.npz', mask1  = mask1 ,                                            #|
                                             mask2  = mask2 ,                                            #|
                                             mask3  = mask3 ,                                            #|
                                             mask4  = mask4 ,                                            #|
                                             mask5  = mask5 ,                                            #|
                                             mask6  = mask6 ,                                            #|
                                             mask7  = mask7 ,                                            #|
                                             mask8  = mask8 ,                                            #|
                                             mask9  = mask9 ,                                            #|
                                             mask10 = mask10,                                            #|
                                             mask11 = mask11,                                            #|
                                             mask12 = mask12,                                            #|
                                             mask13 = mask13,                                            #|
                                             mask14 = mask14,                                            #|
                                             mask15 = mask15)                                            #|
print("Saving Pareto masks... completed")                                                                #|
                                                                                                         #|
# Save tables of Pareto points for each sensitivity pair                                                 #|
print("Saving Paretos of sensitivity pairs... in progress")                                              #|
np.savez('PositiveNegative_SensitivityPairsParetos.npz',paretoset_SensPair1  = paretoset_SensPair1,      #|
                                                        paretoset_SensPair2  = paretoset_SensPair2,      #|
                                                        paretoset_SensPair3  = paretoset_SensPair3,      #|
                                                        paretoset_SensPair4  = paretoset_SensPair4,      #|
                                                        paretoset_SensPair5  = paretoset_SensPair5,      #|
                                                        paretoset_SensPair6  = paretoset_SensPair6,      #|
                                                        paretoset_SensPair7  = paretoset_SensPair7,      #|
                                                        paretoset_SensPair8  = paretoset_SensPair8,      #|
                                                        paretoset_SensPair9  = paretoset_SensPair9,      #|
                                                        paretoset_SensPair10 = paretoset_SensPair10,     #|
                                                        paretoset_SensPair11 = paretoset_SensPair11,     #|
                                                        paretoset_SensPair12 = paretoset_SensPair12,     #|
                                                        paretoset_SensPair13 = paretoset_SensPair13,     #|
                                                        paretoset_SensPair14 = paretoset_SensPair14,     #|
                                                        paretoset_SensPair15 = paretoset_SensPair15)     #|
print("Saving Paretos of sensitivity pair... complete")                                                  #|
                                                                                                         #|
#_________________________________________________________________________________________________________




# __________________ CORRESPONDING PARETO FRONTS IN PARAMETER SPACE ____________________
                                                                                       #|
# Get the corresponding pareto fronts in parameter space                               #|
print("Obtaining Pareto points in parameter space... in progress")                     #|
pareto1_params  = ParamPolyhedron[mask1]                                               #|
pareto2_params  = ParamPolyhedron[mask2]                                               #|
pareto3_params  = ParamPolyhedron[mask3]                                               #|
pareto4_params  = ParamPolyhedron[mask4]                                               #|
pareto5_params  = ParamPolyhedron[mask5]                                               #|
pareto6_params  = ParamPolyhedron[mask6]                                               #|
pareto7_params  = ParamPolyhedron[mask7]                                               #|
pareto8_params  = ParamPolyhedron[mask8]                                               #|
pareto9_params  = ParamPolyhedron[mask9]                                               #|
pareto10_params = ParamPolyhedron[mask10]                                              #|
pareto11_params = ParamPolyhedron[mask11]                                              #|
pareto12_params = ParamPolyhedron[mask12]                                              #|
pareto13_params = ParamPolyhedron[mask13]                                              #|
pareto14_params = ParamPolyhedron[mask14]                                              #|
pareto15_params = ParamPolyhedron[mask15]                                              #|
print("Obtaining Pareto points in parameter space... complete")                        #|
                                                                                       #|
# Save the arrays                                                                      #|
print("Saving Pareto fronts in parameter space... in progress")                        #|
np.savez('PositiveNegative_ParamParetoArrays.npz', pareto1_params  =  pareto1_params,  #|
                                                   pareto2_params  =  pareto2_params,  #|
                                                   pareto3_params  =  pareto3_params,  #|
                                                   pareto4_params  =  pareto4_params,  #|
                                                   pareto5_params  =  pareto5_params,  #|
                                                   pareto6_params  =  pareto6_params,  #|
                                                   pareto7_params  =  pareto7_params,  #|
                                                   pareto8_params  =  pareto8_params,  #|
                                                   pareto9_params  =  pareto9_params,  #|
                                                   pareto10_params = pareto10_params,  #|
                                                   pareto11_params = pareto11_params,  #|
                                                   pareto12_params = pareto12_params,  #|
                                                   pareto13_params = pareto13_params,  #|
                                                   pareto14_params = pareto14_params,  #|
                                                   pareto15_params = pareto15_params)  #|
print("Saving Pareto fronts in parameter space... in progress")                        #|
                                                                                       #|
#_______________________________________________________________________________________

