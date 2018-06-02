# A nested logit model of mode choice for CIV1538
# This model has nests for transit auto and transit active modes
# Nesting coefficient is > 1 so correct nesting structure
# Separate walk and wait time coefficients
# Break out cost and combine travel times (can breakup VOT by income or other measure later)
# Cost divided by distance
# Introduce dummy variables
# Alter nest structure to remove transit auto nest and add carpool nest
# Reconfigure time parameters
# Attempt to get a negative time parameter using age then sex
# Try using simple time and cost parameters
# Cost for parking and driving similar, so fair to combine them
# Everything seems to be working except nest. Try park and rides together.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import collections

df = pd.read_csv('../data/mode.csv')

# Adjust variables to scale parameters
df['AD_TT'] = df['AD_TT'] / df['HW_Dist']
df['AC_TT'] = df['AC_TT'] / df['HW_Dist']
df['PT_IVTT'] = df['PT_IVTT'] / df['HW_Dist']
df['PT_Walk_Time'] = df['PT_Walk_Time'] / df['HW_Dist']
df['PT_Wait_Time'] = df['PT_Wait_Time'] / df['HW_Dist']
df['PR_HT_IVTT'] = df['PR_HT_IVTT'] / df['HW_Dist']
df['PR_TW_IVTT'] = df['PR_TW_IVTT'] / df['HW_Dist']
df['PR_Walk_Time'] = df['PR_Walk_Time'] / df['HW_Dist']
df['PR_Wait_Time'] = df['PR_Wait_Time'] / df['HW_Dist']
df['CPR_HT_IVTT'] = df['CPR_HT_IVTT'] / df['HW_Dist']
df['CPR_TW_IVTT'] = df['CPR_TW_IVTT'] / df['HW_Dist']
df['CPR_Walk_Time'] = df['CPR_Walk_Time'] / df['HW_Dist']
df['CPR_Wait_Time'] = df['CPR_Wait_Time'] / df['HW_Dist']
df['CR_C_Time'] = df['CR_C_Time'] / df['HW_Dist']
df['CR_T_Time'] = df['CR_T_Time'] / df['HW_Dist']
df['CR_OVTT'] = df['CR_OVTT'] / df['HW_Dist']
df['C_Time'] = df['C_Time'] / df['HW_Dist']
df['W_Time'] = df['W_Time'] / df['HW_Dist']

# Dummy variable conversion
dummyEmpStatus = pd.get_dummies(df['Emp_Status'])
dummyFlexWork = pd.get_dummies(df['Flex_Work'])
dummyCompWork = pd.get_dummies(df['Comp_Work'])
dummyAge = pd.get_dummies(df['Age_Bin'])
dummySex = pd.get_dummies(df['Sex'])
dummyACFreq = pd.get_dummies(df['AC_Freq'])
dummyTransFreq = pd.get_dummies(df['PT_Freq'])
dummyCycleFreq = pd.get_dummies(df['C_Freq'])
dummyWalkFreq = pd.get_dummies(df['W_Freq'])

beta_dict = collections.OrderedDict([('ASC_AD', 1.5136), ('TIME', -0.0306), ('TIME', -0.0306), ('COST', -0.0403),
            ('ASC_AC', 0.2625), ('ASC_PT', 0.6305), ('TRANS_TRANSFER', -0.1924),
            ('ASC_PR', -1.9527), ('ASC_CPR', -0.4707), ('ASC_CR', -0.8019), ('ASC_BIKE', -17.6974),
            ('CP_FREQ_1', 0.8701), ('TRANS_FREQ_1', 2.1596), ('TRANS_FREQ_2', 1.1083),
            ('TRANS_FREQ_3', -0.5991), ('C_FREQ_1', 21.0237), ('C_FREQ_2', 17.5739),
            ('C_FREQ_3', 16.3491), ('W_FREQ_1', 4.5196), ('WPT_AD', 0.6065), ('WPT_PR', 0.8722),
            ('WPT_W', -0.6367), ('WFLEX_AD', 0.0592), ('WFLEX_PR', -0.3061), ('WFLEX_W', -0.4058),
            ('WCOMP_AD', -0.6120), ('WCOMP_PR', -0.5680), ('WCOMP_C', 0.5780), ('WCOMP_W', -0.5038),
            ('MU_CP', 2.2147)])

beta_dict_const = collections.OrderedDict([('ASC_AD', 1.5136), ('ASC_AC', 0.2625), ('ASC_PT', 0.6305),
            ('ASC_PR', -1.9527), ('ASC_CPR', -0.4707), ('ASC_CR', -0.8019), ('ASC_BIKE', -17.6974)])

params = beta_dict.values()
params_const = beta_dict_const.values()

def logit_utils(params, const):
    if const == 0:
        # Update the dictionary of betas to reflect the new param estimates
        for i, k in enumerate(beta_dict):
            beta_dict[k] = params[i]
        # Define utility functions for each model
        V_AD = beta_dict['TIME'] * df['AD_TT'] \
            + beta_dict['COST'] * df['AD_Park_Cost'] \
            + beta_dict['COST'] * df['AD_Drive_Cost'] \
            + beta_dict['WPT_AD'] * dummyEmpStatus[1] + beta_dict['WFLEX_AD'] * dummyFlexWork[1] \
            + beta_dict['WCOMP_AD'] * dummyCompWork[1] + beta_dict['ASC_AD']
        V_AC = beta_dict['TIME'] * df['AC_TT'] \
            + beta_dict['COST'] * df['AC_Park_Cost'] \
            + beta_dict['COST'] * df['AC_Drive_Cost'] + beta_dict['CP_FREQ_1'] * dummyACFreq[1] \
            + beta_dict['ASC_AC']
        V_PT = beta_dict['COST'] * df['PT_Fare'] + beta_dict['TIME'] * df['PT_IVTT'] \
            + beta_dict['TIME'] * (df['PT_Wait_Time'] + df['PT_Walk_Time']) + beta_dict['TRANS_TRANSFER'] * df['PT_Transfers'] \
            + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2] \
            + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3] \
            + beta_dict['ASC_PT']
        V_PR = beta_dict['TIME'] * (df['PR_HT_IVTT'] + df['PR_HT_IVTT']) \
            + beta_dict['COST'] * df['PR_Transit_Cost'] + beta_dict['COST'] * df['PR_Drive_Cost'] \
            + beta_dict['TIME'] * (df['PR_Wait_Time'] + df['PR_Walk_Time']) + beta_dict['TRANS_TRANSFER'] * df['PR_Transfers'] \
            + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2] \
            + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3] + beta_dict['WPT_PR'] * dummyEmpStatus[1] \
            + beta_dict['WFLEX_PR'] * dummyFlexWork[1] + beta_dict['WCOMP_PR'] * dummyCompWork[1] + beta_dict['ASC_PR']
        V_CPR = beta_dict['COST'] * df['CPR_Drive_Cost'] + beta_dict['COST'] * df['CPR_Transit_Cost']  \
            + beta_dict['TIME'] * (df['CPR_TW_IVTT'] + df['CPR_HT_IVTT']) \
            + beta_dict['TIME'] * (df['CPR_Wait_Time'] + df['CPR_Walk_Time']) + beta_dict['TRANS_TRANSFER'] * df['CPR_Transfers'] \
            + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2]\
            + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3]  \
            + beta_dict['ASC_CPR']
        V_CR = beta_dict['TIME'] * (df['CR_T_Time']) \
            + beta_dict['TRANS_TRANSFER'] * df['CR_Transfers'] + beta_dict['COST'] * df['CR_Cost'] \
            + beta_dict['TIME'] * df['CR_OVTT'] + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] \
            + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2] + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3] \
            + beta_dict['ASC_CR']
        V_C = beta_dict['TIME'] * df['C_Time'] \
            + beta_dict['C_FREQ_1'] * dummyCycleFreq[1] + beta_dict['C_FREQ_2'] * dummyCycleFreq[2] \
            + beta_dict['C_FREQ_3'] * dummyCycleFreq[3] \
            + beta_dict['WCOMP_C'] * dummyCompWork[1] + beta_dict['ASC_BIKE']
        V_W = beta_dict['TIME'] * df['W_Time'] \
            + beta_dict['WPT_W'] * dummyEmpStatus[1] + beta_dict['W_FREQ_1'] * dummyWalkFreq[1] \
            + beta_dict['WFLEX_W'] * dummyFlexWork[1] + beta_dict['WCOMP_W'] * dummyCompWork[1]
    if const == 1:
        # Update the dictionary of betas to reflect the new param estimates
        for i, k in enumerate(beta_dict_const):
            beta_dict[k] = params[i]
        beta_dict['MU_CP'] = 1
        beta_dict['MU_TRANS_ACT'] = 1
        # Define utility functions for each model
        V_AD = beta_dict['ASC_AD']
        V_AC = beta_dict['ASC_AC']
        V_PT = beta_dict['ASC_PT']
        V_PR = beta_dict['ASC_PR']
        V_CPR = beta_dict['ASC_CPR']
        V_CR = beta_dict['ASC_CR']
        V_C = beta_dict['ASC_BIKE']
        V_W = 0
    return V_AD, V_AC, V_PT, V_PR, V_CPR, V_CR, V_C, V_W

def logit_prob(params, const):
    # Get the utility functions
    V_AD, V_AC, V_PT, V_PR, V_CPR, V_CR, V_C, V_W = logit_utils(params, const)
    # Lower nest calculations
    # Carpool
    PN_AC = df['AV2'] * np.exp(beta_dict['MU_CP'] * V_AC)
    PN_CPR = df['AV5'] * np.exp(beta_dict['MU_CP'] * V_CPR)
    PD_CP = PN_AC + PN_CPR
    PL_AC = PN_AC / PD_CP
    PL_CPR = PN_CPR / PD_CP

    PL_AC = PL_AC.fillna(1)
    PL_CPR = PL_CPR.fillna(1)


    # Upper nest calculations
    PUN_AD = df['AV1'] * np.exp(V_AD)
    PUN_CP = (df['AV2'] + df['AV5'] >0) \
            * np.exp((1/beta_dict['MU_CP']) * np.log(PL_AC + PL_CPR))
    PUN_PR = df['AV4'] * np.exp(V_PR)
    PUN_PT = df['AV3'] * np.exp(V_PT)
    PUN_CR = df['AV6'] * np.exp(V_CR)
    PUN_C = df['AV7'] * np.exp(V_C)
    PUN_W = df['AV8'] * np.exp(V_W)
    PUD = PUN_AD + PUN_CP + PUN_PR + PUN_PT + PUN_CR + PUN_C + PUN_W

    PU_AD = PUN_AD / PUD
    PU_CP = PUN_CP / PUD
    PU_PR = PUN_PR / PUD
    PU_PT = PUN_PT / PUD
    PU_CR = PUN_CR / PUD
    PU_C = PUN_C / PUD
    PU_W = PUN_W / PUD

    PU_AD = PU_AD.fillna(1)
    PU_CP = PU_CP.fillna(1)
    PU_PR = PU_PR.fillna(1)
    PU_PT = PU_PT.fillna(1)
    PU_CR = PU_CR.fillna(1)
    PU_C = PU_C.fillna(1)
    PU_W = PU_W.fillna(1)

    return PU_AD, PL_AC, PU_PT, PL_CPR, PU_CR, PU_PR, PU_C, PU_W, PU_CP

# Calculate unconditional probabilities from conditional probabilities
def sum_prob(params, const):
    PU_AD, PL_AC, PU_PT, PL_CPR, PU_CR, PU_PR, PU_C, PU_W, PU_CP = \
    logit_prob(params, const)

    return PU_AD, PL_AC * PU_CP, PU_PT, PL_CPR * PU_CP, PU_CR, PU_PR, PU_C, PU_W

def nest_logit(params, const):

    CH = {'AD':1, 'AC':2, 'PT':3, 'PR':4, 'CPR':5, 'CR':6, 'C':7, 'W':8}

    # Get probabilies
    PU_AD, PL_AC, PU_PT, PL_CPR, PU_CR, PU_PR, PU_C, PU_W, PU_CP = \
    logit_prob(params, const)


    PU_PT = PU_PT.replace(0, 1)
    PL_AC = PL_AC.replace(0, 1)
    PL_CPR = PL_CPR.replace(0, 1)
    PU_CR = PU_CR.replace(0, 1)
    PU_AD = PU_AD.replace(0, 1)
    PU_CP = PU_CP.replace(0, 1)
    PU_PR = PU_PR.replace(0, 1)
    PU_C = PU_C.replace(0, 1)
    PU_W = PU_W.replace(0, 1)


    # Lower nest calculations

    LL_AC = np.log(PL_AC) * (df['CHOICE'] == CH['AC'])
    LL_CPR = np.log(PL_CPR) * (df['CHOICE'] == CH['CPR'])

    llfun = LL_AC.sum() + LL_CPR.sum()

    # Upper nest calculations

    LL_AD = (df['CHOICE'] == CH['AD']) * np.log(PU_AD)
    LL_CP = ((df['CHOICE'] == CH['AC']) | (df['CHOICE'] == CH['CPR'])) * np.log(PU_CP)
    LL_PR = (df['CHOICE'] == CH['PR']) * np.log(PU_PR)
    LL_PT = (df['CHOICE'] == CH['PT']) * np.log(PU_PT)
    LL_CR = (df['CHOICE'] == CH['CR']) * np.log(PU_CR)
    LL_C = (df['CHOICE'] == CH['C']) * np.log(PU_C)
    LL_W = (df['CHOICE'] == CH['W']) * np.log(PU_W)

    # Add the upper nest values to the lower nest values in the loglikelihood
    llfun = llfun + LL_AD.sum() + LL_CP.sum() + LL_PR.sum() + LL_PT.sum() + LL_CR.sum() + LL_C.sum() + LL_W.sum()
    return -1 * llfun


res = minimize(nest_logit, params, args=(0))
res_const = minimize(nest_logit, params_const, args=(1))
params = res.x

sd = res.hess_inv.diagonal()
tstat = params/sd

data = np.column_stack((params,sd,tstat))
res_df = pd.DataFrame(data, beta_dict.keys(), ['value', 'std err.', 't-stat'])
res_df['loglike'] = -1 * res.fun
res_df = res_df.round(4)

print(res_df)
res_df.to_latex('./params.txt')

# # Calculate rho2-const
print(res.fun)
print(res_const.fun)
rho_const = 1 - (res.fun/res_const.fun)
print(rho_const)

# Calculate the lower choleksy
L = np.linalg.cholesky(res.hess_inv)
ITERATE = 100
MODES = 8
array_prob = np.empty((ITERATE,MODES))
# Simulate 100 runs of the model for probabilities
for i in range(ITERATE):
    eta = np.random.standard_normal(len(params))
    sim_params = params + np.dot(L,eta)
    lst_prob = sum_prob(sim_params, 0)
    ave_prob = np.mean(lst_prob,axis=1)
    array_prob[i] = ave_prob
np.savetxt("./confidence_interval.csv", array_prob)
