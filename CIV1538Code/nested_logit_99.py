# A nested logit model of mode choice for CIV1538
# This model has nests for transit auto and transit active modes
# Nesting coefficient is > 1 so correct nesting structure
# Separate walk and wait time coefficients
# Break out cost and combine travel times (can breakup VOT by income or other measure later)
# Cost divided by distance
# Introduce dummy variables
# Alter nest structure to remove transit auto nest and add carpool nest
# Add age to model as interaction on travel time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import collections

df = pd.read_csv('../data/mode.csv')

# Adjust variables to scale parameters
df['AD_TT'] = (df['AD_TT'])**2 / 10**3
df['AC_TT'] = (df['AC_TT'])**2 / 10**3
df['PT_IVTT'] = (df['PT_IVTT'])**2 / 10**3
df['PT_Walk_Time'] = (df['PT_Walk_Time'])**2 / 10**3
df['PT_Wait_Time'] = (df['PT_Wait_Time'])**0.5
df['PR_HT_IVTT'] = (df['PR_HT_IVTT'])**2 / 10**3
df['PR_TW_IVTT'] = (df['PR_TW_IVTT'])**2 / 10**3
df['PR_Walk_Time'] = (df['PR_Walk_Time'])**2 / 10**3
df['PR_Wait_Time'] = (df['PR_Wait_Time'])**0.5
df['CPR_HT_IVTT'] = (df['CPR_HT_IVTT'])**2 / 10**3
df['CPR_TW_IVTT'] = (df['CPR_TW_IVTT'])**2 / 10**3
df['CPR_Walk_Time'] = (df['CPR_Walk_Time'])**2 / 10**3
df['CPR_Wait_Time'] = (df['CPR_Wait_Time'])**0.5
df['CR_C_Time'] = (df['CR_C_Time'])**2 / 10**3
df['CR_T_Time'] = (df['CR_T_Time'])**2 / 10**3
df['CR_OVTT'] = (df['CR_OVTT'])**0.5
df['C_Time'] = (df['C_Time'])**2 / 10**3
df['W_Time'] = (df['W_Time'])**2 / 10**3

# Dummy variable conversion
dummyEmpStatus = pd.get_dummies(df['Emp_Status'])
dummyFlexWork = pd.get_dummies(df['Flex_Work'])
dummyCompWork = pd.get_dummies(df['Comp_Work'])
dummyAge = pd.get_dummies(df['Age_Bin'])
dummyACFreq = pd.get_dummies(df['AC_Freq'])
dummyTransFreq = pd.get_dummies(df['PT_Freq'])
dummyCycleFreq = pd.get_dummies(df['C_Freq'])
dummyWalkFreq = pd.get_dummies(df['W_Freq'])

beta_dict = collections.OrderedDict([('ASC_AD', -0.1), ('TRAVEL_TIME', -0.1), ('DRIVE_COST', -0.1),
            ('ASC_AC', -0.1), ('ASC_PT', -0.1), ('PARK_COST', -0.1),
            ('TRANS_TRANSFER', -0.1), ('ASC_PR', -0.1), ('ASC_CPR', -0.1), ('ASC_CR', -0.1),
            ('ASC_BIKE', -0.1), ('FARE', -0.1), ('WAIT_TIME', -0.1), ('CP_FREQ_1', -0.1),
            ('TRANS_FREQ_1', -0.1), ('TRANS_FREQ_2', -0.1), ('TRANS_FREQ_3', -0.1), ('C_FREQ_1', -0.1),
            ('C_FREQ_2', -0.1), ('C_FREQ_3', -0.1), ('W_FREQ_1', -0.1),
            ('MU_CP', 1), ('MU_TRANS_ACT', 1)])

params = beta_dict.values()

def nest_logit(params):
    # Update the dictionary of betas to reflect the new param estimates
    for i, k in enumerate(beta_dict):
        beta_dict[k] = params[i]

    # Define utility functions for each model
    V_AD = beta_dict['TRAVEL_TIME'] * df['AD_TT'] \
        + beta_dict['PARK_COST'] * df['AD_Park_Cost'] \
        + beta_dict['DRIVE_COST'] * (df['AD_Drive_Cost'] / df['HW_Dist']) + beta_dict['ASC_AD']
    V_AC = beta_dict['TRAVEL_TIME'] * df['AC_TT'] \
        + beta_dict['PARK_COST'] * df['AC_Park_Cost'] \
        + beta_dict['DRIVE_COST'] * (df['AC_Drive_Cost'] / df['HW_Dist']) + beta_dict['CP_FREQ_1'] * dummyACFreq[1] \
        + beta_dict['ASC_AC']
    V_PT = beta_dict['FARE'] * df['PT_Fare'] + beta_dict['TRAVEL_TIME'] * (df['PT_IVTT'] + df['PT_Walk_Time']) \
        + beta_dict['WAIT_TIME'] * df['PT_Wait_Time'] + beta_dict['TRANS_TRANSFER'] * df['PT_Transfers'] \
        + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2] \
        + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3] + beta_dict['ASC_PT']
    V_PR = beta_dict['TRAVEL_TIME'] * (df['PR_HT_IVTT'] + df['PR_HT_IVTT'] + df['PR_Walk_Time']) \
        + beta_dict['FARE'] * df['PR_Transit_Cost'] + beta_dict['DRIVE_COST'] * (df['PR_Drive_Cost'] / df['HW_Dist']) \
        + beta_dict['WAIT_TIME'] * df['PR_Wait_Time'] + beta_dict['TRANS_TRANSFER'] * df['PR_Transfers'] \
        + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2] \
        + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3] + beta_dict['ASC_PR']
    V_CPR = beta_dict['DRIVE_COST'] * (df['CPR_Drive_Cost'] / df['HW_Dist']) + beta_dict['FARE'] * df['CPR_Transit_Cost'] \
        + beta_dict['TRAVEL_TIME'] * (df['CPR_TW_IVTT'] + df['CPR_HT_IVTT'] + df['CPR_Walk_Time']) \
        + beta_dict['WAIT_TIME'] * df['CPR_Wait_Time'] + beta_dict['TRANS_TRANSFER'] * df['CPR_Transfers'] \
        + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2]\
        + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3] + beta_dict['ASC_CPR']
    V_CR = beta_dict['TRAVEL_TIME'] * (df['CR_C_Time'] + df['CR_T_Time']) \
        + beta_dict['TRANS_TRANSFER'] * df['CR_Transfers'] + beta_dict['FARE'] * df['CR_Cost'] \
        + beta_dict['WAIT_TIME'] * df['CR_OVTT'] + beta_dict['TRANS_FREQ_1'] * dummyTransFreq[1] \
        + beta_dict['TRANS_FREQ_2'] * dummyTransFreq[2] + beta_dict['TRANS_FREQ_3'] * dummyTransFreq[3] \
        + beta_dict['ASC_CR']
    V_C = beta_dict['TRAVEL_TIME'] * df['C_Time'] \
        + beta_dict['C_FREQ_1'] * dummyCycleFreq[1] + beta_dict['C_FREQ_2'] * dummyCycleFreq[2] \
        + beta_dict['C_FREQ_3'] * dummyCycleFreq[3] \
        + beta_dict['ASC_BIKE']
    V_W = beta_dict['TRAVEL_TIME'] * df['W_Time'] \
        + beta_dict['W_FREQ_1'] * dummyWalkFreq[1]

    CH = {'AD':1, 'AC':2, 'PT':3, 'PR':4, 'CPR':5, 'CR':6, 'C':7, 'W':8}

    # Lower nest calculations

    # Carpool
    PN_AC = df['AV2'] * np.exp(beta_dict['MU_CP'] * V_AC)
    PN_CPR = df['AV5'] * np.exp(beta_dict['MU_CP'] * V_CPR)
    PD_CP = PN_AC + PN_CPR
    PL_AC = PN_AC / PD_CP
    PL_CPR = PN_CPR / PD_CP

    # Transit active
    PN_PT = df['AV3'] * np.exp(beta_dict['MU_TRANS_ACT'] * V_PT)
    PN_CR = df['AV6'] * np.exp(beta_dict['MU_TRANS_ACT'] * V_CR)
    PD_Trans_Act = PN_PT + PN_CR
    PL_PT = PN_PT / PD_Trans_Act
    PL_CR = PN_CR / PD_Trans_Act

    PL_PT = PL_PT.fillna(1)
    PL_AC = PL_AC.fillna(1)
    PL_CPR = PL_CPR.fillna(1)
    PL_CR = PL_CR.fillna(1)
    PL_PT = PL_PT.replace(0, 1)
    PL_AC = PL_AC.replace(0, 1)
    PL_CPR = PL_CPR.replace(0, 1)
    PL_CR = PL_CR.replace(0, 1)

    LL_PT = np.log(PL_PT) * (df['CHOICE'] == CH['PT'])
    LL_CR = np.log(PL_CR) * (df['CHOICE'] == CH['CR'])
    LL_AC = np.log(PL_AC) * (df['CHOICE'] == CH['AC'])
    LL_CPR = np.log(PL_CPR) * (df['CHOICE'] == CH['CPR'])

    llfun = LL_PT.sum() + LL_AC.sum() + LL_CPR.sum() + LL_CR.sum()

    # Upper nest calculations
    PUN_AD = df['AV1'] * np.exp(V_AD)
    PUN_CP = (df['AV2'] + df['AV5'] >0) \
            * np.exp((1/beta_dict['MU_CP']) * np.log(PL_AC + PL_CPR))
    PUN_PR = df['AV4'] * np.exp(V_PR)
    PUN_Trans_Act = (df['AV3'] + df['AV6'] >0) \
            * np.exp((1/beta_dict['MU_TRANS_ACT']) * np.log(PL_PT + PL_CR))
    PUN_C = df['AV7'] * np.exp(V_C)
    PUN_W = df['AV8'] * np.exp(V_W)
    PUD = PUN_AD + PUN_CP + PUN_PR + PUN_Trans_Act + PUN_C + PUN_W

    PU_AD = PUN_AD / PUD
    PU_CP = PUN_CP / PUD
    PU_PR = PUN_PR / PUD
    PU_Trans_Act = PUN_Trans_Act / PUD
    PU_C = PUN_C / PUD
    PU_W = PUN_W / PUD

    PU_AD = PU_AD.fillna(1)
    PU_CP = PU_CP.fillna(1)
    PU_PR = PU_PR.fillna(1)
    PU_Trans_Act = PU_Trans_Act.fillna(1)
    PU_C = PU_C.fillna(1)
    PU_W = PU_W.fillna(1)
    PU_AD = PU_AD.replace(0, 1)
    PU_CP = PU_CP.replace(0, 1)
    PU_PR = PU_PR.replace(0, 1)
    PU_Trans_Act = PU_Trans_Act.replace(0, 1)
    PU_C = PU_C.replace(0, 1)
    PU_W = PU_W.replace(0, 1)

    LL_AD = (df['CHOICE'] == CH['AD']) * np.log(PU_AD)
    LL_CP = ((df['CHOICE'] == CH['AC']) | (df['CHOICE'] == CH['CPR'])) * np.log(PU_CP)
    LL_PR = (df['CHOICE'] == CH['PR']) * np.log(PU_PR)
    LL_Trans_Act = ((df['CHOICE'] == CH['PT']) | (df['CHOICE'] == CH['CR'])) * np.log(PU_Trans_Act)
    LL_C = (df['CHOICE'] == CH['C']) * np.log(PU_C)
    LL_W = (df['CHOICE'] == CH['W']) * np.log(PU_W)

    # Add the upper nest values to the lower nest values in the loglikelihood
    llfun = llfun + LL_AD.sum() + LL_CP.sum() + LL_PR.sum() + LL_Trans_Act.sum() + LL_C.sum() + LL_W.sum()
    return -1 * llfun

res = minimize(nest_logit, params)

params = res.x
sd = res.hess_inv.diagonal()
tstat = params/sd

data = np.column_stack((params,sd,tstat))
res_df = pd.DataFrame(data, beta_dict.keys(), ['value', 'std err.', 't-stat'])
res_df['loglike'] = -1 * res.fun
res_df = res_df.round(4)

print(res_df)
res_df.to_csv('./nest_trans_4_5.csv')
