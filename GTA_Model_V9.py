# -*- coding: utf-8 -*-

"""
Created on Tues May 29

Main file for testing of bid-choice model framework. I combined previous conversion
with nested logit model from CIV1538 A03.

Model 9: Updated 8 with findings in Models 1-7

"ValueError: cannot set a frame with no defined index and a scalar" ?? CHECK INPUTS FOR BLANK LINES

@author: jason
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import collections
from scipy.stats import norm

MAKE_DATA = 0 # CHANGE AFTER INITIAL MODEL RUN

def drop_y(df):
    # list comprehension of the cols that end with '_y'
    to_drop = [x for x in df if x.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)

if MAKE_DATA == 1:
    dfTAZ = pd.read_csv('/home/jason/Documents/Conference Submissions/IATBR2018/data/testDataTAZ.csv', ',')
    dfHH = pd.read_csv('/home/jason/Documents/Conference Submissions/IATBR2018/data/testDataHH.csv', ',')
    dfAIVTT = pd.read_csv('/home/jason/Documents/Conference Submissions/IATBR2018/data/aivtt.csv', ',', index_col='pq')
    dfTIVTT = pd.read_csv('/home/jason/Documents/Conference Submissions/IATBR2018/data/tivtt.csv', ',', index_col='pq')
    ALTS = 30  # CHANGE ME FOR EACH MODEL SPECIFICATION

    # Iterate over rows to generate sample set with 9 random samples. Need a UID for each situation.
    # I just use the first row id for each HH.
    for i, row in enumerate(dfHH.itertuples()):
        # Define the results DataFrame
        if row.memb_id == 0:
            UID = i
            if i == 0:
                dfResult = dfHH[dfHH.hh_mem_id == row.hh_mem_id]
                dfResult.loc[:, 'chosen'] = 1
                dfResult.loc[:, 'UID'] = UID
            else:
                dfRow = dfHH[dfHH.hh_mem_id == row.hh_mem_id]
                dfRow.loc[:, 'chosen'] = 1
                dfRow.loc[:, 'UID'] = UID
                dfResult = dfResult.append(dfRow)

            # Remove records where the TAZ is the row respondent TAZ
            dfSample = dfHH[dfHH.taz_struc != row.taz_struc]
            # Sample ALTS-1 records from remaining records
            dfSample = dfSample.sample(n=(ALTS-1))
            # Need the additional rows for each sample hh for the other hh members
            for samRow in dfSample.itertuples():
                dfSubSample = dfHH[(dfHH.hh_id == samRow.hh_id) & (dfHH.hh_mem_id != samRow.hh_mem_id)]
                dfSample = dfSample.append(dfSubSample)
            dfSample.loc[:, 'taz_struc'] = row.taz_struc
            dfSample.loc[:, 'taz'] = row.taz
            dfSample.loc[:, 'chosen'] = 0
            dfSample.loc[:, 'UID'] = UID
            # Append the results together
            dfResult = dfResult.append(dfSample)
        else:
            dfRow = dfHH[dfHH.hh_mem_id == row.hh_mem_id]
            dfRow.loc[:, 'chosen'] = 1
            dfRow.loc[:, 'UID'] = UID
            dfResult = dfResult.append(dfRow)

    dfHH_TAZ = dfResult.merge(dfTAZ, how='left', on=['taz_struc'], suffixes=('', '_y'))
    drop_y(dfHH_TAZ)
    # Super kludgy auto IVTT method!
    df1 = dfAIVTT.loc[dfHH_TAZ.taz]
    df2 = df1.transpose()
    df3 = df2.loc[dfHH_TAZ.wtaz.astype(str)]
    dfHH_TAZ['aivtt'] = np.diag(df3.values)
    # Same thing for transit IVTT
    df4 = dfTIVTT.loc[dfHH_TAZ.taz]
    df5 = df4.transpose()
    df6 = df5.loc[dfHH_TAZ.wtaz.astype(str)]
    dfHH_TAZ['tivtt'] = np.diag(df6.values)
    dfHH_TAZ.to_csv('/home/jason/Documents/Conference Submissions/IATBR2018/data/HH_TAZ.csv')

else:
    dfHH_TAZ = pd.read_csv('/home/jason/Documents/Conference Submissions/IATBR2018/data/HH_TAZ.csv', ',')

# Income quintile thresholds in 2014 for Canada
incomeQuins = [28900, 51700, 101100, 129400]

# Adjust variables to scale parameters and create derived variables
# # ASC2
dfHH_TAZ['ASC2'] = ((dfHH_TAZ['hh_income'] < incomeQuins[1]) & (dfHH_TAZ['hh_income'] >= incomeQuins[0])).astype(int)
# # ASC3
dfHH_TAZ['ASC3'] = ((dfHH_TAZ['hh_income'] < incomeQuins[2]) & (dfHH_TAZ['hh_income'] >= incomeQuins[1])).astype(int)
# # ASC4
dfHH_TAZ['ASC4'] = ((dfHH_TAZ['hh_income'] < incomeQuins[3]) & (dfHH_TAZ['hh_income'] >= incomeQuins[2])).astype(int)
# # ASC5
dfHH_TAZ['ASC5'] = (dfHH_TAZ['hh_income'] > incomeQuins[3]).astype(int)
# house and more than 2 HH members
dfHH_TAZ['house_hh2+'] = ((dfHH_TAZ['struc_type'] == 1) * (dfHH_TAZ['hh_size'] > 2)).astype(int)
# high income HH and high income TAZ
dfHH_TAZ['h_h_inc'] = ((dfHH_TAZ['hh_income'] > incomeQuins[2]) & (dfHH_TAZ['taz_income'] > incomeQuins[2])).astype(int)
# low income HH and high income TAZ
dfHH_TAZ['l_h_inc'] = ((dfHH_TAZ['hh_income'] < incomeQuins[1]) & (dfHH_TAZ['taz_income'] > incomeQuins[2])).astype(int)
# surface area of dwellings in TAZ (100s m2) multiplied by HH size
dfHH_TAZ['area'] = np.log(dfHH_TAZ['hh_size']) * (dfHH_TAZ['own'] * dfHH_TAZ['areaO'] + (1 - dfHH_TAZ['own']) * dfHH_TAZ['areaR']) / 10 ** 2
# price of dwellings in TAZ (for rent or own depending on HH)
dfHH_TAZ['price'] = (dfHH_TAZ['own'] * dfHH_TAZ['taz_priceO'] + (1 - dfHH_TAZ['own']) * dfHH_TAZ[
    'taz_priceO']) / 10 ** 2
# hundreds of jobs in zone interacted with number of workers
dfHH_TAZ['taz_job'] = (dfHH_TAZ['taz_job'] * np.log(dfHH_TAZ['hh_count'])) / 10**3
# tens of dollars parking cost
dfHH_TAZ['work_park_cost'] = dfHH_TAZ['work_park_cost'] / 10
dfHH_TAZ['taz_school'] = np.log(dfHH_TAZ['taz_school']) * (dfHH_TAZ['children'] > 0).astype(int)
# Number of childcare interacted with number of children
# dfHH_TAZ['taz_childcare'] = np.log(dfHH_TAZ['taz_childcare']) * dfHH_TAZ['children']
dfHH_TAZ['taz_childcare'] = np.log(dfHH_TAZ['taz_childcare']) * (dfHH_TAZ['children'] > 0).astype(int)
# Portion of taz zoned park x log of number children
# dfHH_TAZ['zone_park'] = dfHH_TAZ['zone_park'] * np.log(dfHH_TAZ['travel_walk_bike'])
# Transit stations and whether individual has transit pass
dfHH_TAZ['go_transit_pass'] = dfHH_TAZ['transit_pass'] * dfHH_TAZ['Ct_GO']
dfHH_TAZ['ttc_transit_pass'] = dfHH_TAZ['transit_pass'] * dfHH_TAZ['Ct_TTC']
# Interact own with HH income
#dfHH_TAZ['delta_area'] = dfHH_TAZ['delta_area'] * dfHH_TAZ['taz_school']

# Log of zero will give nan error, so add the zeros back to avoid model errors
dfHH_TAZ = dfHH_TAZ.fillna(value=0)
beta_dict = collections.OrderedDict([('ASC', -0.0105), ('HOUSE_HH2+', 0.2351),
                                    ('H_H_INC', 0.7359), ('L_H_INC', -0.7359), ('HW_DIST_INC', -2.9908), ('AREA', 0.1682),
                                    ('HW_AUTO_TIME', -0.0226), ('PARK_COST', 0.0015), ('TTC', 0), ('FREQ_DRIVE', 1.6823),
                                    ('HW_SAME', 0.01), ('VEH_LIC', 0), ('DELTA_AREA', 0), ('W_R1F', -1.1373),
                                    ('W_R1NF', -1.1373), ('W_R2F', -0.7219), ('W_R2NF', -0.7219), ('W_R3', -0.0253), ('W_R4', -0.0253),
                                     ('GAMMA', 0.1422), ('ALPHA', 10.0436), ('SIGMA',  0.2631)])

params = beta_dict.values()

def pcl_indiv_util(params):
    global dfHH_TAZ

    # Update the dictionary of betas to reflect the new param estimates
    for i, k in enumerate(beta_dict):
        beta_dict[k] = params[i]

    # Define utility functions for each model (weighted)
    V = beta_dict['HW_AUTO_TIME'] * dfHH_TAZ['aivtt'] + beta_dict['PARK_COST'] * dfHH_TAZ['work_park_cost'] \
        + beta_dict['FREQ_DRIVE'] * dfHH_TAZ['freq_drive'] + beta_dict['TTC'] * dfHH_TAZ['go_transit_pass'] \
        + beta_dict['TTC'] * dfHH_TAZ['ttc_transit_pass'] + beta_dict['HW_SAME'] * dfHH_TAZ['hw_region'] \
        + beta_dict['HW_DIST_INC'] * dfHH_TAZ['hw_dist_inc']

    dummyHHRole = pd.get_dummies(dfHH_TAZ['hh_role'])
    dummyW = beta_dict['W_R1F'] * dummyHHRole[1] * (dfHH_TAZ['children'] > 0).astype(int) + beta_dict['W_R1NF'] * dummyHHRole[1] * (dfHH_TAZ['children'] == 0).astype(int) \
             + beta_dict['W_R2F'] * dummyHHRole[2] * (dfHH_TAZ['children'] > 0).astype(int) + beta_dict['W_R2NF'] * dummyHHRole[2] * (dfHH_TAZ['children'] == 0).astype(int) \
             + beta_dict['W_R3'] * dummyHHRole[3] + beta_dict['W_R4'] * dummyHHRole[4]
    dummyW = np.exp(dummyW)

    dfHH_TAZ.loc[:, 'W'] = dummyW
    dfTemp = dfHH_TAZ.groupby(['hh_id', 'UID'])['W'].sum()
    dfTemp = dfTemp.to_frame()
    dfTemp.columns = ['W_sum']
    if 'W_sum' in dfHH_TAZ.columns:
        dfHH_TAZ = dfHH_TAZ.drop(columns=['W_sum'])
    dfHH_TAZ = dfHH_TAZ.merge(dfTemp, how='right', left_on=['hh_id', 'UID'], right_index=True, suffixes=('', '_y'))
    drop_y(dfHH_TAZ)
    dfHH_TAZ = dfHH_TAZ.sort_index(axis=0)

    W = dfHH_TAZ['W'] / dfHH_TAZ['W_sum']
    #W = 1 / dfHH_TAZ['hh_count']

    V = V * W

    return V

def pcl_group_util(params):
    global dfHH_TAZ
    # Update the dictionary of betas to reflect the new param estimates
    for i, k in enumerate(beta_dict):
        beta_dict[k] = params[i]
    # Define utility functions for each model. Total jobs blows this up! Maybe log jobs...
    V = beta_dict['HOUSE_HH2+'] * dfHH_TAZ['house_hh2+'] + beta_dict['H_H_INC'] * dfHH_TAZ['h_h_inc']  \
        + beta_dict['AREA'] * dfHH_TAZ['area'] \
        + beta_dict['L_H_INC'] * dfHH_TAZ['l_h_inc'] + beta_dict['VEH_LIC'] * dfHH_TAZ['hh_veh_per_licensed_com'] \
        + beta_dict['DELTA_AREA'] * dfHH_TAZ['delta_area'] + beta_dict['ASC'] * dfHH_TAZ['own']

    return V

def pcl_probs(params):
    global dfHH_TAZ
    # Get the utility functions
    V_i = pcl_indiv_util(params)
    V_g = pcl_group_util(params)

    # Update the dictionary of betas to reflect the new param estimates
    for i, k in enumerate(beta_dict):
        beta_dict[k] = params[i]

    dfHH_TAZ.loc[:, 'V_comb'] = V_g + V_i
    dfHH_TAZ.loc[:, 'P_N'] = np.exp(dfHH_TAZ['V_comb'])
    # Split utility between respondent rows
    dfHH_TAZ.loc[:, 'P_N'] = dfHH_TAZ['P_N'] / dfHH_TAZ['hh_count']
    dfTemp = dfHH_TAZ.groupby('UID')['P_N'].sum()
    dfTemp = dfTemp.to_frame()
    dfTemp.columns = ['P_N_sum']
    if 'P_N_sum' in dfHH_TAZ.columns:
        dfHH_TAZ = dfHH_TAZ.drop(columns=['P_N_sum'])

    dfHH_TAZ = dfHH_TAZ.merge(dfTemp, how='right', left_on='UID', right_index=True, suffixes=('', '_y'))
    drop_y(dfHH_TAZ)
    dfHH_TAZ = dfHH_TAZ.sort_index(axis=0)

    dfHH_TAZ.loc[:, 'P'] = dfHH_TAZ['P_N'] / dfHH_TAZ['P_N_sum']
    dfHH_TAZ.loc[:, 'P'] = dfHH_TAZ['P']
    return dfHH_TAZ


def br_vals(params):
    global dfHH_TAZ

    # Update the dictionary of betas to reflect the new param estimates
    for i, k in enumerate(beta_dict):
        beta_dict[k] = params[i]

    ri = dfHH_TAZ['P_N_sum']
    ri = np.log(ri)

    # Reference the observed prices to the same set of endog and take the average between own/rent in this case
    Ri = np.log((dfHH_TAZ['taz_priceO'] + dfHH_TAZ['taz_priceR']) / 2)
    x = Ri - beta_dict['ALPHA'] - beta_dict['GAMMA'] * ri
    #pF = np.exp(-(Ri - beta_dict['ALPHA'] - beta_dict['GAMMA'] * ri) / (2 * beta_dict['SIGMA'] ** 2))
    #pF = pF / np.sqrt(2 * np.pi * beta_dict['SIGMA'] ** 2)
    pF = norm.pdf(x, scale=beta_dict['SIGMA'])

    return pF

def bc_mod(params):
    # Get probabilies for location choice component of model
    LOC_PROB = pcl_probs(params)

    # Get bid-rents for price component of model
    BID_RENT = br_vals(params)

    # Assemble log likelihood function
    LP_BR = np.log(LOC_PROB['P']*BID_RENT)
    #LP_BR = np.log(LOC_PROB['P'])
    llfun = LP_BR*LOC_PROB['chosen']
    llfun = llfun.sum()
    return -1 * llfun

res = minimize(bc_mod, params)
params = res.x

var = res.hess_inv.diagonal()
sd = np.sqrt(var)
tstat = params/sd

data = np.column_stack((params, sd, tstat))
res_df = pd.DataFrame(data, beta_dict.keys(), ['value', 'std err.', 't-stat'])
res_df['loglike'] = -1 * res.fun
res_df = res_df.round(4)

print(res_df)
cov_mat = pd.DataFrame(res.hess_inv)
cov_mat.to_csv('cov_mat_9.csv')
sd = sd[:, np.newaxis]
sdt = np.transpose(sd)
corr_mat = pd.DataFrame(res.hess_inv / np.multiply(sd,sdt))
corr_mat.to_csv('corr_mat_9.csv')

#print(res.hess_inv)
# res_df.to_latex('./params.txt')
#
# # # Calculate rho2-const
# print(res.fun)
# print(res_const.fun)
# rho_const = 1 - (res.fun/res_const.fun)
# print(rho_const)
