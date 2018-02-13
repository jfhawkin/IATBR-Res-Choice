# -*- coding: utf-8 -*-

"""
Created on Thu Jan 18 16:14:39 2018

Main file for testing of bid-choice model framework

@author: jason
"""

import pandas as pd
#import shapefile #the pyshp module
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import patsy
import bid_choice as bc



#read file, parse out the records and shapes
#shapefile_path = r'tts06_83_region_Demo_Join.shp'
#sf = shapefile.Reader(shapefile_path)

#grab the shapefile's field names (omit the first psuedo field)
#fields = [x[0] for x in sf.fields][1:]
#records = sf.records()
#shps = [s.points for s in sf.shapes()]

#write the records into a dataframe
#zData = pd.DataFrame(columns=fields, data=records)
#grab the hh data from CHOICE dataset for 2-worker households

"""
Exogeneous variables in the model are:
ASC1 - fixed at 0
0. ASC2 - income constant for quintile 2
1. ASC3 - income constant for quintile 3
2. ASC4 - income constant for quintile 4
3. ASC5 - income constant for quintile 5
4. house - indicates the zone structure type is house x HH has more than 2 members
5. high_inc - indicates the househould is high income (quintile 4/5) and zone has high average income
6 low_inc  - indicates the household is high income (quintile 4/5) and zone has low average income
7. access - Accessibility measure from mode choice
8. ind - area of land devoted to industry
9. com - area of land devoted to commercial
10. gov - area of land devoted to government
11. park - area of land devoted to parks
12. res - area of land devoted to residential
13. num_job - number of jobs in zone in x100 jobs
14. num_school - number of schools in zone
15. surface - surface area of dwelling in x100 m2 x log(HH size)
16. obs price - distinguishes between if hh rents/owns and house/townhouse/apartment in $/100 m2 over 25 years (parameter is gamma in log likelihood)
17. a - a price-specific constant
18. sigma - a price-specific constant
"""

dfTAZ = pd.read_csv('../data/testDataTAZ.csv',',')
dfHH = pd.read_csv('../data/testDataHH.csv',',')
HH_COLS = dfHH.shape[1]
TAZ_COLS = dfTAZ.shape[1]
EXOG_COLS = 18 #CHANGE ME FOR EACH MODEL SPECIFICATION
ALTS = len(dfTAZ.groupby(['struc_type','taz_region'])) #CHANGE ME FOR EACH MODEL SPECIFICATION

hh_taz_alts = np.zeros((len(dfHH)*ALTS,(HH_COLS+TAZ_COLS)))

for i, row in dfHH.iterrows():
    if i==0:
        hh_taz_alts[i:(i+ALTS),0:HH_COLS] = np.repeat(row.as_matrix()[np.newaxis,:],ALTS,axis=0)
        hh_taz_alts[i,HH_COLS:] = dfTAZ[dfTAZ['taz_struc']==row['taz_struc']].as_matrix()
        
        #remove the group for the chosen alternative and create random alternatives from the other groups
        tempdfTAZ = dfTAZ[(dfTAZ['struc_type']!=hh_taz_alts[i,HH_COLS+2]) | (dfTAZ['taz_region']!=hh_taz_alts[i,HH_COLS+3])]
        hh_taz_alts[(i+1):(i+ALTS),HH_COLS:] = tempdfTAZ.groupby(['struc_type','taz_region']).apply(lambda x: x.sample(1)).reset_index(drop=True).as_matrix()
    else:
        hh_taz_alts[i*ALTS:i*ALTS+ALTS,0:HH_COLS] = np.repeat(row.as_matrix()[np.newaxis,:],ALTS,axis=0)
        hh_taz_alts[i*ALTS,HH_COLS:] = dfTAZ[dfTAZ['taz_struc']==row['taz_struc']].as_matrix()
        
        #remove the group for the chosen alternative and create random alternatives from the other groups
        tempdfTAZ = dfTAZ[(dfTAZ['struc_type']!=hh_taz_alts[i*ALTS,HH_COLS+2]) | (dfTAZ['taz_region']!=hh_taz_alts[i*ALTS,HH_COLS+3])]
        hh_taz_alts[i*ALTS+1:i*ALTS+ALTS,HH_COLS:] = tempdfTAZ.groupby(['struc_type','taz_region']).apply(lambda x: x.sample(1)).reset_index(drop=True).as_matrix()

##Income quintile thresholds in 2014 for Canada
incomeQuins = [28900,51700,101100,129400]

endog_chosen = dfHH['taz_struc'].as_matrix()
endog = hh_taz_alts[:,HH_COLS].tolist()
#Repeat the endog_chosen array for each alternative
endog_chosen = np.repeat(endog_chosen,ALTS)
#Get the number of columns specific to each zone for each alternative
exog = np.zeros((len(endog),EXOG_COLS))

#define a vector of chosen/unchosen
chosen = (endog==endog_chosen).astype(int)

for i in range(hh_taz_alts.shape[0]):
    #ASC2
    exog[i,0] = ((hh_taz_alts[i,2]<incomeQuins[1]) & (hh_taz_alts[i,2]>=incomeQuins[0])).astype(int)
    #ASC3
    exog[i,1] = ((hh_taz_alts[i,2]<incomeQuins[2]) & (hh_taz_alts[i,2]>=incomeQuins[1])).astype(int)
    #ASC4
    exog[i,2] = ((hh_taz_alts[i,2]<incomeQuins[3]) & (hh_taz_alts[i,2]>=incomeQuins[2])).astype(int)
    #ASC5
    exog[i,3] = (hh_taz_alts[i,2]>incomeQuins[3]).astype(int)
    #house
    exog[i,4] = ((hh_taz_alts[i,HH_COLS+2]==1)*(hh_taz_alts[i,3]>2)).astype(int)
    #high income
    exog[i,5] = ((hh_taz_alts[i,2]>incomeQuins[2]) & (hh_taz_alts[i,HH_COLS+4]>incomeQuins[2])).astype(int)
    #low income
    exog[i,6] = ((hh_taz_alts[i,2]>incomeQuins[2]) & (hh_taz_alts[i,HH_COLS+4]>incomeQuins[2])).astype(int)
    #industrial land
    exog[i,7] = hh_taz_alts[i,HH_COLS+5]
    #commercial land
    exog[i,8] = hh_taz_alts[i,HH_COLS+6]
    #government land
    exog[i,9] = hh_taz_alts[i,HH_COLS+7]
    #park land
    exog[i,10] = hh_taz_alts[i,HH_COLS+8]
    #residential land
    exog[i,11] = hh_taz_alts[i,HH_COLS+9]
    #number of jobs
    exog[i,12] = hh_taz_alts[i,HH_COLS+10]/10**2
    #number of schools
    exog[i,13] = hh_taz_alts[i,HH_COLS+11]
    #surface area of dwellings in TAZ
    exog[i,14] = (np.log(hh_taz_alts[i,3])*hh_taz_alts[i,7]*hh_taz_alts[i,HH_COLS+12]+np.log(hh_taz_alts[i,3])*(1-hh_taz_alts[i,7])*hh_taz_alts[i,HH_COLS+13])/10**2
    #price of dwellings in TAZ
    exog[i,15] = (hh_taz_alts[i,7]*hh_taz_alts[i,HH_COLS+14]+(1-hh_taz_alts[i,7])*hh_taz_alts[i,HH_COLS+15])/10**2

#vector of ones for a and sigma
exog[:,-2:] = np.ones((hh_taz_alts.shape[0],2))

params = np.array([-3.53,-3.71,-4.14,-1.41,-1.37,0.226,0.568,5.56,-4.67,0.1,0.1,0.1,0.1,0.1,0.002,0.446,1.04,-1.89])
bc_mod = bc.BCLogit(endog,exog,EXOG_COLS,chosen,ALTS)
loglike = bc_mod.loglike


Nfeval = 1

def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], loglike(Xi)))
    Nfeval += 1
    
print('Beginning optimization...')
print('{0:5s}   {1: 5s}   {2: 5s}   {3: 5s}   {4: 10s}'.format('Iter', 'X0', 'X1', 'X2', 'loglike'))

mod = minimize(loglike, params, method='BFGS', tol=1e-3, jac=False, callback=callbackF, options={'maxiter':500, 'disp': True})
print(mod.x)