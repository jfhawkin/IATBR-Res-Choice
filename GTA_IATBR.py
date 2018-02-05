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
import patsy
from bid_choice import BCLogit as bc



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
4. house - indicates the unit is a house x HH has more than 2 members
5. high_inc - indicates the househould is high income (quintile 4/5) and zone has high average income
6 low_inc  - indicates the household is high income (quintile 4/5) and zone has low average income
7. access - Accessibility measure from mode choice
8. ind - area of land devoted to industry
9. com - area of land devoted to commercial
10. gov - area of land devoted to government
11. park - area of land devoted to parks
12. res - area of land devoted to residential
13. num_job - number of jobs in zone
14. num_school - number of schools in zone
15. surface - surface area of dwelling m2 x log(HH size)
16. price - distinguishes between if hh rents/owns and house/townhouse/apartment
17. a - a price-specific constant
18. gamma - a price-specific constant to transform price utility into dollars
19. sigma - a price-specific constant
"""

data = pd.read_csv('../data/testData.csv',',')

#Income quintile thresholds in 2011 for Canada
incomeQuins = [28900,51700,101100,129400]

endog = data['taz']
exog = pd.DataFrame()

exog['ASC2'] = ((data['hh_income']<incomeQuins[1]) & (data['hh_income']>=incomeQuins[0])).astype(int)
exog['ASC3'] = ((data['hh_income']<incomeQuins[2]) & (data['hh_income']>=incomeQuins[1])).astype(int)
exog['ASC4'] = ((data['hh_income']<incomeQuins[3]) & (data['hh_income']>=incomeQuins[2])).astype(int)
exog['ASC5'] = (data['hh_income']>incomeQuins[3]).astype(int)
exog['house'] = data['house']*(data['hh_size']>2).astype(int)
exog['high_inc'] = ((data['hh_income']>incomeQuins[2]) & (data['taz_income']>incomeQuins[2])).astype(int)
exog['low_inc'] = ((data['hh_income']>incomeQuins[2]) & (data['taz_income']<incomeQuins[1])).astype(int)
exog['ind'] = data['zone_ind']
exog['com'] = data['zone_com']
exog['gov'] = data['zone_gov']
exog['park'] = data['zone_park']
exog['res'] = data['zone_res']
exog['num_jobs'] = data['taz_job']
exog['num_schools'] = data['taz_school']
exog['a'] = np.ones(endog.shape[0])
exog['gamma'] = np.ones(endog.shape[0])
exog['sigma'] = np.ones(endog.shape[0])

tempExog = np.zeros((endog.shape[0],2))
#Loop through rows of data and create an array with data as we go along. 
#More efficient than series slices.
for i, row in data.iterrows():
    if row['house']*row['own']==1:
        tempExog[i][0] = row['areaH']*np.log(row['hh_size'])
        tempExog[i][1] = row['taz_priceH']
    if row['house']*row['rent']==1:
        tempExog[i][0] = row['areaHR']*np.log(row['hh_size'])
        tempExog[i][1]= row['taz_priceHR']
    if row['town']*row['own']==1:
        tempExog[i][0] = row['areaT']*np.log(row['hh_size'])
        tempExog[i][1] = row['taz_priceT']
    if row['town']*row['rent']==1:
        tempExog[i][0] = row['areaTR']*np.log(row['hh_size'])
        tempExog[i][1] = row['taz_priceTR']
    if row['condo']*row['own']==1:
        tempExog[i][0] = row['areaC']*np.log(row['hh_size'])
        tempExog[i][1] = row['taz_priceC']
    if row['condo']*row['rent']==1:
        tempExog[i][0] = row['areaCR']*np.log(row['hh_size'])
        tempExog[i][1] = row['taz_priceCR']

#Now add the two columns to the exog dataframe
exog['surface'] = tempExog[:,0]
exog['price'] = tempExog[:,1]

print(exog.head())
print(exog.tail())
