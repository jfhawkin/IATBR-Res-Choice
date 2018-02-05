"""
Created on Sat Feb 03 @ 09:11 2018

@author: jason

This file contains a variety of helper functions for bid-choice and discrete choice models. 
It will be added to as generic functions become necessary.

"""
# Helper functions for converting endogenous categories into a dummy matrix
def np_to_dummies(endog):
    if endog.dtype.kind in ['S', 'O']:
        endog_dummies, ynames = tools.categorical(endog, drop=True,
                                                  dictnames=True)
    elif endog.ndim == 2:
        endog_dummies = endog
        ynames = range(endog.shape[1])
    else:
        endog_dummies, ynames = tools.categorical(endog, drop=True,
                                                  dictnames=True)
    return endog_dummies, ynames


def pd_to_dummies(endog):
    if endog.ndim == 2:
        if endog.shape[1] == 1:
            yname = endog.columns[0]
            endog_dummies = get_dummies(endog.iloc[:, 0])
        else:  # series
            yname = 'y'
            endog_dummies = endog
    else:
        yname = endog.name
        endog_dummies = get_dummies(endog)
    ynames = endog_dummies.columns.tolist()

    return endog_dummies, ynames, yname
