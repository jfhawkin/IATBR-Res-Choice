"""
Created on Sat Feb 03 @ 09:11 2018

@author: jason

Residential location choice parameter estimation - GTA Model for IATBR.
Based on bid-choice proposed by Martinez (1996) and expanded by Hurtubia (2012).
Begins with calculation of a location choice decision and latent price functions.
Latent price function of hedonic component and WTP component.

"""
import numpy as np
import pandas as pd

class BCLogit(object):
    __doc__ = """
    Generic logit-type model of bid-choice decision making according to Martinez paradigm

    Parameters
    ----------
    endog : array-like
        `endog` is a 1-d vector of the endogenous response.  `endog` can
        contain strings, ints, or floats.  Note that if it contains strings,
        every distinct string will be a category.  No stripping of whitespace
        is done.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See `statsmodels.tools.add_constant`.
    price : array-like
          'price' is a 1-d vector of the prices in each of the k zones
    %(extra_params)s

    Attributes
    ----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    J : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one.

    Notes
    -----
    See developer notes for further information on `BCLogit` internals.
    """

    def __init__(self,endog,exog,exog_var,chosen,alts,**kwds):
        self.endog = endog
        self.exog = exog
        self.ch = chosen
        self.J = alts
        self.K = exog_var

#    def nloglikeobs(self,params):
#        ll = loglike(self,params)
#        return -ll

    def locFunc(self, X):
        """
        Multinomial logit cumulative distribution function.

        Parameters
        ----------
        X : array
            The linear predictor of the model XB.

        Returns
        --------
        cdf : ndarray
            The cdf evaluated at `X`.

        Notes
        -----
        In the multinomial logit model.
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        eXB = np.exp(X)
        eXB = eXB.reshape(-1,self.J)
        return eXB/eXB.sum(1)[:,None]
    
    def pricefunc(self, X, pparams):
        """
        Multinomial logit cumulative distribution function.

        Parameters
        ----------
        X : array
            The linear predictor of the model XB.

        Returns
        --------
        cdf : ndarray
            The cdf evaluated at `X`.

        Notes
        -----
        In the multinomial logit model.
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        
        """
        #Reference the utility values for rents to the endog list and take logsums over these indices
        eXB = pd.DataFrame(data=np.exp(X),index=self.endog)
        ri = np.log(eXB.groupby(eXB.index).sum())
        #Reference the observed prices to the same set of endog and take the average between own/rent in this case
        Ri = pd.DataFrame(data=self.exog[:,15],index=self.endog)
        Ri = Ri.groupby(Ri.index).mean()
        pF = np.exp((-Ri-pparams[1]-pparams[0]*ri)/(2*pparams[2]**2))
        pF = pF/np.sqrt(2*np.pi*pparams[2]**2)
        if pF.values.sum()==0:
            pF = pd.DataFrame(data=np.ones((pF.shape[0],pF.shape[1])),index=self.endog)
        
        pF_hh = np.zeros(len(self.endog))
        for i,v in enumerate(self.endog):
            pF_hh[i] = pF.loc[pF.index==v].values
            
        pF_hh = pF_hh.reshape(-1,self.J)   
        
        return pF_hh

    def loglike(self, params):
        """
        Log-likelihood of the multinomial logit model.

        Parameters
        ----------
        params : array-like
            The parameters of the multinomial logit model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i=1}^{n}\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        d = self.ch.reshape(-1, self.J)
        locComp = self.locFunc(np.dot(self.exog[:,:-3],params[:-3]))
        priceComp = self.pricefunc(np.dot(self.exog[:,:-3],params[:-3]),params[-3:])
        logProb = np.log(locComp*priceComp)
        return -1*np.sum(d * logProb)

    def score(self, params):
        """
        NotImplemented
        """
        raise NotImplementedError
    #    """
    #    Score matrix for multinomial logit model log-likelihood

    #    Parameters
    #    ----------
    #    params : array
    #        The parameters of the multinomial logit model.

    #    Returns
    #    --------
    #    score : ndarray, (K * (J-1),)
    #        The 2-d score vector, i.e. the first derivative of the
    #        loglikelihood function, of the multinomial logit model evaluated at
    #        `params`.

    #    Notes
    #    -----
    #    .. math:: \\frac{\\partial\\ln L}{\\partial\\beta_{j}}=\\sum_{i}\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

    #    for :math:`j=1,...,J`

    #    In the multinomial model the score matrix is K x J-1 but is returned
    #    as a flattened array to work with the solvers.
    #    """
    #    params = params.reshape(self.K, -1, order='F')
    #    firstterm = self.wendog[:,1:] - self.cdf(np.dot(self.exog,
    #                                              params))[:,1:]
    #    #NOTE: might need to switch terms if params is reshaped
    #    return np.dot(firstterm.T, self.exog).flatten()