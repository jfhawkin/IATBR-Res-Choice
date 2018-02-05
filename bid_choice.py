"""
Created on Sat Feb 03 @ 09:11 2018

@author: jason

Residential location choice parameter estimation - GTA Model for IATBR.
Based on bid-choice proposed by Martinez (1996) and expanded by Hurtubia (2012).
Begins with calculation of a location choice decision and latent price functions.
Latent price function of hedonic component and WTP component.

"""
import numpy as np
from statsmodels.base.data import handle_data
from pandas.core.api import get_dummies
from statsmodels.base.model import GenericLikelihoodModel
import tools

class BCLogit(GenericLikelihoodModel):
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
    names : dict
        A dictionary mapping the column number in `wendog` to the variables
        in `endog`.
    wendog : array
        An n x j array where j is the number of unique categories in `endog`.
        Each column of j is a dummy variable indicating the category of
        each observation. See `names` for a dictionary mapping each column to
        its category.

    Notes
    -----
    See developer notes for further information on `BCLogit` internals.
    """

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        if data_tools._is_using_ndarray_type(endog, None):
            endog_dummies, ynames = _numpy_to_dummies(endog)
            yname = 'y'
        elif data_tools._is_using_pandas(endog, None):
            endog_dummies, ynames, yname = tools.pd_to_dummies(endog)
        else:
            endog = np.asarray(endog)
            endog_dummies, ynames = tools.np_to_dummies(endog)
            yname = 'y'

        if not isinstance(ynames, dict):
            ynames = dict(zip(range(endog_dummies.shape[1]), ynames))

        self._ynames_map = ynames
        data = handle_data(endog_dummies, exog, missing, hasconst, **kwargs)
        data.ynames = yname  # overwrite this to single endog name
        data.orig_endog = endog
        self.wendog = data.endog

        # repeating from upstream...
        for key in kwargs:
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:
                pass
        return data

    def _init_(self,endog,exog,**kwds):
        super(BCLogit, self).__init__(endog,exog,**kwds)
        self.J = self.wendog.shape[1]
        self.K = self.exog.shape[1]
    
    def pdf(self, eXB):
        """
        NotImplemented
        """
        raise NotImplementedError

    def nloglikeobs(self,params):
        ll = loglike(self.endog,self.exog,params)
        return -ll

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
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
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
        
        Don't forget you changed the GenericLikelihood model to include a function for nloglikeobs()
        """
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
        ri = np.log(eXB.sum(0)[:,None])
        Ri = np.column_stack((self.endog,self.exog[:,-3]))
        Ri = np.unique(Ri,axis=0)[:-1,1]
        pF = np.exp(-Ri-pparams[1]-pparams[0]*ri/(2*pparams[2]**2))
        pF = pF/np.sqrt(2*np.pi*pparams[2]**2)
        if np.sum(pF)==0:
            pF = np.ones((pF.shape[0],pF.shape[1]))
        return np.column_stack((np.ones(pF.shape[0]),pF))

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
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        locComp = self.locFunc(np.dot(self.exog[:,:-3],params[:-3]))
        priceComp = self.pricefunc(np.dot(self.exog[:,:-3],params[:-3]),params[-3:])
        logprob = np.log(np.dot(locComp,priceComp))
        return np.sum(d * logprob)


    def loglikeobs(self, params):
        """
        NotImplemented
        """
        raise NotImplementedError
    #    """
    #    Log-likelihood of the multinomial logit model for each observation.

    #    Parameters
    #    ----------
    #    params : array-like
    #        The parameters of the multinomial logit model.

    #    Returns
    #    -------
    #    loglike : ndarray (nobs,)
    #        The log likelihood for each observation of the model evaluated
    #        at `params`. See Notes

    #    Notes
    #    ------
    #    .. math:: \\ln L_{i}=\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

    #    for observations :math:`i=1,...,n`

    #    where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
    #    if not.
    #    """
    #    params = params.reshape(self.K, -1, order='F')
    #    d = self.wendog
    #    logprob = np.log(self.cdf(np.dot(self.exog,params)))
    #    return d * logprob


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


    def loglike_and_score(self, params):
        """
        NotImplemented
        """
        raise NotImplementedError
    #    """
    #    Returns log likelihood and score, efficiently reusing calculations.

    #    Note that both of these returned quantities will need to be negated
    #    before being minimized by the maximum likelihood fitting machinery.

    #    """
    #    params = params.reshape(self.K, -1, order='F')
    #    cdf_dot_exog_params = self.cdf(np.dot(self.exog, params))
    #    loglike_value = np.sum(self.wendog * np.log(cdf_dot_exog_params))
    #    firstterm = self.wendog[:, 1:] - cdf_dot_exog_params[:, 1:]
    #    score_array = np.dot(firstterm.T, self.exog).flatten()
    #    return loglike_value, score_array


    def score_obs(self, params):
        """
        NotImplemented
        """
        raise NotImplementedError
    #    """
    #    Jacobian matrix for multinomial logit model log-likelihood

    #    Parameters
    #    ----------
    #    params : array
    #        The parameters of the multinomial logit model.

    #    Returns
    #    --------
    #    jac : ndarray, (nobs, k_vars*(J-1))
    #        The derivative of the loglikelihood for each observation evaluated
    #        at `params` .

    #    Notes
    #    -----
    #    .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta_{j}}=\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

    #    for :math:`j=1,...,J`, for observations :math:`i=1,...,n`

    #    In the multinomial model the score vector is K x (J-1) but is returned
    #    as a flattened array. The Jacobian has the observations in rows and
    #    the flatteded array of derivatives in columns.
    #    """
    #    params = params.reshape(self.K, -1, order='F')
    #    firstterm = self.wendog[:,1:] - self.cdf(np.dot(self.exog,
    #                                              params))[:,1:]
    #    #NOTE: might need to switch terms if params is reshaped
    #    return (firstterm[:,:,None] * self.exog[:,None,:]).reshape(self.exog.shape[0], -1)

    def hessian(self, params):
        """
        NotImplemented
        """
        raise NotImplementedError
    #    """
    #    Multinomial logit Hessian matrix of the log-likelihood

    #    Parameters
    #    -----------
    #    params : array-like
    #        The parameters of the model

    #    Returns
    #    -------
    #    hess : ndarray, (J*K, J*K)
    #        The Hessian, second derivative of loglikelihood function with
    #        respect to the flattened parameters, evaluated at `params`

    #    Notes
    #    -----
    #    .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta_{j}\\partial\\beta_{l}}=-\\sum_{i=1}^{n}\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\left[\\boldsymbol{1}\\left(j=l\\right)-\\frac{\\exp\\left(\\beta_{l}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right]x_{i}x_{l}^{\\prime}

    #    where
    #    :math:`\\boldsymbol{1}\\left(j=l\\right)` equals 1 if `j` = `l` and 0
    #    otherwise.

    #    The actual Hessian matrix has J**2 * K x K elements. Our Hessian
    #    is reshaped to be square (J*K, J*K) so that the solvers can use it.

    #    This implementation does not take advantage of the symmetry of
    #    the Hessian and could probably be refactored for speed.
    #    """
    #    params = params.reshape(self.K, -1, order='F')
    #    X = self.exog
    #    pr = self.cdf(np.dot(X,params))
    #    partials = []
    #    J = self.wendog.shape[1] - 1
    #    K = self.exog.shape[1]
    #    for i in range(J):
    #        for j in range(J): # this loop assumes we drop the first col.
    #            if i == j:
    #                partials.append(\
    #                    -np.dot(((pr[:,i+1]*(1-pr[:,j+1]))[:,None]*X).T,X))
    #            else:
    #                partials.append(-np.dot(((pr[:,i+1]*-pr[:,j+1])[:,None]*X).T,X))
    #    H = np.array(partials)
    #    # the developer's notes on multinomial should clear this math up
    #    H = np.transpose(H.reshape(J,J,K,K), (0,2,1,3)).reshape(J*K,J*K)
    #    return H

    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if start_params == None:
            start_params = np.repeat(0,len(self.exog.columns)) 
        return super(BCLogit, self).fit(start_params = start_params,maxiter = maxiter, maxfun = maxfun,**kwds)