import pandas as pd

from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller

def fit_ols(y, x):
    '''
    Estimates long-run and short-run cointegration relationship for series y and x.
    2-step process:
    1) estimate coefficients for the long-run relationship y_t=c+gamma*x_t+z_t
    2) estimate coefficients for the short-run relationship y_t-y_{t-1}=alpha*z_{t-1}+epsilon_t
    
    Params:
    @ y: pd.Series, the first time series of the pair to analyze
    @ x: pd.Series, the second time series of the pair to analyze
    Return:
    @ c: float, constant in the long-run relationship, describing the static shift of y wrt gamma*x
    @ gamma: float,
    @ alpha: float,
    @ z: pd.Series
    '''

    assert isinstance(y, pd.Series), 'Input series y should be of type pd.Series'
    assert isinstance(x, pd.Series), 'Input series x should be of type pd.Series'
    assert sum(y.isnull()) == 0
    assert sum(x.isnull()) == 0
    assert y.index.equals(x.index)

    long_run_ols = OLS(y, add_constant(x), has_const=True)
    long_run_ols_fit = long_run_ols.fit()