from scipy.stats import t, pareto, rv_continuous
from scipy.stats import bernoulli, chi2, norm
import numpy as np
import pandas as pd

class double_power_law(rv_continuous): 
    "double power law"
    def __init__(self, gamma, rho, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma= gamma
        self.rho = rho
        
    def _pdf(self, x):
        if x > 1:
            return 0.5*(x**(-1/self.gamma - 1)/self.gamma + (1/self.gamma - self.rho)*x**(-1/self.gamma+self.rho-1))
        else: 
            return 0
    def _cdf(self, x):
        if x >= 1:
            return 1 - (x**(-1/self.gamma) + x**(-1/self.gamma + self.rho))/2
        else: 
            return 0
    def _stats(self,gamma, rho):
        return [0,0,0,0]
    
class ma1_rvs():
    def __init__(self):
        pass
    def rvs(gamma, size):
        student_values = t.rvs(gamma, size = size)
        return student_values[:-1] + student_values[1:]
    
class stoch_vol():
    def __init__(self):
        pass
    def rvs(gamma, size):
        U = bernoulli.rvs(0.5, size = size)
        U[U == 0] = -1

        Z = chi2.rvs(3, size = size)
        X = np.sqrt(57/Z)

        Q = norm.rvs(size = size)

        H = [Q[0]]
        for i in range(1, size):
            H.append(0.1*Q[i] + 0.9*H[i-1])

        Y = U*X*H
        return Y
    
class SnP500():
    def __init_(self):
        pass
    def rvs(gamma, size):
        data = pd.read_csv("SP500.csv")["Close/Last"]
        return np.array(data[1:size+1])/np.array(data[:size])
        