import sys
import time
import argparse
import os
import warnings
import numpy as np
from matplotlib import pyplot as plt
from tail_estimation_expectiles import hill_dbs, moments_dbs, get_moments_estimates_1
import scipy.optimize as opt
from tqdm import tqdm
from scipy.stats import t, pareto, rv_continuous
from scipy.stats import bernoulli, chi2, norm
import sympy
import numpy as np
import os
import seaborn as sns
import pandas as pd
from distributions import double_power_law, ma1_rvs, stoch_vol, SnP500


class DBS_result:
    def __init__(self):
        self.k_arr = {}
        self.xi_arr= {}
        self.k_star = {}
        self.xi_star = {}
        self.x1_arr = {}
        self.n1_amse = {}
        self.k1 = {}
        self.max_index1 = {}
        self.x2_arr = {}
        self.n2_amse = {}
        self.k2 = {}
        self.max_index2 = {}
        self.rho = {}
        


# =========================================
# ========== Auxiliary Functions ==========
# =========================================

def add_uniform_noise(data_sequence, p = 1):
    """
    Function to add uniform random noise to a given dataset.
    Uniform noise in range [-5*10^(-p), 5*10^(-p)] is added to each
    data entry. For integer-valued sequences, p = 1.

    Args:
        data_sequence: numpy array of data to be processed.
        p: integer parameter controlling noise amplitude.

    Returns:
        numpy array with noise-added entries.
    """
    if p < 1:
        print("Parameter p should be greater or equal to 1.")
        return None
    noise = np.random.uniform(-5.*10**(-p), 5*10**(-p), size = len(data_sequence))
    randomized_data_sequence = data_sequence + noise
    # ensure there are no negative entries after noise is added
    randomized_data_sequence = randomized_data_sequence[np.where(randomized_data_sequence > 0)]
    return randomized_data_sequence


def get_distribution(data_sequence, number_of_bins = 30):
    """
    Function to get a log-binned distribution of a given dataset.

    Args:
        data_sequence: numpy array with data to calculate
                       log-binned PDF on.
        number_of_bins: number of logarithmic bins to use.

    Returns:
        x, y: numpy arrays containing midpoints of bins
              and corresponding PDF values.

    """
    # define the support of the distribution
    lower_bound = min(data_sequence)
    upper_bound = max(data_sequence)
    # define bin edges
    log = np.log10
    lower_bound = log(lower_bound) if lower_bound > 0 else -1
    upper_bound = log(upper_bound)
    bins = np.logspace(lower_bound, upper_bound, number_of_bins)
    
    # compute the histogram using numpy
    y, __ = np.histogram(data_sequence, bins = bins, density = True)
    # for each bin, compute its midpoint
    x = bins[1:] - np.diff(bins) / 2.0
    # if bin is empty, drop it from the resulting list
    drop_indices = [i for i,k in enumerate(y) if k == 0.0]
    x = [k for i,k in enumerate(x) if i not in drop_indices]
    y = [k for i,k in enumerate(y) if i not in drop_indices]
    return x, y

def get_ccdf(degree_sequence):
    """
    Function to get CCDF of the list of degrees.
    
    Args:
        degree_sequence: numpy array of nodes' degrees.

    Returns:
        uniques: unique degree values met in the sequence.
        1-CDF: CCDF values corresponding to the unique values
               from the 'uniques' array.
    """
    uniques, counts = np.unique(degree_sequence, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / (degree_sequence.size)
    return uniques[::-1], (1. - cumprob)[::-1]
    
# ================================================
# ========== Hill Tail Index Estimation ==========
# ================================================
def get_moments_estimates_1(ordered_data):
    """
    Function to calculate first moments array given an ordered data
    sequence. Decreasing ordering is required.

    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st moment (Hill estimator)
                      is calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.

    """
    logs_1 = np.log(ordered_data)
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    return M1

def get_moments_estimates_2(ordered_data):
    """
    Function to calculate first and second moments arrays
    given an ordered data sequence. 
    Decreasing ordering is required.

    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st (Hill estimator) and 2nd moments 
                      are calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.
        M2: numpy array of 2nd moments corresponding to all 
            possible order statistics of the dataset.

    """
    logs_1 = np.log(ordered_data)
    logs_2 = (np.log(ordered_data))**2
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    M2 = (1./k_vector)*logs_2_cumsum - (2.*logs_1[1:]/k_vector)*logs_1_cumsum\
         + logs_2[1:]
    return M1, M2

def get_moments_estimates_3(ordered_data):
    """
    Function to calculate first, second and third moments 
    arrays given an ordered data sequence. 
    Decreasing ordering is required.

    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st (Hill estimator), 2nd and 3rd moments 
                      are calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.
        M2: numpy array of 2nd moments corresponding to all 
            possible order statistics of the dataset.
        M3: numpy array of 3rd moments corresponding to all 
            possible order statistics of the dataset.

    """
    logs_1 = np.log(ordered_data)
    logs_2 = (np.log(ordered_data))**2
    logs_3 = (np.log(ordered_data))**3
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    logs_3_cumsum = np.cumsum(logs_3[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    M2 = (1./k_vector)*logs_2_cumsum - (2.*logs_1[1:]/k_vector)*logs_1_cumsum\
         + logs_2[1:]
    M3 = (1./k_vector)*logs_3_cumsum - (3.*logs_1[1:]/k_vector)*logs_2_cumsum\
         + (3.*logs_2[1:]/k_vector)*logs_1_cumsum - logs_3[1:]
    # cleaning exceptional cases
    clean_indices = np.where((M2 <= 0) | (M3 == 0) | (np.abs(1.-(M1**2)/M2) < 1e-10)\
                             |(np.abs(1.-(M1*M2)/M3) < 1e-10))
    M1[clean_indices] = np.nan
    M2[clean_indices] = np.nan
    M3[clean_indices] = np.nan
    return M1, M2, M3

def hill_dbs(ordered_data, t_bootstrap = 0.5,
            r_bootstrap = 500, eps_stop = 1.0,
            verbose = False, diagn_plots = False):
    """
        Function to perform double-bootstrap procedure for
        Hill estimator.

        Args:
            ordered_data: numpy array for which double-bootstrap
                          is performed. Decreasing ordering is required.
            t_bootstrap:  parameter controlling the size of the 2nd
                          bootstrap. Defined from n2 = n*(t_bootstrap).
            r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                          bootstraps.
            eps_stop:     parameter controlling range of AMSE minimization.
                          Defined as the fraction of order statistics to consider
                          during the AMSE minimization step.
            verbose:      flag controlling bootstrap verbosity. 
            diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                          plots.

        Returns:
            k_star:     number of order statistics optimal for estimation
                        according to the double-bootstrap procedure.
            x1_arr:     array of fractions of order statistics used for the
                        1st bootstrap sample.
            n1_amse:    array of AMSE values produced by the 1st bootstrap
                        sample.
            k1_min:     value of fraction of order statistics corresponding
                        to the minimum of AMSE for the 1st bootstrap sample.
            max_index1: index of the 1st bootstrap sample's order statistics
                        array corresponding to the minimization boundary set
                        by eps_stop parameter.
            x2_arr:     array of fractions of order statistics used for the
                        2nd bootstrap sample.
            n2_amse:    array of AMSE values produced by the 2nd bootstrap
                        sample.
            k2_min:     value of fraction of order statistics corresponding
                        to the minimum of AMSE for the 2nd bootstrap sample.
            max_index2: index of the 2nd bootstrap sample's order statistics
                        array corresponding to the minimization boundary set
                        by eps_stop parameter.

    """
    if verbose:
        print("Performing Hill double-bootstrap...")
    n = len(ordered_data)
    eps_bootstrap = 0.5*(1+np.log(int(t_bootstrap*n))/np.log(n))
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(n1-1)
    good_counts1 = np.zeros(n1-1)
    k1 = None
    k2 = None
    min_index1 = 1
    min_index2 = 1
    while k2 == None:
        # first bootstrap with n1 sample size
        for i in range(r_bootstrap):
            sample = np.random.choice(ordered_data, n1, replace = True)
            sample[::-1].sort()
            M1, M2 = get_moments_estimates_2(sample)
            current_amse1 = (M2 - 2.*(M1)**2)**2
            samples_n1 += current_amse1
            good_counts1[np.where(current_amse1 != np.nan)] += 1
        averaged_delta = samples_n1 / good_counts1
        
        max_index1 = (np.abs(np.linspace(1./n1, 1.0, n1) - eps_stop)).argmin()
        k1 = np.nanargmin(averaged_delta[min_index1:max_index1]) + 1 + min_index1 #take care of indexing
        if diagn_plots:
            n1_amse = averaged_delta
            x1_arr = np.linspace(1./n1, 1.0, n1)
        
        # second bootstrap with n2 sample size
        n2 = int(n1*n1/float(n))
        samples_n2 = np.zeros(n2-1)
        good_counts2 = np.zeros(n2-1)
    
        for i in range(r_bootstrap):
            sample = np.random.choice(ordered_data, n2, replace = True)
            sample[::-1].sort()
            M1, M2 = get_moments_estimates_2(sample)
            current_amse2 = (M2 - 2.*(M1**2))**2
            samples_n2 += current_amse2
            good_counts2[np.where(current_amse2 != np.nan)] += 1
        max_index2 = (np.abs(np.linspace(1./n2, 1.0, n2) - eps_stop)).argmin()
        averaged_delta = samples_n2 / good_counts2
        
        
        k2 = np.nanargmin(averaged_delta[min_index2:max_index2]) + 1 + min_index2 #take care of indexing
        if diagn_plots:
            n2_amse = averaged_delta
            x2_arr = np.linspace(1./n2, 1.0, n2)

        if k2 > k1:
            print("Warning (Hill): k2 > k1, AMSE false minimum suspected, resampling...")
            # move left AMSE boundary to avoid numerical issues
            min_index1 = min_index1 + int(0.005*n)
            min_index2 = min_index2 + int(0.005*n)
            k2 = None
    
    '''
    # this constant is provided in the Danielsson's paper
    # use instead of rho below if needed
    rho = (np.log(k1)/(2.*np.log(n1) - np.log(k1)))\
          **(2.*(np.log(n1) - np.log(k1))/(np.log(n1)))
    '''
    
    # this constant is provided in Qi's paper
    rho = (1. - (2*(np.log(k1) - np.log(n1))/(np.log(k1))))**(np.log(k1)/np.log(n1) - 1.)
    
    k_star = (k1*k1/float(k2)) * rho
    k_star = int(np.round(k_star))
    
    # enforce k_star to pick 2nd value (rare cases of extreme cutoffs)
    if k_star == 0:
        k_star = 2
    if int(k_star) >= len(ordered_data):
        print("WARNING: estimated threshold k is larger than the size of data")
        k_star = len(ordered_data)-1
    if verbose:
        print("--- Hill double-bootstrap information ---")
        print("Size of the 1st bootstrap sample n1:", n1)
        print("Size of the 2nd bootstrap sample n2:", n2)
        print("Estimated k1:", k1)
        print("Estimated k2:", k2)
        print("Estimated constant rho:", rho)
        print("Estimated optimal k:", k_star)
        print("-----------------------------------------")
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    return k_star, x1_arr, n1_amse, k1/float(n1), max_index1, x2_arr, n2_amse, k2/float(n2), max_index2, rho

def hill_estimator(ordered_data,
                   bootstrap = True, t_bootstrap = 0.5,
                   r_bootstrap = 500, verbose = False,
                   diagn_plots = False, eps_stop = 0.99):
    """
    Function to calculate Hill estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity. 
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        results: list containing an array of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """
    k_arr = np.arange(1, len(ordered_data))
    xi_arr = get_moments_estimates_1(ordered_data)
    if bootstrap:
        results = hill_dbs(ordered_data,
                           t_bootstrap = t_bootstrap,
                           r_bootstrap = r_bootstrap,
                           verbose = verbose, 
                           diagn_plots = diagn_plots,
                           eps_stop = eps_stop)
        k_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2, rho = results
        while k_star == None:
            print("Resampling...")
            results = hill_dbs(ordered_data,
                           t_bootstrap = t_bootstrap,
                           r_bootstrap = r_bootstrap,
                           verbose = verbose, 
                           diagn_plots = diagn_plots,
                           eps_stop = eps_stop)
            k_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2, rho = results
        xi_star = xi_arr[k_star-1]
        print("Adjusted Hill estimated gamma:", 1 + 1./xi_star)
        print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4*[None]
        x2_arr, n2_amse, k2, max_index2 = 4*[None]
    results = [k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1,\
               x2_arr, n2_amse, k2, max_index2, rho]
    return results

def smooth_hill_estimator(ordered_data, r_smooth = 2):
    """
    Function to calculate smooth Hill estimator for a
    given ordered dataset.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        r_smooth:     integer parameter controlling the width
                      of smoothing window. Typically small
                      value such as 2 or 3.
    Returns:
        k_arr:  numpy array of order statistics based on the data provided.
        xi_arr: numpy array of tail index estimates corresponding to 
                the order statistics array k_arr.
    """
    n = len(ordered_data)
    M1 = get_moments_estimates_1(ordered_data)
    xi_arr = np.zeros(int(np.floor(float(n)/r_smooth)))
    k_arr = np.arange(1, int(np.floor(float(n)/r_smooth))+1)
    xi_arr[0] = M1[0]
    bin_lengths = np.array([1.]+[float((r_smooth-1)*k) for k in k_arr[:-1]])
    cum_sum = 0.0
    for i in range(1, r_smooth*int(np.floor(float(n)/r_smooth))-1):
        k = i
        cum_sum += M1[k]
        if (k+1) % (r_smooth) == 0:
            xi_arr[int(k+1)//int(r_smooth)] = cum_sum
            cum_sum -= M1[int(k+1)//int(r_smooth)]
    xi_arr = xi_arr/bin_lengths
    return k_arr, xi_arr

# ===================================================
# ========== Moments Tail Index Estimation ==========
# ===================================================

def moments_dbs_prefactor(xi_n, n1, k1):
    """ 
    Function to calculate pre-factor used in moments
    double-bootstrap procedure.

    Args:
        xi_n: moments tail index estimate corresponding to
              sqrt(n)-th order statistic.
        n1:   size of the 1st bootstrap in double-bootstrap
              procedure.
        k1:   estimated optimal order statistic based on the 1st
              bootstrap sample.

    Returns:
        prefactor: constant used in estimation of the optimal
                   stopping order statistic for moments estimator.
    """
    def V_sq(xi_n):
        if xi_n >= 0:
            V = 1. + (xi_n)**2
            return V
        else:
            a = (1.-xi_n)**2
            b = (1-2*xi_n)*(6*((xi_n)**2)-xi_n+1)
            c = (1.-3*xi_n)*(1-4*xi_n)
            V = a*b/c
            return V

    def V_bar_sq(xi_n):
        if xi_n >= 0:
            V = 0.25*(1+(xi_n)**2)
            return V
        else:
            a = 0.25*((1-xi_n)**2)
            b = 1-8*xi_n+48*(xi_n**2)-154*(xi_n**3)
            c = 263*(xi_n**4)-222*(xi_n**5)+72*(xi_n**6)
            d = (1.-2*xi_n)*(1-3*xi_n)*(1-4*xi_n)
            e = (1.-5*xi_n)*(1-6*xi_n)
            V = a*(b+c)/(d*e)
            return V
    
    def b(xi_n, rho):
        if xi_n < rho:
            a1 = (1.-xi_n)*(1-2*xi_n)
            a2 = (1.-rho-xi_n)*(1.-rho-2*xi_n)
            return a1/a2
        elif xi_n >= rho and xi_n < 0:
            return 1./(1-xi_n)
        else:
            b = (xi_n/(rho*(1.-rho))) + (1./((1-rho)**2))
            return b

    def b_bar(xi_n, rho):
        if xi_n < rho:
            a1 = 0.5*(-rho*(1-xi_n)**2)
            a2 = (1.-xi_n-rho)*(1-2*xi_n-rho)*(1-3*xi_n-rho)
            return a1/a2
        elif xi_n >= rho and xi_n < 0:
            a1 = 1-2*xi_n-np.sqrt((1-xi_n)*(1-2.*xi_n))
            a2 = (1.-xi_n)*(1-2*xi_n)
            return a1/a2
        else:
            b = (-1.)*((rho + xi_n*(1-rho))/(2*(1-rho)**3))
            return b

    rho = np.log(k1)/(2*np.log(k1) - 2.*np.log(n1))
    a = (V_sq(xi_n)) * (b_bar(xi_n, rho)**2)
    b = V_bar_sq(xi_n) * (b(xi_n, rho)**2)
    prefactor = (a/b)**(1./(1. - 2*rho))
    return prefactor, rho

def moments_dbs(ordered_data, xi_n, t_bootstrap = 0.5,
                r_bootstrap = 500, eps_stop = 1.0,
                verbose = False, diagn_plots = False):
    """
    Function to perform double-bootstrap procedure for 
    moments estimator.

    Args:
        ordered_data: numpy array for which double-bootstrap
                      is performed. Decreasing ordering is required.
        xi_n:         moments tail index estimate corresponding to
                      sqrt(n)-th order statistic.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity. 
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.
        

    Returns:
        k_star:     number of order statistics optimal for estimation
                    according to the double-bootstrap procedure.
        x1_arr:     array of fractions of order statistics used for the
                    1st bootstrap sample.
        n1_amse:    array of AMSE values produced by the 1st bootstrap
                    sample.
        k1_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 1st bootstrap sample.
        max_index1: index of the 1st bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.
        x2_arr:     array of fractions of order statistics used for the
                    2nd bootstrap sample.
        n2_amse:    array of AMSE values produced by the 2nd bootstrap
                    sample.
        k2_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 2nd bootstrap sample.
        max_index2: index of the 2nd bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.
    """
    if verbose:
        print("Performing moments double-bootstrap...")
    n = len(ordered_data)
    eps_bootstrap = 0.5*(1+np.log(int(t_bootstrap*n))/np.log(n))

    # first bootstrap with n1 sample size
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(n1-1)
    good_counts1 = np.zeros(n1-1)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n1, replace = True)
        sample[::-1].sort()
        M1, M2, M3 = get_moments_estimates_3(sample)
        xi_2 = M1 + 1. - 0.5*((1. - (M1*M1)/M2))**(-1.)
        xi_3 = np.sqrt(0.5*M2) + 1. - (2./3.)*(1. / (1. - M1*M2/M3))
        samples_n1 += (xi_2 - xi_3)**2
        good_counts1[np.where((xi_2 - xi_3)**2 != np.nan)] += 1
    max_index1 = (np.abs(np.linspace(1./n1, 1.0, n1) - eps_stop)).argmin()
    averaged_delta = samples_n1 / good_counts1
    k1 = np.nanargmin(averaged_delta[:max_index1]) + 1 #take care of indexing
    if diagn_plots:
        n1_amse = averaged_delta
        x1_arr = np.linspace(1./n1, 1.0, n1)
    

    #r second bootstrap with n2 sample size
    n2 = int(n1*n1/float(n))
    samples_n2 = np.zeros(n2-1)
    good_counts2 = np.zeros(n2-1)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n2, replace = True)
        sample[::-1].sort()
        M1, M2, M3 = get_moments_estimates_3(sample)
        xi_2 = M1 + 1. - 0.5*(1. - (M1*M1)/M2)**(-1.)
        xi_3 = np.sqrt(0.5*M2) + 1. - (2./3.)*(1. / (1. - M1*M2/M3))
        samples_n2 += (xi_2 - xi_3)**2
        good_counts2[np.where((xi_2 - xi_3)**2 != np.nan)] += 1
    max_index2 = (np.abs(np.linspace(1./n2, 1.0, n2) - eps_stop)).argmin()
    averaged_delta = samples_n2 / good_counts2
    k2 = np.nanargmin(averaged_delta[:max_index2]) + 1 #take care of indexing
    if diagn_plots:
        n2_amse = averaged_delta
        x2_arr = np.linspace(1./n2, 1.0, n2)
    
    if k2 > k1:
        print("WARNING(moments): estimated k2 is greater than k1! Re-doing bootstrap...") 
        return 9*[None]
    
    #calculate estimated optimal stopping k
    prefactor, rho = moments_dbs_prefactor(xi_n, n1, k1)
    k_star = int((k1*k1/float(k2)) * prefactor)

    if int(k_star) >= len(ordered_data):
        print("WARNING: estimated threshold k is larger than the size of data")
        k_star = len(ordered_data)-1
    if verbose:
        print("--- Moments double-bootstrap information ---")
        print("Size of the 1st bootstrap sample n1:", n1)
        print("Size of the 2nd bootstrap sample n2:", n2)
        print("Estimated k1:", k1)
        print("Estimated k2:", k2)
        print("Estimated constant:", prefactor)
        print("Estimated optimal k:", k_star)
        print("--------------------------------------------")
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    return k_star, x1_arr, n1_amse, k1/float(n1), max_index1, x2_arr, n2_amse, k2/float(n2), max_index2, rho

def moments_estimator(ordered_data,
                      bootstrap = True, t_bootstrap = 0.5,
                      r_bootstrap = 500, verbose = False,
                      diagn_plots = False, eps_stop = 0.99):
    """
    Function to calculate moments estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity. 
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        results: list containing an array of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """
    n =  len(ordered_data)
    M1, M2 = get_moments_estimates_2(ordered_data)
    xi_arr = M1 + 1. - 0.5*(1. - (M1*M1)/M2)**(-1)
    k_arr = np.arange(1, len(ordered_data))
    if bootstrap:
        xi_n = xi_arr[int(np.floor(n**0.5))-1]
        results = moments_dbs(ordered_data, xi_n,
                              t_bootstrap = t_bootstrap,
                              r_bootstrap = r_bootstrap,
                              verbose = verbose, 
                              diagn_plots = diagn_plots,
                              eps_stop = eps_stop)
        while results[0] == None:
            print("Resampling...")
            results = moments_dbs(ordered_data, xi_n,
                                  t_bootstrap = t_bootstrap,
                                  r_bootstrap = r_bootstrap,
                                  verbose = verbose, 
                                  diagn_plots = diagn_plots,
                                  eps_stop = eps_stop)
        k_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2, rho = results
        xi_star = xi_arr[k_star-1]
        if xi_star <= 0:
            print ("Moments estimated gamma: infinity (xi <= 0).")
        else:
            print ("Moments estimated gamma:", 1 + 1./xi_star)
        print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4*[None]
        x2_arr, n2_amse, k2, max_index2 = 4*[None]
    results = [k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1,\
              x2_arr, n2_amse, k2, max_index2, rho]
    return results

# =======================================================
# ========== Kernel-type Tail Index Estimation ==========
# =======================================================

def get_biweight_kernel_estimates(ordered_data, hsteps, alpha):
    """
    Function to calculate biweight kernel-type estimates for tail index.
    Biweight kernel is defined as:
    phi(u) = (15/8) * (1 - u^2)^2

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.

    Returns:
        h_arr:  numpy array of fractions of order statistics included
                in kernel-type tail index estimation.
        xi_arr: numpy array with tail index estimated corresponding
                to different fractions of order statistics included
                listed in h_arr array.
    """
    n = len(ordered_data)
    logs = np.log(ordered_data)
    differences = logs[:-1] - logs[1:]
    i_arr = np.arange(1, n)/float(n)
    i3_arr = i_arr**3
    i5_arr = i_arr**5
    i_alpha_arr = i_arr**alpha
    i_alpha2_arr = i_arr**(2.+alpha)
    i_alpha4_arr = i_arr**(4.+alpha)
    t1 = np.cumsum(i_arr*differences)
    t2 = np.cumsum(i3_arr*differences)
    t3 = np.cumsum(i5_arr*differences)
    t4 = np.cumsum(i_alpha_arr*differences)
    t5 = np.cumsum(i_alpha2_arr*differences)
    t6 = np.cumsum(i_alpha4_arr*differences)
    h_arr = np.logspace(np.log10(1./n), np.log10(1.0), hsteps)
    max_i_vector = (np.floor((n*h_arr))-2.).astype(int)
    gamma_pos = (15./(8*h_arr))*t1[max_i_vector]\
                - (15./(4*(h_arr**3)))*t2[max_i_vector]\
                + (15./(8*(h_arr**5)))*t3[max_i_vector]

    q1 = (15./(8*h_arr))*t4[max_i_vector]\
         + (15./(8*(h_arr**5)))*t6[max_i_vector]\
         - (15./(4*(h_arr**3)))*t5[max_i_vector]

    q2 = (15.*(1+alpha)/(8*h_arr))*t4[max_i_vector]\
         + (15.*(5+alpha)/(8*(h_arr**5)))*t6[max_i_vector]\
         - (15.*(3+alpha)/(4*(h_arr**3)))*t5[max_i_vector]

    xi_arr = gamma_pos -1. + q2/q1
    return h_arr, xi_arr


def get_triweight_kernel_estimates(ordered_data, hsteps, alpha):
    """
    Function to calculate triweight kernel-type estimates for tail index.
    Triweight kernel is defined as:
    phi(u) = (35/16) * (1 - u^2)^3

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.

    Returns:
        h_arr:  numpy array of fractions of order statistics included
                in kernel-type tail index estimation.
        xi_arr: numpy array with tail index estimated corresponding
                to different fractions of order statistics included
                listed in h_arr array.
    """
    n = len(ordered_data)
    logs = np.log(ordered_data)
    differences = logs[:-1] - logs[1:]
    i_arr = np.arange(1, n)/float(n)
    i3_arr = i_arr**3
    i5_arr = i_arr**5
    i7_arr = i_arr**7
    i_alpha_arr = i_arr**alpha
    i_alpha2_arr = i_arr**(2.+alpha)
    i_alpha4_arr = i_arr**(4.+alpha)
    i_alpha6_arr = i_arr**(6.+alpha)
    t1 = np.cumsum(i_arr*differences)
    t2 = np.cumsum(i3_arr*differences)
    t3 = np.cumsum(i5_arr*differences)
    t4 = np.cumsum(i7_arr*differences)
    t5 = np.cumsum(i_alpha_arr*differences)
    t6 = np.cumsum(i_alpha2_arr*differences)
    t7 = np.cumsum(i_alpha4_arr*differences)
    t8 = np.cumsum(i_alpha6_arr*differences)
    h_arr = np.logspace(np.log10(1./n), np.log10(1.0), hsteps)
    max_i_vector = (np.floor((n*h_arr))-2.).astype(int)

    gamma_pos = (35./(16*h_arr))*t1[max_i_vector]\
                - (105./(16*(h_arr**3)))*t2[max_i_vector]\
                + (105./(16*(h_arr**5)))*t3[max_i_vector]\
                - (35./(16*(h_arr**7)))*t4[max_i_vector]

    q1 = (35./(16*h_arr))*t5[max_i_vector]\
         + (105./(16*(h_arr**5)))*t7[max_i_vector]\
         - (105./(16*(h_arr**3)))*t6[max_i_vector]\
         - (35./(16*(h_arr**7)))*t8[max_i_vector]

    q2 = (35.*(1+alpha)/(16*h_arr))*t5[max_i_vector] \
        + (105.*(5+alpha)/(16*(h_arr**5)))*t7[max_i_vector] \
        - (105.*(3+alpha)/(16*(h_arr**3)))*t6[max_i_vector] \
        - (35.*(7+alpha)/(16*(h_arr**7)))*t8[max_i_vector]

    xi_arr = gamma_pos - 1. + q2/q1
    return h_arr, xi_arr

def kernel_type_dbs(ordered_data, hsteps, t_bootstrap = 0.5,
                    r_bootstrap = 500, alpha = 0.6, eps_stop = 1.0,
                    verbose = False, diagn_plots = False):
    """
    Function to perform double-bootstrap procedure for 
    moments estimator.

    Args:
        ordered_data: numpy array for which double-bootstrap
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity. 
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.
        

    Returns:
        h_star:       fraction of order statistics optimal for estimation
                      according to the double-bootstrap procedure.
        x1_arr:       array of fractions of order statistics used for the
                      1st bootstrap sample.
        n1_amse:      array of AMSE values produced by the 1st bootstrap
                      sample.
        h1:           value of fraction of order statistics corresponding
                      to the minimum of AMSE for the 1st bootstrap sample.
        max_k_index1: index of the 1st bootstrap sample's order statistics
                      array corresponding to the minimization boundary set
                      by eps_stop parameter.
        x2_arr:       array of fractions of order statistics used for the
                      2nd bootstrap sample.
        n2_amse:      array of AMSE values produced by the 2nd bootstrap
                      sample.
        h2:           value of fraction of order statistics corresponding
                      to the minimum of AMSE for the 2nd bootstrap sample.
        max_k_index2: index of the 2nd bootstrap sample's order statistics
                      array corresponding to the minimization boundary set
                      by eps_stop parameter.
    """
    if verbose:
        print("Performing kernel double-bootstrap...")
    n = len(ordered_data)
    eps_bootstrap = 0.5*(1+np.log(int(t_bootstrap*n))/np.log(n))
    
    # first bootstrap with n1 sample size
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(hsteps)
    good_counts1 = np.zeros(hsteps)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n1, replace = True)
        sample[::-1].sort()
        _, xi2_arr = get_biweight_kernel_estimates(sample, hsteps, alpha)
        _, xi3_arr = get_triweight_kernel_estimates(sample, hsteps, alpha)
        samples_n1 += (xi2_arr - xi3_arr)**2
        good_counts1[np.where((xi2_arr - xi3_arr)**2 != np.nan)] += 1
    max_index1 = (np.abs(np.logspace(np.log10(1./n1), np.log10(1.0), hsteps) - eps_stop)).argmin()
    x1_arr = np.logspace(np.log10(1./n1), np.log10(1.0), hsteps)
    averaged_delta = samples_n1 / good_counts1
    h1 = x1_arr[np.nanargmin(averaged_delta[:max_index1])] 
    if diagn_plots:
        n1_amse = averaged_delta
        
    
    # second bootstrap with n2 sample size
    n2 = int(n1*n1/float(n))
    if n2 < hsteps:
        sys.exit("Number of h points is larger than number "+\
                 "of order statistics! Please either increase "+\
                 "the size of 2nd bootstrap or decrease number "+\
                 "of h grid points.")
    samples_n2 = np.zeros(hsteps)
    good_counts2 = np.zeros(hsteps)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n2, replace = True)
        sample[::-1].sort()
        _, xi2_arr = get_biweight_kernel_estimates(sample, hsteps, alpha)
        _, xi3_arr = get_triweight_kernel_estimates(sample, hsteps, alpha)
        samples_n2 += (xi2_arr - xi3_arr)**2
        good_counts2[np.where((xi2_arr - xi3_arr)**2 != np.nan)] += 1
    max_index2 = (np.abs(np.logspace(np.log10(1./n2), np.log10(1.0), hsteps) - eps_stop)).argmin()
    x2_arr = np.logspace(np.log10(1./n2), np.log10(1.0), hsteps)
    averaged_delta = samples_n2 / good_counts2
    h2 = x2_arr[np.nanargmin(averaged_delta[:max_index2])]
    if diagn_plots:
        n2_amse = averaged_delta
    
    A = (143.*((np.log(n1) + np.log(h1))**2)/(3*(np.log(n1) - 13. * np.log(h1))**2))\
        **(-np.log(h1)/np.log(n1))
    
    h_star = (h1*h1/float(h2)) * A

    if h_star > 1:
        print("WARNING: estimated threshold is larger than the size of data!")
        print("WARNING: optimal h is set to 1...")
        h_star = 1.

    if verbose:
        print("--- Kernel-type double-bootstrap information ---")
        print("Size of the 1st bootstrap sample n1:", n1)
        print("Size of the 2nd bootstrap sample n2:", n2)
        print("Estimated h1:", h1)
        print("Estimated h2:", h2)
        print("Estimated constant A:", A)
        print("Estimated optimal h:", h_star)
        print("------------------------------------------------")
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    if x1_arr is not None:
        max_k_index1 = x1_arr[max_index1]
    else:
        max_k_index1 = None
    if x2_arr is not None:
        max_k_index2 = x2_arr[max_index2]
    else:
        max_k_index2 = None
    return h_star, x1_arr, n1_amse, h1, max_k_index1, x2_arr, n2_amse, h2, max_k_index2

def kernel_type_estimator(ordered_data, hsteps, alpha = 0.6,
                         bootstrap = True, t_bootstrap = 0.5,
                         r_bootstrap = 500, verbose = False,
                         diagn_plots = False, eps_stop = 0.99):
    """
    Function to calculate kernel-type estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        hsteps:       parameter controlling number of bandwidth steps
                      of the kernel-type estimator.
        alpha:        parameter controlling the amount of "smoothing"
                      for the kernel-type estimator. Should be greater
                      than 0.5.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity. 
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        results: list containing an array of fractions of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """

    n = len(ordered_data)
    h_arr, xi_arr = get_biweight_kernel_estimates(ordered_data, hsteps,
                                                  alpha = alpha)
    if bootstrap:
        results = kernel_type_dbs(ordered_data, hsteps,
                                  t_bootstrap = t_bootstrap,
                                  alpha = alpha, r_bootstrap = r_bootstrap,
                                  verbose = verbose, diagn_plots = diagn_plots,
                                  eps_stop = eps_stop)
        h_star, x1_arr, n1_amse, h1, max_index1, x2_arr, n2_amse, h2, max_index2 = results
        while h_star == None:
            print("Resampling...")
            results = kernel_type_dbs(ordered_data, hsteps,
                                  t_bootstrap = t_bootstrap,
                                  alpha = alpha, r_bootstrap = r_bootstrap,
                                  verbose = verbose, diagn_plots = diagn_plots,
                                  eps_stop = eps_stop)
            h_star, x1_arr, n1_amse, h1, max_index1, x2_arr, n2_amse, h2, max_index2 = results
        
        #get k index which corresponds to h_star
        k_star = np.argmin(np.abs(h_arr - h_star))
        xi_star = xi_arr[k_star]
        k_arr = []
        k_star = int(np.floor(h_arr[k_star]*n))-1
        k_arr = np.floor((h_arr * n))
        if xi_star <= 0:
            print ("Kernel-type estimated gamma: infinity (xi <= 0).")
        else:
            print ("Kernel-type estimated gamma:", 1 + 1./xi_star)
        print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, h1, max_index1 = 4*[None]
        x2_arr, n2_amse, h2, max_index2 = 4*[None]
        k_arr = np.floor(h_arr * n)
    results = [np.array(k_arr), xi_arr, k_star, xi_star, x1_arr, n1_amse, h1, max_index1,\
              x2_arr, n2_amse, h2, max_index2]
    return results

# ====================================================
# ========== Pickands Tail Index Estimation ==========
# ====================================================

def pickands_estimator(ordered_data):
    """
    Function to calculate Pickands estimator for the tail index.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.

    Returns:
        k_arr:  array containing order statistics used for
                Pickands estimator calculation. Note that only estimates
                up to floor(n/4)-th order statistic can be calculated.
        xi_arr: array containing tail index estimates corresponding
                to k-order statistics provided in k_arr.
    """
    n = len(ordered_data)
    indices_k = np.arange(1, int(np.floor(n/4.))+1)
    indices_2k = 2*indices_k
    indices_4k = 4*indices_k
    Z_k = ordered_data[indices_k-1]
    Z_2k = ordered_data[indices_2k-1]
    Z_4k = ordered_data[indices_4k-1]
    xi_arr = (1./np.log(2)) * np.log((Z_k - Z_2k) / (Z_2k - Z_4k))
    k_arr = np.array([float(i) for i in range(1, int(np.floor(n/4.))+1)])
    return k_arr, xi_arr




# ==================================================
# ============= Expectile bootstrap ================
# ==================================================

def ER_loss(eps, tau, sample):
    n = len(sample)
    above_loss = 1/n * np.sum(np.square(sample[sample  > eps] - eps))
    below_loss = 1/n * np.sum(np.square(sample[sample <= eps] - eps))
    return tau * above_loss + (1-tau) * below_loss

def expectile_opt(sample, tau):
    """
    compute sample expectiles for multiple taus using optimization routine
    """
    if tau == 0:
        return np.min(sample)
    return opt.minimize(lambda eps: ER_loss(eps, tau, sample), x0=0).x[0]


def Hill_variance(gamma):
    return gamma**2

def Hill_bias(rho):
    return 1/(1.-rho)

def Expectile_bias(gamma, rho, expected_value):
    if np.abs(gamma) > np.abs(rho):
        return ((1.-gamma)*(1./gamma - 1.)**(-rho))/((1.-rho)*(1.-gamma-rho))
    else:
        return ((expected_value*(gamma**2))*(1./gamma - 1.)**(gamma))/(gamma + 1.)
    
def Expectile_variance(gamma):
    return 2.*(gamma**3)/(1.-2.*gamma)

def Hill_expectile_bias(gamma, rho, expected_value):
    if np.abs(gamma) > np.abs(rho):
        return Hill_bias(rho) - Expectile_bias(gamma, rho, expected_value)
    else: 
        return Expectile_bias(gamma, rho, expected_value)

def Hill_expectile_variance(gamma):
    return Hill_variance(gamma) + Expectile_variance(gamma) - 2*Hill_expectile_expectation(gamma)

def Hill_expectile_expectation(gamma):
    return -gamma**2 + gamma*(gamma**(1-gamma))/((1-gamma)**(1-gamma))

def Moment_variance(gamma):
    return (gamma**3)*(-8.*gamma**4+8.*gamma**3+14.*gamma-20.)/((gamma-1.)**2*(2.*gamma - 1.))

def Moment_expectile_expectation(gamma):
    return 2.*gamma**3*(gamma - 3.)/((gamma - 1.)*(2.*gamma - 1.))

def Moment_expectile_variance(gamma):
    return Moment_variance(gamma)-4.*gamma*Moment_expectile_expectation(gamma)+4.*gamma**2*Expectile_variance(gamma)


def get_expectiles_estimates(sample_expectiles):
    logs_1 = np.log(sample_expectiles)
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    k_vector = np.arange(1, len(sample_expectiles))
    gamma_exp = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    return gamma_exp


def bootstrap_prefactor(gamma, expected_value, n1, k1):
    power = np.log(k1)/(2*np.log(k1) - 2.*np.log(n1))
    return ((Expectile_variance(gamma)*(Hill_expectile_bias(gamma, power, expected_value))**2)/(Hill_expectile_variance(gamma)*(Expectile_bias(gamma, power, expected_value))**2))**(1./(1. - 2*power)), power


def expectiles_dbs(ordered_data_exp, ordered_data, xi_n, t_bootstrap = 0.5,
                r_bootstrap = 500, eps_stop = 1.0, 
                verbose = False, diagn_plots = False):
    """
    Function to perform double-bootstrap procedure for 
    hill and expectile estimator.

    Args:
        ordered_data:     numpy array for which double-bootstrap
                          is performed. Decreasing ordering is required.
        ordered_data_exp: numpy array of expectiles for which double-bootstrap
                          is performed. Decreasing ordering is required.
        xi_n:             moments tail index estimate corresponding to
                          sqrt(n)-th order statistic.
        t_bootstrap:      parameter controlling the size of the 2nd
                          bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:      number of bootstrap resamplings for the 1st and 2nd
                          bootstraps.
        eps_stop:         parameter controlling range of AMSE minimization.
                          Defined as the fraction of order statistics to consider
                          during the AMSE minimization step.
        verbose:          flag controlling bootstrap verbosity. 
        diagn_plots:      flag to switch on/off generation of AMSE diagnostic
                          plots.
        

    Returns:
        k_star:     number of order statistics optimal for estimation
                    according to the double-bootstrap procedure.
        x1_arr:     array of fractions of order statistics used for the
                    1st bootstrap sample.
        n1_amse:    array of AMSE values produced by the 1st bootstrap
                    sample.
        k1_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 1st bootstrap sample.
        max_index1: index of the 1st bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.
        x2_arr:     array of fractions of order statistics used for the
                    2nd bootstrap sample.
        n2_amse:    array of AMSE values produced by the 2nd bootstrap
                    sample.
        k2_min:     value of fraction of order statistics corresponding
                    to the minimum of AMSE for the 2nd bootstrap sample.
        max_index2: index of the 2nd bootstrap sample's order statistics
                    array corresponding to the minimization boundary set
                    by eps_stop parameter.
    """
    if verbose:
        print("Performing moments double-bootstrap...")
    n = len(ordered_data)
    eps_bootstrap = 0.5*(1+np.log(int(t_bootstrap*n))/np.log(n))

    # first bootstrap with n1 sample size
    n1 = int(n**eps_bootstrap)
    samples_n1 = np.zeros(n1-1)
    good_counts1 = np.zeros(n1-1)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n1, replace = True)
        sample_expectiles = np.random.choice(ordered_data_exp, n1, replace = True)
        sample[::-1].sort()
        sample_expectiles[::-1].sort()
        H = get_moments_estimates_1(sample)
        E = get_expectiles_estimates(sample_expectiles)
        samples_n1 += (H - E)**2
        good_counts1[np.where((H - E)**2 != np.nan)] += 1
    max_index1 = (np.abs(np.linspace(1./n1, 1.0, n1) - eps_stop)).argmin()
    averaged_delta = samples_n1 / good_counts1
    k1 = np.nanargmin(averaged_delta[:max_index1]) + 1 #take care of indexing
    if diagn_plots:
        n1_amse = averaged_delta
        x1_arr = np.linspace(1./n1, 1.0, n1)
    

    #r second bootstrap with n2 sample size
    n2 = int(n1*n1/float(n))
    samples_n2 = np.zeros(n2-1)
    good_counts2 = np.zeros(n2-1)
    for i in range(r_bootstrap):
        sample = np.random.choice(ordered_data, n2, replace = True)
        sample[::-1].sort()
        sample_expectiles = np.random.choice(ordered_data_exp, n2, replace = True)
        sample_expectiles[::-1].sort()
        H = get_moments_estimates_1(sample)
        E = get_expectiles_estimates(sample_expectiles)
        samples_n2 += (H - E)**2
        good_counts2[np.where((H - E)**2 != np.nan)] += 1
    max_index2 = (np.abs(np.linspace(1./n2, 1.0, n2) - eps_stop)).argmin()
    averaged_delta = samples_n2 / good_counts2
    k2 = np.nanargmin(averaged_delta[:max_index2]) + 1 #take care of indexing
    if diagn_plots:
        n2_amse = averaged_delta
        x2_arr = np.linspace(1./n2, 1.0, n2)
    
    if k2 > k1:
        print("WARNING(moments): estimated k2 is greater than k1! Re-doing bootstrap...") 
        return 10*[None]
    
    #calculate estimated optimal stopping k
    prefactor, power = bootstrap_prefactor(xi_n, np.mean(ordered_data), n1, k1)
    try:
        k_star = int((k1*k1/float(k2)) * prefactor)
    except:
        print(f"WARNING:{prefactor}, {power}")

    if int(k_star) >= len(ordered_data):
        print("WARNING: estimated threshold k is larger than the size of data")
        k_star = len(ordered_data)-1
    if verbose:
        print("--- Moments double-bootstrap information ---")
        print("Size of the 1st bootstrap sample n1:", n1)
        print("Size of the 2nd bootstrap sample n2:", n2)
        print("Estimated k1:", k1)
        print("Estimated k2:", k2)
        print("Estimated constant:", prefactor)
        print("Estimated optimal k:", k_star)
        print("--------------------------------------------")
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    
    final_expectile_estimator = get_expectiles_estimates(ordered_data_exp)[min(k_star, n-2)]
    return k_star, x1_arr, n1_amse, k1/float(n1), max_index1, x2_arr, n2_amse, k2/float(n2), max_index2, power



def expectile_estimator(ordered_data, expectiles_parameter = None,
                   bootstrap = True, t_bootstrap = 0.5,
                   r_bootstrap = 500, verbose = False,
                   diagn_plots = False, eps_stop = 0.99):
    """
    Function to calculate Hill estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.

    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity. 
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.

    Returns:
        results: list containing an array of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """
    if expectiles_parameter is None:
        expectiles_parameter = ordered_data.shape[0]
    taus = np.linspace(0,1, expectiles_parameter)
    expectiles = np.flip(np.array([expectile_opt(ordered_data, tau) for tau in taus]))
    k_arr = np.arange(1, len(ordered_data))
    xi_arr = get_expectiles_estimates(expectiles)
    counter = 0
    if bootstrap:
        results = expectiles_dbs(expectiles, ordered_data, min(xi_arr[int(np.sqrt(len(ordered_data)))], 0.4999),
                           t_bootstrap = t_bootstrap,
                           r_bootstrap = r_bootstrap,
                           verbose = verbose, 
                           diagn_plots = diagn_plots,
                           eps_stop = eps_stop)
        k_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2, power = results
        while k_star == None or k_star == 0:
            counter += 1
            if counter == 30:
                return None
            print("Resampling...")
            results = expectiles_dbs(expectiles, ordered_data, min(xi_arr[int(np.sqrt(len(ordered_data)))], 0.4999),
                           t_bootstrap = t_bootstrap,
                           r_bootstrap = r_bootstrap,
                           verbose = verbose, 
                           diagn_plots = diagn_plots,
                           eps_stop = eps_stop)
            k_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2, power = results
        xi_star = xi_arr[k_star-1]
        print("Expectiles estimated gamma:", 1 + 1./xi_star)
        print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4*[None]
        x2_arr, n2_amse, k2, max_index2 = 4*[None]
    results = [k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1,\
               x2_arr, n2_amse, k2, max_index2, power]
    return results



# ==================================================
# ========== Plotting and Data Processing ==========
# ==================================================

def make_plots(ordered_data, output_file_path, distribution_name, real_parameter, number_of_bins,
               r_smooth, alpha, hsteps, bootstrap_flag, t_bootstrap,
               r_bootstrap, diagn_plots, eps_stop, theta1, theta2, 
               verbose, noise_flag, p_noise, savedata):
    """ 
    Function to create plots and save tail index estimation data.

    Args:
        ordered_data:     numpy array for which tail index estimation
                          is performed. Decreasing ordering is required.
        output_file_path: file path to which plots should be saved.
        number_of_bins:   number of log-bins for degree distribution.
        r_smooth:         integer parameter controlling the width
                          of smoothing window. Typically small
                          value such as 2 or 3.
        alpha:            parameter controlling the amount of "smoothing"
                          for the kernel-type estimator. Should be greater
                          than 0.5.
        hsteps:           parameter controlling number of bandwidth steps
                          of the kernel-type estimator.
        bootstrap_flag:   flag to switch on/off double-bootstrap procedure.
        t_bootstrap:      parameter controlling the size of the 2nd
                          bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:      number of bootstrap resamplings for the 1st and 2nd
                          bootstraps.
        diagn_plots:      flag to switch on/off generation of AMSE diagnostic
                          plots.
        eps_stop:         parameter controlling range of AMSE minimization.
                          Defined as the fraction of order statistics to
                          consider during the AMSE minimization step.
        theta1:           Lower bound of plotting range, defined as
                          k_min = ceil(n^theta1).
                          Overwritten if plots behave badly within the range.
        theta2:           Upper bound of plotting range, defined as
                          k_max = floor(n^theta2).
                          Overwritten if plots behave badly within the range.
        verbose:          flag controlling bootstrap verbosity.
        noise_flag:       Switch on/off uniform noise in range
                          [-5*10^(-p), 5*10^(-p)] that is added to each
                          data point. Used for integer-valued sequences
                          with p = 1 (default = 1).
        p_noise:          integer parameter controlling noise amplitude.
        savedata:         Flag to save data files in the directory with plots.
    """
    output_dir = os.path.dirname(os.path.realpath(output_file_path))
    output_name = os.path.splitext(os.path.basename(output_file_path))[0]
    #find expectiles
    taus = np.linspace(0, 1, len(ordered_data))
    expectiles = np.flip(np.array([expectile_opt(ordered_data, tau) for tau in taus]))
    # calculate log-binned PDF
    if verbose:
        print("Calculating PDF...")
    t1 =time.time()
    x_pdf, y_pdf = get_distribution(ordered_data,
                                    number_of_bins = number_of_bins)
    t2 =time.time()
    if verbose:
        print("Elapsed time(PDF):", t2-t1)
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_pdf.dat"), "w") as f:
            for i in range(len(x_pdf)):
                f.write(str(x_pdf[i]) + " " + str(y_pdf[i]) + "\n")

    # calculate CCDF
    if verbose:
        print("Calculating CCDF...")
    t1 = time.time()
    x_ccdf, y_ccdf = get_ccdf(ordered_data)
    t2 = time.time()
    if verbose:
        print("Elapsed time:", t2-t1)
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_ccdf.dat"), "w") as f:
            for i in range(len(x_ccdf)):
                f.write(str(x_ccdf[i]) + " " + str(y_ccdf[i]) + "\n")

    # add noise if needed
    if noise_flag:
        original_discrete_data = ordered_data
        discrete_ordered_data = ordered_data
        discrete_ordered_data[::-1].sort()
        ordered_data = add_uniform_noise(ordered_data, p = p_noise)
    ordered_data[::-1].sort()
    
    # perform Pickands estimation
    if verbose:
        print("Calculating Pickands...")
    t1=time.time()
    k_p_arr, xi_p_arr = pickands_estimator(ordered_data)
    t2 =time.time()
    if verbose:
        print("Elapsed time (Pickands):", t2-t1)
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_pickands.dat"), "w") as f:
            for i in range(len(k_p_arr)):
                f.write(str(k_p_arr[i]) + " " + str(xi_p_arr[i]) + "\n")
    

    # perform smooth Hill estimation
    if verbose:
        print("Calculating smooth Hill...")
    t1=time.time()
    k_sh_arr, xi_sh_arr = smooth_hill_estimator(ordered_data,
                                                r_smooth = r_smooth)
    t2=time.time()
    if verbose:
        print("Elapsed time (smooth Hill):", t2-t1)
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_sm_hill.dat"), "w") as f:
            for i in range(len(k_sh_arr)):
                f.write(str(k_sh_arr[i]) + " " + str(xi_sh_arr[i]) + "\n")
    
    
    # perform adjusted Hill estimation
    if verbose:
        print("Calculating adjusted Hill...")
    t1 = time.time()
    hill_results = hill_estimator(ordered_data,
                                        bootstrap = bootstrap_flag,
                                        t_bootstrap = t_bootstrap,
                                        r_bootstrap = r_bootstrap,
                                        diagn_plots = diagn_plots,
                                        eps_stop = eps_stop, 
                                        verbose = verbose)
    t2 =time.time()
    if verbose:
        print("Elapsed time (Hill):", t2-t1)
    k_h_arr = hill_results[0]
    xi_h_arr = hill_results[1]
    k_h_star = hill_results[2]
    xi_h_star = hill_results[3]
    x1_h_arr, n1_h_amse, k1_h, max_h_index1 = hill_results[4:8]
    x2_h_arr, n2_h_amse, k2_h, max_h_index2, _ = hill_results[8:]
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_adj_hill_plot.dat"), "w") as f:
            for i in range(len(k_h_arr)):
                f.write(str(k_h_arr[i]) + " " + str(xi_h_arr[i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_adj_hill_estimate.dat"), "w") as f:
            f.write(str(k_h_star) + " " + str(xi_h_star) + "\n")

    # perform moments estimation
    if verbose:
        print("Calculating moments...")
    t1 = time.time()
    moments_results = moments_estimator(ordered_data,
                                        bootstrap = bootstrap_flag,
                                        t_bootstrap = t_bootstrap,
                                        r_bootstrap = r_bootstrap,
                                        diagn_plots = diagn_plots,
                                        eps_stop = eps_stop, 
                                        verbose = verbose)
    t2 = time.time()
    if verbose:
        print("Elapsed time (moments):", t2-t1)
    k_m_arr = moments_results[0]
    xi_m_arr = moments_results[1]
    k_m_star = moments_results[2]
    xi_m_star = moments_results[3]
    x1_m_arr, n1_m_amse, k1_m, max_m_index1 = moments_results[4:8]
    x2_m_arr, n2_m_amse, k2_m, max_m_index2, _ = moments_results[8:]
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_mom_plot.dat"), "w") as f:
            for i in range(len(k_m_arr)):
                f.write(str(k_m_arr[i]) + " " + str(xi_m_arr[i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_mom_estimate.dat"), "w") as f:
            f.write(str(k_m_star) + " " + str(xi_m_star) + "\n")
    # perform kernel-type estimation
    if verbose:
        print("Calculating kernel-type...")
    t1 = time.time()
    kernel_type_results = kernel_type_estimator(ordered_data, hsteps,
                                                alpha = alpha,
                                                bootstrap = bootstrap_flag,
                                                t_bootstrap = t_bootstrap,
                                                r_bootstrap = r_bootstrap,
                                                diagn_plots = diagn_plots,
                                                eps_stop = eps_stop, 
                                                verbose = verbose)
    t2 = time.time()
    if verbose:
        print("Elapsed time (kernel-type):", t2-t1)
    k_k_arr = kernel_type_results[0]
    xi_k_arr = kernel_type_results[1]
    k_k_star = kernel_type_results[2]
    xi_k_star = kernel_type_results[3]
    x1_k_arr, n1_k_amse, h1, max_k_index1 = kernel_type_results[4:8]
    x2_k_arr, n2_k_amse, h2, max_k_index2 = kernel_type_results[8:]

    if bootstrap_flag:
        k_k1_star = np.argmin(np.abs(k_k_arr - k_k_star))
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_kern_plot.dat"), "w") as f:
            for i in range(len(k_k_arr)):
                f.write(str(k_k_arr[i]) + " " + str(xi_k_arr[i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_kern_estimate.dat"), "w") as f:
            f.write(str(k_k_arr[k_k1_star]) + " " + str(xi_k_arr[k_k1_star]) + "\n")
            
    
    #perform expectile estimation
    if verbose:
        print("Calculating adjusted Hill...")
    t1 = time.time()
    expectile_results = expectile_estimator(ordered_data,
                                        bootstrap = bootstrap_flag,
                                        t_bootstrap = t_bootstrap,
                                        r_bootstrap = r_bootstrap,
                                        diagn_plots = diagn_plots,
                                        eps_stop = eps_stop, 
                                        verbose = verbose)
    t2 =time.time()
    if verbose:
        print("Elapsed time (Hill):", t2-t1)
    k_e_arr = expectile_results[0]
    xi_e_arr = expectile_results[1]
    k_e_star = expectile_results[2]
    xi_e_star = expectile_results[3]
    x1_e_arr, n1_e_amse, k1_e, max_e_index1 = expectile_results[4:8]
    x2_e_arr, n2_e_amse, k2_e, max_e_index2, _ = expectile_results[8:]
    if savedata == 1:
        with open(os.path.join(output_dir+"/"+output_name+"_expectile_plot.dat"), "w") as f:
            for i in range(len(k_e_arr)):
                f.write(str(k_e_arr[i]) + " " + str(xi_e_arr[i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_expectile_estimate.dat"), "w") as f:
            f.write(str(k_e_star) + " " + str(xi_e_star) + "\n")
    
    # plotting part
    if verbose:
        print("Making plots...")

    fig, axes = plt.subplots(3, 2, figsize = (12, 16))
    for ax in axes.reshape(-1):
        ax.tick_params(direction='out', length=6, width=1.5,
                       labelsize = 12, which = 'major')
        ax.tick_params(direction='out', length=3, width=1, which = 'minor')
        [i.set_linewidth(1.5) for i in ax.spines.values()]
    
    # plot PDF
    axes[0,0].set_xlabel(r"Degree $k$", fontsize = 20)
    axes[0,0].set_ylabel(r"$P(k)$", fontsize = 20)
    axes[0,0].loglog(x_pdf, y_pdf, color = "#386cb0", marker = "s",
                     lw = 1.5, markeredgecolor = "black")

    # plot CCDF
    axes[0,1].set_xlabel(r"Degree $k$", fontsize = 20)
    axes[0,1].set_ylabel(r"$\bar{F}(k)$", fontsize = 20)
    axes[0,1].set_xscale("log")
    axes[0,1].set_yscale("log")
    axes[0,1].step(x_ccdf, y_ccdf, color = "#386cb0", lw = 1.5)
    
    # draw scalings
    if noise_flag:
        xmin = discrete_ordered_data[k_h_star]
    else:
        xmin = ordered_data[k_h_star]
    x = x_ccdf[np.where(x_ccdf >= xmin)]
    l = np.mean(y_ccdf[np.where(x == xmin)])
    alpha = 1./xi_h_star
    if xi_h_star > 0:
        axes[0,1].plot(x, [l*(float(xmin)/k)**alpha for k in x],
                       color = '#fb8072', ls = '--', lw = 2,
                       label = r"Adj. Hill Scaling $(\alpha="+\
                       str(np.round(1./xi_h_star, decimals = 3))+r")$")
        axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(alpha)],
                                   color = "#fb8072", ls = 'none', marker = 'o',
                                   markerfacecolor = 'none', markeredgecolor = "#fb8072",
                                   markeredgewidth = 3, markersize = 10)
    if noise_flag:
        xmin = discrete_ordered_data[k_m_star]
    else:
        xmin = ordered_data[k_m_star]
    x = x_ccdf[np.where(x_ccdf >= xmin)]
    l = np.mean(y_ccdf[np.where(x == xmin)])
    alpha = 1./xi_m_star
    if xi_m_star > 0:
        axes[0,1].plot(x, [l*(float(xmin)/k)**alpha for k in x],
                       color = '#8dd3c7', ls = '--', lw = 2,
                       label = r"Moments Scaling $(\alpha="+\
                       str(np.round(1./xi_m_star, decimals = 3))+r")$")
        axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(alpha)],
                                   color = "#8dd3c7", ls = 'none', marker = 'o',
                                   markerfacecolor = 'none', markeredgecolor = "#8dd3c7",
                                   markeredgewidth = 3, markersize = 10)
        
    if noise_flag:
        xmin = discrete_ordered_data[k_k_star]
    else:
        xmin = ordered_data[k_k_star]
    
    x = x_ccdf[np.where(x_ccdf >= xmin)]
    l = np.mean(y_ccdf[np.where(x == xmin)])
    alpha = 1./xi_k_star
    if xi_k_star > 0:
        axes[0,1].plot(x, [l*(float(xmin)/k)**alpha for k in x],
                       color = '#fdb462', ls = '--', lw = 2,
                       label = r"Kernel Scaling $(\alpha="+\
                       str(np.round(1./xi_k_star, decimals = 3))+r")$")
        axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(alpha)],
                                   color = "#fdb462", ls = 'none', marker = 'o',
                                   markerfacecolor = 'none', markeredgecolor = "#fdb462",
                                   markeredgewidth = 3, markersize = 10)
        
    if noise_flag:
        xmin = discrete_ordered_data[k_e_star]
    else:
        xmin = ordered_data[k_e_star]
    x = x_ccdf[np.where(x_ccdf >= xmin)]
    l = np.mean(y_ccdf[np.where(x == xmin)])
    alpha = 1./xi_e_star
    if xi_e_star > 0:
        axes[0,1].plot(x, [l*(float(xmin)/k)**alpha for k in x],
                       color = '#08876c', ls = '--', lw = 2,
                       label = r"Expectile Scaling $(\alpha="+\
                       str(np.round(1./xi_e_star, decimals = 3))+r")$")
        axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(alpha)],
                                   color = "#08876c", ls = 'none', marker = 'o',
                                   markerfacecolor = 'none', markeredgecolor = "#08876c",
                                   markeredgewidth = 3, markersize = 10)
    
    if real_parameter:
        axes[0,1].plot(x, [l*(float(xmin)/k)**(1./real_parameter) for k in x],
                       color = '#212212', ls = '--', lw = 2,
                       label = r"Real Scaling $(\alpha="+\
                       str(np.round(1./real_parameter, decimals = 3))+r")$")
        axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(1./real_parameter)],
                                   color = "#212212", ls = 'none', marker = 'o',
                                   markerfacecolor = 'none', markeredgecolor = "#08876c",
                                   markeredgewidth = 3, markersize = 10)
    
    axes[0,1].legend(loc = 'best')

    # define min and max order statistics to plot
    min_k = int(np.ceil(len(k_h_arr)**theta1)) - 1
    max_k = int(np.floor(len(k_h_arr)**theta2)) - 1
    # check if estimators' values are not too off in these bounds
    min_k_index = (np.abs(k_sh_arr - min_k)).argmin()
    max_k_index = (np.abs(k_sh_arr - max_k)).argmin()
    if (xi_sh_arr[min_k_index] <= -3 or xi_sh_arr[min_k_index] >= 3):
        indices_to_plot_sh = np.where((xi_sh_arr <= 3) & (xi_sh_arr >= -3))
    elif (xi_sh_arr[max_k_index] <= -3 or xi_sh_arr[max_k_index] >= 3):
        indices_to_plot_sh = np.where((xi_sh_arr <= 3) & (xi_sh_arr >= -3))
    else:
        indices_to_plot_sh = np.where((k_sh_arr <= max_k) & (k_sh_arr >= min_k))
    axes[1,0].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[1,0].set_ylabel(r"Estimated $\hat{\gamma}$", fontsize = 20)
    # plot smooth Hill
    
    axes[1,0].plot(k_sh_arr[indices_to_plot_sh], xi_sh_arr[indices_to_plot_sh],
                   color = "#b3de69", alpha = 0.8, label = "Smooth Hill",
                   zorder = 10)

    # plot adjusted Hill
    # check if estimators' values are not too off in these bounds
    if (xi_h_arr[min_k-1] <= -3 or xi_h_arr[min_k-1] >= 3):
        indices_to_plot_h = np.where((xi_h_arr <= 3) & (xi_h_arr >= -3))
    elif (xi_h_arr[max_k-1] <= -3 or xi_h_arr[max_k-1] >= 3):
        indices_to_plot_h = np.where((xi_h_arr <= 3) & (xi_h_arr >= -3))
    else:
        indices_to_plot_h = np.where((k_h_arr <= max_k) & (k_h_arr >= min_k))
    axes[1,0].plot(k_h_arr[indices_to_plot_h], xi_h_arr[indices_to_plot_h],
                   color = "#fb8072", alpha = 0.8, label = "Adjusted Hill",
                   zorder = 10)
    if bootstrap_flag:
        axes[1,0].scatter([k_h_arr[k_h_star-1]], [xi_h_arr[k_h_star-1]],
                       color = "#fb8072", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Hill}="\
                           +str(np.round([xi_h_arr[k_h_star-1]][0], decimals = 3))\
                           +r"$")
    if real_parameter:
        x = np.linspace(0, len(ordered_data), len(ordered_data) + 1)
        axes[1,0].plot(x, [real_parameter for i in x],
                   color = "#212212", alpha = 0.8, label = f"Real Parameter = {real_parameter:.3f}",
                   zorder = 10)
    axes[1,0].legend(loc = "best")

    
    axes[1,1].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[1,1].set_ylabel(r"Estimated $\hat{\gamma}$", fontsize = 20)
    axes[1,1].set_xscale("log")   
    
    # plot smooth Hill
    axes[1,1].plot(k_sh_arr[indices_to_plot_sh], xi_sh_arr[indices_to_plot_sh],
                   color = "#b3de69", alpha = 0.8, label = "Smooth Hill",
                   zorder = 10)
    # plot adjusted Hill
    indices_to_plot = np.where((k_h_arr <= max_k) & (k_h_arr >= min_k))
    axes[1,1].plot(k_h_arr[indices_to_plot_h], xi_h_arr[indices_to_plot_h],
                   color = "#fb8072", alpha = 0.8, label = "Adjusted Hill",
                   zorder = 10)
    if bootstrap_flag:
        axes[1,1].scatter([k_h_arr[k_h_star-1]], [xi_h_arr[k_h_star-1]],
                       color = "#fb8072", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Hill}="\
                           +str(np.round([xi_h_arr[k_h_star-1]][0], decimals = 3))\
                           +r"$")
    
    if real_parameter:
        x = np.linspace(0, len(ordered_data), len(ordered_data) + 1)
        axes[1,1].plot(x, [real_parameter for i in x],
                   color = "#212212", alpha = 0.8, label = f"Real Parameter = {real_parameter:.3f}",
                   zorder = 10)
    axes[1,1].legend(loc = "best")

    axes[2,0].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[2,0].set_ylabel(r"Estimated $\hat{\gamma}$", fontsize = 20)
    #plot Pickands
    min_k_index = (np.abs(k_p_arr - min_k)).argmin()
    max_k_index = (np.abs(k_p_arr - max_k)).argmin()
    if (xi_p_arr[min_k_index] <= -3 or xi_p_arr[min_k_index] >= 3):
        indices_to_plot_p = np.where((xi_p_arr <= 3) & (xi_p_arr >= -3))
    elif (xi_p_arr[max_k_index] <= -3 or xi_p_arr[max_k_index] >= 3):
        indices_to_plot_p = np.where((xi_p_arr <= 3) & (xi_p_arr >= -3))
    else:
        indices_to_plot_p = np.where((k_p_arr <= max_k) & (k_p_arr >= min_k))
    axes[2,0].plot(k_p_arr[indices_to_plot_p], xi_p_arr[indices_to_plot_p],
                   color = "#bc80bd", alpha = 0.8, label = "Pickands",
                   zorder = 10)
    
    #plot expectiles
    min_k_index = (np.abs(k_e_arr - min_k)).argmin()
    max_k_index = (np.abs(k_e_arr - max_k)).argmin()
    if (xi_e_arr[min_k_index] <= -3 or xi_e_arr[min_k_index] >= 3):
        indices_to_plot_e = np.where((xi_e_arr <= 3) & (xi_e_arr >= -3))
    elif (xi_e_arr[max_k_index] <= -3 or xi_e_arr[max_k_index] >= 3):
        indices_to_plot_e = np.where((xi_e_arr <= 3) & (xi_e_arr >= -3))
    else:
        indices_to_plot_e = np.where((k_e_arr <= max_k) & (k_e_arr >= min_k))
    axes[2,0].plot(k_e_arr[indices_to_plot_e], xi_e_arr[indices_to_plot_e],
                   color = "#08876c", alpha = 0.8, label = "Expectile",
                   zorder = 10)
    if bootstrap_flag:
        axes[2,0].scatter([k_e_arr[k_e_star-1]], [xi_e_arr[k_e_star-1]],
                       color = "#08876c", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Expectile}="\
                           +str(np.round([xi_e_arr[k_e_star-1]][0], decimals = 3))\
                           +r"$")
    
    #plot moments
    if (xi_m_arr[min_k-1] <= -3 or xi_m_arr[min_k-1] >= 3):
        indices_to_plot_m = np.where((xi_m_arr <= 3) & (xi_m_arr >= -3))
    elif (xi_m_arr[max_k-1] <= -3 or xi_m_arr[max_k-1] >= 3):
        indices_to_plot_m = np.where((xi_m_arr <= 3) & (xi_m_arr >= -3))
    else:
        indices_to_plot_m = np.where((k_m_arr <= max_k) & (k_m_arr >= min_k))
    
    axes[2,0].plot(k_m_arr[indices_to_plot_m], xi_m_arr[indices_to_plot_m],
                   color = "#8dd3c7", alpha = 0.8, label = "Moments",
                   zorder = 10)
    if bootstrap_flag:
        axes[2,0].scatter([k_m_arr[k_m_star-1]], [xi_m_arr[k_m_star-1]],
                       color = "#8dd3c7", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Moments}="\
                           +str(np.round([xi_m_arr[k_m_star-1]][0], decimals = 3))\
                           +r"$")
    #plot kernel-type
    min_k_index = (np.abs(k_k_arr - min_k)).argmin()
    max_k_index = (np.abs(k_k_arr - max_k)).argmin()
    if (xi_k_arr[min_k_index] <= -3 or xi_k_arr[min_k_index] >= 3):
        indices_to_plot_k = np.where((xi_k_arr <= 3) & (xi_k_arr >= -3))
    elif (xi_k_arr[max_k_index] <= -3 or xi_k_arr[max_k_index] >= 3):
        indices_to_plot_k = np.where((xi_k_arr <= 3) & (xi_k_arr >= -3))
    else:
        indices_to_plot_k = list(range(min_k_index, max_k_index))
    #indices_to_plot_k = np.where((xi_k_arr <= 3) & (xi_k_arr >= -3))
    axes[2,0].plot(k_k_arr[indices_to_plot_k], xi_k_arr[indices_to_plot_k],
                   color = "#fdb462", alpha = 0.8, label = "Kernel",
                   zorder = 10)
    if bootstrap_flag:
        axes[2,0].scatter([k_k_arr[k_k1_star-1]], [xi_k_arr[k_k1_star-1]],
                       color = "#fdb462", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Kernel}="\
                           +str(np.round([xi_k_arr[k_k1_star-1]][0], decimals = 3))\
                           +r"$")
 

    if real_parameter:
        x = np.linspace(0, len(ordered_data), len(ordered_data) + 1)
        axes[2,0].plot(x, [real_parameter for i in x],
                   color = "#212212", alpha = 0.8, label = f"Real Parameter = {real_parameter:.3f}",
                   zorder = 20)
    axes[2,0].legend(loc = "best")
    # for clarity purposes, display only xi region between -1 and 1
    axes[2,0].set_ylim((-0.5,1.5))
    print("!!!!!!!!!!!!!!!", real_parameter)

    axes[2,1].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[2,1].set_ylabel(r"Estimated $\hat{\gamma}$", fontsize = 20)
    axes[2,1].set_xscale("log")

    #plot Pickands
    axes[2,1].plot(k_p_arr[indices_to_plot_p], xi_p_arr[indices_to_plot_p],
                   color = "#bc80bd", alpha = 0.8, label = "Pickands",
                   zorder = 10)
    
    #plot expectiles
    axes[2,1].plot(k_e_arr[indices_to_plot_e], xi_e_arr[indices_to_plot_e],
                   color = "#08876c", alpha = 0.8, label = "Expectile",
                   zorder = 10)
    if bootstrap_flag:
        axes[2,1].scatter([k_e_arr[k_e_star-1]], [xi_e_arr[k_e_star-1]],
                       color = "#08876c", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Expectile}="\
                           +str(np.round([xi_e_arr[k_e_star-1]][0], decimals = 3))\
                           +r"$")
    #plot hill
    axes[2,1].plot(k_h_arr[indices_to_plot_h], xi_h_arr[indices_to_plot_h],
                   color = "#234567", alpha = 0.8, label = "Hill",
                   zorder = 10)
    if bootstrap_flag:
        axes[2,1].scatter([k_h_arr[k_h_star-1]], [xi_h_arr[k_h_star-1]],
                       color = "#08876c", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Hill}="\
                           +str(np.round([xi_h_arr[k_h_star-1]][0], decimals = 3))\
                           +r"$")
    
    #plot moments
    axes[2,1].plot(k_m_arr[indices_to_plot_m], xi_m_arr[indices_to_plot_m],
                   color = "#8dd3c7", alpha = 0.8, label = "Moments",
                   zorder = 10)
    if bootstrap_flag:
        axes[2,1].scatter([k_m_arr[k_m_star-1]], [xi_m_arr[k_m_star-1]],
                       color = "#8dd3c7", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Moments}="\
                           +str(np.round([xi_m_arr[k_m_star-1]][0], decimals = 3))\
                           +r"$")
    #plot kernel-type
    axes[2,1].plot(k_k_arr[indices_to_plot_k], xi_k_arr[indices_to_plot_k],
                   color = "#fdb462", alpha = 0.8, label = "Kernel",
                   zorder = 10)
    if bootstrap_flag:
        axes[2,1].scatter([k_k_arr[k_k1_star-1]], [xi_k_arr[k_k1_star-1]],
                       color = "#fdb462", marker = "*", s = 100,
                       edgecolor = "black", zorder = 20, 
                       label = r"$\widehat{\xi}^{Kernel}="\
                           +str(np.round([xi_k_arr[k_k1_star-1]][0], decimals = 3))\
                           +r"$")
    if real_parameter:
        x = np.linspace(0, len(ordered_data), len(ordered_data) + 1)
        axes[2,1].plot(x, [real_parameter for i in x],
                   color = "#212212", alpha = 0.8, label = f"Real Parameter = {real_parameter:.3f}",
                   zorder = 20)
        
    # for clarity purposes, display only xi region between -1 and 1
    axes[2,1].set_ylim((-0.5,1.5))
    axes[2,1].legend(loc = "best")


    if diagn_plots:
        fig_d, axes_d = plt.subplots(2, 4, figsize = (24, 12))
        fig_d.suptitle(f"Профили AMSE - распределение {distribution_name} с параметром {real_parameter:.3f}")

        # filter out boundary values using theta parameters for Hill
        min_k1 = 2
        max_k1 = len(x1_h_arr) - 1
        min_k2 = 2
        max_k2 = len(x2_h_arr) - 1
        axes_d[0,0].set_xscale("log")
        axes_d[0,1].set_xscale("log")
        axes_d[0,2].set_xscale("log")
        axes_d[0,3].set_xscale("log")
        n1_h_amse[np.where((n1_h_amse == np.inf) |\
                           (n1_h_amse == -np.inf))] = np.nan
        for slot in range(2):
            axes_d[slot,0].set_yscale("log")
            axes_d[slot,0].set_ylim((0.1*np.nanmin(n1_h_amse[min_k1:max_k1]), 1.0))
            axes_d[slot,0].set_xlabel("Fraction of Bootstrap Order Statistics",
                               fontsize = 20)
            axes_d[slot,0].set_ylabel(r"$\langle AMSE \rangle$", fontsize = 20)
            axes_d[slot,0].set_title("Adjusted Hill Estimator", fontsize = 20)
            # plot AMSE and corresponding minimum
            axes_d[slot,0].plot(x1_h_arr[min_k1:max_k1], n1_h_amse[min_k1:max_k1],
                           alpha = 0.5, lw = 1.5,
                           color = "#d55e00", label = r"$n_1$ samples")
            axes_d[slot,0].scatter([k1_h], [n1_h_amse[int(len(x1_h_arr)*k1_h)-1]],
                              color = "#d55e00",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_1$ sample")
            axes_d[slot,0].plot(x2_h_arr[min_k2:max_k2], n2_h_amse[min_k2:max_k2],
                           alpha = 0.5, lw = 1.5,
                           color = "#0072b2", label = r"$n_2$ samples")
            axes_d[slot,0].scatter([k2_h], [n2_h_amse[int(len(x2_h_arr)*k2_h)-1]],
                              color = "#0072b2",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_2$ sample")
            axes_d[slot,0].axvline(max_h_index1/float(len(x1_h_arr)), color = "#d55e00",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_1$ sample")
            axes_d[slot,0].axvline(max_h_index2/float(len(x2_h_arr)), color = "#0072b2",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_2$ sample")


            axes_d[slot,0].legend(loc = "best")
            if savedata == 1:
                with open(os.path.join(output_dir+"/"+output_name+"_adjhill_diagn1.dat"), "w") as f:
                    for i in range(len(x1_h_arr[min_k1:max_k1])):
                        f.write(str(x1_h_arr[min_k1:max_k1][i]) + " " + str(n1_h_amse[min_k1:max_k1][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_adjhill_diagn2.dat"), "w") as f:
                    for i in range(len(x2_h_arr[min_k2:max_k2])):
                        f.write(str(x2_h_arr[min_k2:max_k2][i]) + " " + str(n2_h_amse[min_k2:max_k2][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_adjhill_diagn_points.dat"), "w") as f:
                    f.write("Min for n1 sample: "+str(k1_h)+" "+str(n1_h_amse[int(len(x1_h_arr)*k1_h)-1])+"\n")
                    f.write("Min for n2 sample: "+str(k2_h)+" "+str(n2_h_amse[int(len(x2_h_arr)*k2_h)-1])+"\n")
                    f.write("Minimization boundary for n1 sample: "+str(max_h_index1/float(len(x1_h_arr)))+"\n")
                    f.write("Minimization boundary for n2 sample: "+str(max_h_index2/float(len(x2_h_arr)))+"\n")

            # filter out boundary values using theta parameters for moments
            min_k1 = 2
            max_k1 = len(x1_m_arr) - 1
            min_k2 = 2
            max_k2 = len(x2_m_arr) - 1
            n1_m_amse[np.where((n1_m_amse == np.inf) |\
                               (n1_m_amse == -np.inf))] = np.nan
            axes_d[slot,1].set_yscale("log")
            axes_d[slot,1].set_ylim((0.1*np.nanmin(n1_m_amse[min_k1:max_k1]), 1.0))
            axes_d[slot,1].set_xlabel("Fraction of Bootstrap Order Statistics",
                               fontsize = 20)
            axes_d[slot,1].set_ylabel(r"$\langle AMSE \rangle$", fontsize = 20)
            axes_d[slot,1].set_title("Moments Estimator", fontsize = 20)
            # plot AMSE and corresponding minimum
            axes_d[slot,1].plot(x1_m_arr[min_k1:max_k1], n1_m_amse[min_k1:max_k1],
                           alpha = 0.5, lw = 1.5,
                           color = "#d55e00", label = r"$n_1$ samples")
            axes_d[slot,1].scatter([k1_m], [n1_m_amse[int(len(x1_m_arr)*k1_m)-1]],
                              color = "#d55e00",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_1$ sample")
            axes_d[slot,1].plot(x2_m_arr[min_k2:max_k2], n2_m_amse[min_k2:max_k2],
                           alpha = 0.5, lw = 1.5,
                           color = "#0072b2", label = r"$n_2$ samples")
            axes_d[slot,1].scatter([k2_m], [n2_m_amse[int(len(x2_m_arr)*k2_m)-1]],
                              color = "#0072b2",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_2$ sample")
            axes_d[slot,1].axvline(max_m_index1/float(len(x1_m_arr)), color = "#d55e00",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_1$ sample")
            axes_d[slot,1].axvline(max_m_index2/float(len(x2_m_arr)), color = "#0072b2",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_2$ sample")
            axes_d[slot,1].legend(loc = "best")
            if savedata == 1:
                with open(os.path.join(output_dir+"/"+output_name+"_mom_diagn1.dat"), "w") as f:
                    for i in range(len(x1_m_arr[min_k1:max_k1])):
                        f.write(str(x1_m_arr[min_k1:max_k1][i]) + " " + str(n1_m_amse[min_k1:max_k1][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_mom_diagn2.dat"), "w") as f:
                    for i in range(len(x2_m_arr[min_k2:max_k2])):
                        f.write(str(x2_m_arr[min_k2:max_k2][i]) + " " + str(n2_m_amse[min_k2:max_k2][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_mom_diagn_points.dat"), "w") as f:
                    f.write("Min for n1 sample: "+str(k1_m)+" "+str(n1_m_amse[int(len(x1_m_arr)*k1_m)-1])+"\n")
                    f.write("Min for n2 sample: "+str(k2_m)+" "+str(n2_m_amse[int(len(x2_m_arr)*k2_m)-1])+"\n")
                    f.write("Minimization boundary for n1 sample: "+str(max_m_index1/float(len(x1_m_arr)))+"\n")
                    f.write("Minimization boundary for n2 sample: "+str(max_m_index2/float(len(x2_m_arr)))+"\n")        


            min_k1 = 2
            max_k1 = len(x1_k_arr)
            min_k2 = 2
            max_k2 = len(x2_k_arr)
            n1_k_amse[np.where((n1_k_amse == np.inf) |\
                               (n1_k_amse == -np.inf))] = np.nan
            axes_d[slot,2].set_yscale("log")
            axes_d[slot,2].set_ylim((0.1*np.nanmin(n1_k_amse[min_k1:max_k1]), 1.0))
            axes_d[slot,2].set_xlabel("Fraction of Bootstrap Order Statistics",
                               fontsize = 20)
            axes_d[slot,2].set_ylabel(r"$\langle AMSE \rangle$", fontsize = 20)
            axes_d[slot,2].set_title("Kernel-type Estimator", fontsize = 20)
            # plot AMSE and corresponding minimum
            axes_d[slot,2].plot(x1_k_arr[min_k1:max_k1], n1_k_amse[min_k1:max_k1],
                           alpha = 0.5, lw = 1.5,
                           color = "#d55e00", label = r"$n_1$ samples")
            axes_d[slot,2].scatter([h1], [n1_k_amse[np.where(x1_k_arr == h1)]], color = "#d55e00",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_1$ sample")
            # plot boundary of minimization
            axes_d[slot,2].axvline(max_k_index1, color = "#d55e00",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_2$ sample")
            axes_d[slot,2].plot(x2_k_arr[min_k2:max_k2], n2_k_amse[min_k2:max_k2],
                           alpha = 0.5, lw = 1.5,
                           color = "#0072b2", label = r"$n_2$ samples")
            axes_d[slot,2].scatter([h2], [n2_k_amse[np.where(x2_k_arr == h2)]], color = "#0072b2",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_2$ sample")
            axes_d[slot,2].axvline(max_k_index2, color = "#0072b2",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_2$ sample")
            axes_d[slot,2].legend(loc = "best")
            if savedata == 1:
                with open(os.path.join(output_dir+"/"+output_name+"_kern_diagn1.dat"), "w") as f:
                    for i in range(len(x1_k_arr[min_k1:max_k1])):
                        f.write(str(x1_k_arr[min_k1:max_k1][i]) + " " + str(n1_k_amse[min_k1:max_k1][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_kern_diagn2.dat"), "w") as f:
                    for i in range(len(x2_m_arr[min_k2:max_k2])):
                        f.write(str(x2_k_arr[min_k2:max_k2][i]) + " " + str(n2_k_amse[min_k2:max_k2][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_kern_diagn_points.dat"), "w") as f:
                    f.write("Min for n1 sample: "+str(h1)+" "+str(n1_k_amse[np.where(x1_k_arr == h1)][0])+"\n")
                    f.write("Min for n2 sample: "+str(h2)+" "+str(n2_k_amse[np.where(x2_k_arr == h2)][0])+"\n")
                    f.write("Minimization boundary for n1 sample: "+str(n1_k_amse[int(max_k_index1*hsteps)-1])+"\n")
                    f.write("Minimization boundary for n2 sample: "+str(n2_k_amse[int(max_k_index2*hsteps)-1])+"\n")


            # filter out boundary values using theta parameters for expectiles
            min_k1 = 2
            max_k1 = len(x1_m_arr) - 1
            min_k2 = 2
            max_k2 = len(x2_m_arr) - 1
            n1_m_amse[np.where((n1_e_amse == np.inf) |\
                               (n1_e_amse == -np.inf))] = np.nan
            axes_d[slot,3].set_yscale("log")
            axes_d[slot,3].set_ylim((0.1*np.nanmin(n1_e_amse[min_k1:max_k1]), 1.0))
            axes_d[slot,3].set_xlabel("Fraction of Bootstrap Order Statistics",
                               fontsize = 20)
            axes_d[slot,3].set_ylabel(r"$\langle AMSE \rangle$", fontsize = 20)
            axes_d[slot,3].set_title("Expectiles Estimator", fontsize = 20)
            # plot AMSE and corresponding minimum
            axes_d[slot,3].plot(x1_e_arr[min_k1:max_k1], n1_e_amse[min_k1:max_k1],
                           alpha = 0.5, lw = 1.5,
                           color = "#d55e00", label = r"$n_1$ samples")
            axes_d[slot,3].scatter([k1_e], [n1_e_amse[int(len(x1_e_arr)*k1_e)-1]],
                              color = "#d55e00",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_1$ sample")
            axes_d[slot,3].plot(x2_e_arr[min_k2:max_k2], n2_e_amse[min_k2:max_k2],
                           alpha = 0.5, lw = 1.5,
                           color = "#0072b2", label = r"$n_2$ samples")
            axes_d[slot,3].scatter([k2_e], [n2_e_amse[int(len(x2_e_arr)*k2_e)-1]],
                              color = "#0072b2",
                              marker = 'o', edgecolor = "black", alpha = 0.5,
                              label = r"Min for $n_2$ sample")
            axes_d[slot,3].axvline(max_e_index1/float(len(x1_e_arr)), color = "#d55e00",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_1$ sample")
            axes_d[slot,3].axvline(max_e_index2/float(len(x2_e_arr)), color = "#0072b2",
                              ls = '--', alpha = 0.5,
                              label = r"Minimization boundary for $n_2$ sample")
            axes_d[slot,3].legend(loc = "best")
            if savedata == 1:
                with open(os.path.join(output_dir+"/"+output_name+"_exp_diagn1.dat"), "w") as f:
                    for i in range(len(x1_e_arr[min_k1:max_k1])):
                        f.write(str(x1_e_arr[min_k1:max_k1][i]) + " " + str(n1_e_amse[min_k1:max_k1][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_exp_diagn2.dat"), "w") as f:
                    for i in range(len(x2_e_arr[min_k2:max_k2])):
                        f.write(str(x2_e_arr[min_k2:max_k2][i]) + " " + str(n2_e_amse[min_k2:max_k2][i]) + "\n")
                with open(os.path.join(output_dir+"/"+output_name+"_exp_diagn_points.dat"), "w") as f:
                    f.write("Min for n1 sample: "+str(k1_e)+" "+str(n1_m_amse[int(len(x1_e_arr)*k1_e)-1])+"\n")
                    f.write("Min for n2 sample: "+str(k2_e)+" "+str(n2_m_amse[int(len(x2_e_arr)*k2_e)-1])+"\n")
                    f.write("Minimization boundary for n1 sample: "+str(max_m_index1/float(len(x1_e_arr)))+"\n")
                    f.write("Minimization boundary for n2 sample: "+str(max_m_index2/float(len(x2_e_arr)))+"\n")
                

        fig_d.tight_layout()
        diag_plots_path = output_dir+"/"+output_name+"_diag.pdf"
        fig_d.savefig(diag_plots_path)

    fig.tight_layout(pad = 0.2)
    curr_plots_path = output_dir+"/"+output_name+".pdf"
    fig.savefig(curr_plots_path)
    return xi_h_arr, xi_e_arr
    
    
# ==========================
# ======= Box plots ========
# ==========================

def make_boxplots(sample_size, iterations, output_file_path, given_type, distribution_name, gamma_grid, bootstrap_flag, t_bootstrap, r_bootstrap, diagn_plots, eps_stop, verbose):
    
    output_dir = os.path.dirname(os.path.realpath(output_file_path))
    output_name = os.path.splitext(os.path.basename(output_file_path))[0]
    
    # perform adjusted Hill estimation
    k_h_arr = {}
    xi_h_arr= {}
    k_h_star = {}
    xi_h_star = {}
    x1_h_arr = {}
    n1_h_amse = {}
    k1_h = {}
    max_h_index1 = {}
    x2_h_arr = {}
    n2_h_amse = {}
    k2_h = {}
    max_h_index2 = {}
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        k_h_arr[gamma_string] = []
        xi_h_arr[gamma_string] = []
        k_h_star[gamma_string] = []
        xi_h_star[gamma_string] = []
        x1_h_arr[gamma_string] = []
        n1_h_amse[gamma_string] = []
        k1_h[gamma_string] = []
        max_h_index1[gamma_string] = []
        x2_h_arr[gamma_string] = []
        n2_h_amse[gamma_string] = []
        k2_h [gamma_string]= []
        max_h_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(np.abs(given_type.rvs(gamma, size = sample_size)))[::-1]
            if verbose:
                print(f"Calculating adjusted Hill... iteration {iteration}")
            t1 = time.time()
            hill_results = hill_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Hill), iteration {iteration}: {t2-t1}")
            k_h_arr[gamma_string].append(hill_results[0])
            xi_h_arr[gamma_string].append(hill_results[1])
            k_h_star[gamma_string].append(hill_results[2])
            xi_h_star[gamma_string].append(hill_results[3])
            x1_h_arr[gamma_string].append(hill_results[4])
            n1_h_amse[gamma_string].append(hill_results[5])
            k1_h[gamma_string].append(hill_results[6])
            max_h_index1[gamma_string].append(hill_results[7])
            x2_h_arr[gamma_string].append(hill_results[8])
            n2_h_amse[gamma_string].append(hill_results[9])
            k2_h[gamma_string].append(hill_results[10])
            max_h_index2[gamma_string].append(hill_results[11])
            
    # perform expectile estimation
    k_e_arr = {}
    xi_e_arr = {}
    k_e_star = {}
    xi_e_star = {}
    x1_e_arr = {}
    n1_e_amse = {}
    k1_e = {}
    max_e_index1 = {}
    x2_e_arr = {}
    n2_e_amse = {}
    k2_e = {}
    max_e_index2 = {}
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        k_e_arr[gamma_string] = []
        xi_e_arr[gamma_string] = []
        k_e_star[gamma_string] = []
        xi_e_star[gamma_string] = []
        x1_e_arr[gamma_string] = []
        n1_e_amse[gamma_string] = []
        k1_e[gamma_string] = []
        max_e_index1[gamma_string] = []
        x2_e_arr[gamma_string] = []
        n2_e_amse[gamma_string] = []
        k2_e[gamma_string]= []
        max_e_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(np.abs(given_type.rvs(gamma, size = sample_size)))[::-1]
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            hill_results = expectile_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            k_e_arr[gamma_string].append(hill_results[0])
            xi_e_arr[gamma_string].append(hill_results[1])
            k_e_star[gamma_string].append(hill_results[2])
            xi_e_star[gamma_string].append(hill_results[3])
            x1_e_arr[gamma_string].append(hill_results[4])
            n1_e_amse[gamma_string].append(hill_results[5])
            k1_e[gamma_string].append(hill_results[6])
            max_e_index1[gamma_string].append(hill_results[7])
            x2_e_arr[gamma_string].append(hill_results[8])
            n2_e_amse[gamma_string].append(hill_results[9])
            k2_e[gamma_string].append(hill_results[10])
            max_e_index2[gamma_string].append(hill_results[11])
    
    fig, ax = plt.subplots(3, 1, figsize =(15, 25))
    grid_data_h = []
    grid_data_e = []
    for index in gamma_grid:
        index_string = "{:.2f}".format(index)
        grid_data_e.append(np.array(xi_e_star[index_string]) - 1/index)
        grid_data_h.append(np.array(xi_h_star[index_string]) - 1/index)
    ax[0].boxplot(grid_data_e)
    ax[0].set_xlabel("Гамма", labelpad=10)
    ax[0].set_ylabel("Значение гаммы", labelpad=10)
    ax[0].set_xticklabels(gamma_grid)
    ax[0].set_title(f"Распределение {distribution_name}: зависимость экспек от истинного значения гамма")
        
    ax[1].boxplot(grid_data_h)
    ax[1].set_xlabel("Гамма", labelpad=10)
    ax[1].set_ylabel("Значение гаммы", labelpad=10)
    ax[1].set_xticklabels(gamma_grid)
    ax[1].set_title(f"Распределение {distribution_name}: зависимость экспек от истинного значения гамма")
        
    ax[2].boxplot(grid_data_h)
    ax[2].set_xlabel("Гамма", labelpad=10)
    ax[2].set_ylabel("Значение гаммы", labelpad=10)
    ax[2].set_xticklabels(gamma_grid)
    ax[2].set_title(f"Распределение {distribution_name}: зависимость экспек от истинного значения гамма")
    
    plt.savefig(f'Boxplots_{distribution_name}.pdf')
       
    

def data_for_first_table(sample_size, iterations, output_file_path, given_type, distribution_name, gamma_grid, bootstrap_flag, t_bootstrap, r_bootstrap, diagn_plots, eps_stop, verbose):
    hsteps = int(sample_size/10)
    
    output_dir = os.path.dirname(os.path.realpath(output_file_path))
    output_name = os.path.splitext(os.path.basename(output_file_path))[0]
    
    # perform adjusted Hill estimation
    Hill_class = DBS_result()
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        Hill_class.k_arr[gamma_string] = []
        Hill_class.xi_arr[gamma_string] = []
        Hill_class.k_star[gamma_string] = []
        Hill_class.xi_star[gamma_string] = []
        Hill_class.x1_arr[gamma_string] = []
        Hill_class.n1_amse[gamma_string] = []
        Hill_class.k1[gamma_string] = []
        Hill_class.max_index1[gamma_string] = []
        Hill_class.x2_arr[gamma_string] = []
        Hill_class.n2_amse[gamma_string] = []
        Hill_class.k2[gamma_string]= []
        Hill_class.max_index2[gamma_string] = []
        Hill_class.rho[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(np.abs(given_type.rvs(gamma, size = sample_size)))[::-1]
            if verbose:
                print(f"Calculating adjusted Hill... iteration {iteration}")
            t1 = time.time()
            hill_results = hill_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Hill), iteration {iteration}: {t2-t1}")
            Hill_class.k_arr[gamma_string].append(hill_results[0])
            Hill_class.xi_arr[gamma_string].append(hill_results[1])
            Hill_class.k_star[gamma_string].append(hill_results[2])
            Hill_class.xi_star[gamma_string].append(hill_results[3])
            Hill_class.x1_arr[gamma_string].append(hill_results[4])
            Hill_class.n1_amse[gamma_string].append(hill_results[5])
            Hill_class.k1[gamma_string].append(hill_results[6])
            Hill_class.max_index1[gamma_string].append(hill_results[7])
            Hill_class.x2_arr[gamma_string].append(hill_results[8])
            Hill_class.n2_amse[gamma_string].append(hill_results[9])
            Hill_class.k2[gamma_string].append(hill_results[10])
            Hill_class.max_index2[gamma_string].append(hill_results[11])
            Hill_class.rho[gamma_string].append(hill_results[12])
            
    # perform expectile estimation
    Expectile_class = DBS_result()
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        Expectile_class.k_arr[gamma_string] = []
        Expectile_class.xi_arr[gamma_string] = []
        Expectile_class.k_star[gamma_string] = []
        Expectile_class.xi_star[gamma_string] = []
        Expectile_class.x1_arr[gamma_string] = []
        Expectile_class.n1_amse[gamma_string] = []
        Expectile_class.k1[gamma_string] = []
        Expectile_class.max_index1[gamma_string] = []
        Expectile_class.x2_arr[gamma_string] = []
        Expectile_class.n2_amse[gamma_string] = []
        Expectile_class.k2[gamma_string]= []
        Expectile_class.max_index2[gamma_string] = []
        Expectile_class.rho[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(np.abs(given_type.rvs(gamma, size = sample_size)))[::-1]
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            expectile_results = expectile_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            while expectile_results == None:
                ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1]
                expectile_results = expectile_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            Expectile_class.k_arr[gamma_string].append(expectile_results[0])
            Expectile_class.xi_arr[gamma_string].append(expectile_results[1])
            Expectile_class.k_star[gamma_string].append(expectile_results[2])
            Expectile_class.xi_star[gamma_string].append(expectile_results[3])
            Expectile_class.x1_arr[gamma_string].append(expectile_results[4])
            Expectile_class.n1_amse[gamma_string].append(expectile_results[5])
            Expectile_class.k1[gamma_string].append(expectile_results[6])
            Expectile_class.max_index1[gamma_string].append(expectile_results[7])
            Expectile_class.x2_arr[gamma_string].append(expectile_results[8])
            Expectile_class.n2_amse[gamma_string].append(expectile_results[9])
            Expectile_class.k2[gamma_string].append(expectile_results[10])
            Expectile_class.max_index2[gamma_string].append(expectile_results[11])
            Expectile_class.rho[gamma_string].append(expectile_results[12])
    
    # perform moments estimation
    Moments_class = DBS_result()
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        Moments_class.k_arr[gamma_string] = []
        Moments_class.xi_arr[gamma_string] = []
        Moments_class.k_star[gamma_string] = []
        Moments_class.xi_star[gamma_string] = []
        Moments_class.x1_arr[gamma_string] = []
        Moments_class.n1_amse[gamma_string] = []
        Moments_class.k1[gamma_string] = []
        Moments_class.max_index1[gamma_string] = []
        Moments_class.x2_arr[gamma_string] = []
        Moments_class.n2_amse[gamma_string] = []
        Moments_class.k2[gamma_string]= []
        Moments_class.max_index2[gamma_string] = []
        Moments_class.rho[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(np.abs(given_type.rvs(gamma, size = sample_size)))[::-1]
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            moment_results = moments_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            Moments_class.k_arr[gamma_string].append(moment_results[0])
            Moments_class.xi_arr[gamma_string].append(moment_results[1])
            Moments_class.k_star[gamma_string].append(moment_results[2])
            Moments_class.xi_star[gamma_string].append(moment_results[3])
            Moments_class.x1_arr[gamma_string].append(moment_results[4])
            Moments_class.n1_amse[gamma_string].append(moment_results[5])
            Moments_class.k1[gamma_string].append(moment_results[6])
            Moments_class.max_index1[gamma_string].append(moment_results[7])
            Moments_class.x2_arr[gamma_string].append(moment_results[8])
            Moments_class.n2_amse[gamma_string].append(moment_results[9])
            Moments_class.k2[gamma_string].append(moment_results[10])
            Moments_class.max_index2[gamma_string].append(moment_results[11])
            Moments_class.rho[gamma_string].append(moment_results[12])
            
    return Hill_class, Expectile_class, Moments_class
    

def make_boxplots_for_translation_sns(sample_size, iterations, output_file_path, given_type, distribution_name, gamma, iterative_bias, bootstrap_flag, t_bootstrap, r_bootstrap, diagn_plots, eps_stop, verbose):
    hsteps = int(sample_size/10)
    
    output_dir = os.path.dirname(os.path.realpath(output_file_path))
    output_name = os.path.splitext(os.path.basename(output_file_path))[0]
    
    # perform adjusted Hill estimation
    k_h_arr = {}
    xi_h_arr= {}
    k_h_star = {}
    xi_h_star = {}
    x1_h_arr = {}
    n1_h_amse = {}
    k1_h = {}
    max_h_index1 = {}
    x2_h_arr = {}
    n2_h_amse = {}
    k2_h = {}
    max_h_index2 = {}
    for bias in iterative_bias:
        gamma_string = "{:.2f}".format(bias)
        k_h_arr[gamma_string] = []
        xi_h_arr[gamma_string] = []
        k_h_star[gamma_string] = []
        xi_h_star[gamma_string] = []
        x1_h_arr[gamma_string] = []
        n1_h_amse[gamma_string] = []
        k1_h[gamma_string] = []
        max_h_index1[gamma_string] = []
        x2_h_arr[gamma_string] = []
        n2_h_amse[gamma_string] = []
        k2_h [gamma_string]= []
        max_h_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1] + bias
            if verbose:
                print(f"Calculating adjusted Hill... iteration {iteration}")
            t1 = time.time()
            hill_results = hill_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Hill), iteration {iteration}: {t2-t1}")
            k_h_arr[gamma_string].append(hill_results[0])
            xi_h_arr[gamma_string].append(hill_results[1])
            k_h_star[gamma_string].append(hill_results[2])
            xi_h_star[gamma_string].append(hill_results[3])
            x1_h_arr[gamma_string].append(hill_results[4])
            n1_h_amse[gamma_string].append(hill_results[5])
            k1_h[gamma_string].append(hill_results[6])
            max_h_index1[gamma_string].append(hill_results[7])
            x2_h_arr[gamma_string].append(hill_results[8])
            n2_h_amse[gamma_string].append(hill_results[9])
            k2_h[gamma_string].append(hill_results[10])
            max_h_index2[gamma_string].append(hill_results[11])
            
    # perform expectile estimation
    k_e_arr = {}
    xi_e_arr = {}
    k_e_star = {}
    xi_e_star = {}
    x1_e_arr = {}
    n1_e_amse = {}
    k1_e = {}
    max_e_index1 = {}
    x2_e_arr = {}
    n2_e_amse = {}
    k2_e = {}
    max_e_index2 = {}
    for bias in iterative_bias:
        gamma_string = "{:.2f}".format(bias)
        k_e_arr[gamma_string] = []
        xi_e_arr[gamma_string] = []
        k_e_star[gamma_string] = []
        xi_e_star[gamma_string] = []
        x1_e_arr[gamma_string] = []
        n1_e_amse[gamma_string] = []
        k1_e[gamma_string] = []
        max_e_index1[gamma_string] = []
        x2_e_arr[gamma_string] = []
        n2_e_amse[gamma_string] = []
        k2_e [gamma_string]= []
        max_e_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1] + bias
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            hill_results = expectile_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            k_e_arr[gamma_string].append(hill_results[0])
            xi_e_arr[gamma_string].append(hill_results[1])
            k_e_star[gamma_string].append(hill_results[2])
            xi_e_star[gamma_string].append(hill_results[3])
            x1_e_arr[gamma_string].append(hill_results[4])
            n1_e_amse[gamma_string].append(hill_results[5])
            k1_e[gamma_string].append(hill_results[6])
            max_e_index1[gamma_string].append(hill_results[7])
            x2_e_arr[gamma_string].append(hill_results[8])
            n2_e_amse[gamma_string].append(hill_results[9])
            k2_e[gamma_string].append(hill_results[10])
            max_e_index2[gamma_string].append(hill_results[11])
    
    # perform moments estimation
    k_m_arr = {}
    xi_m_arr = {}
    k_m_star = {}
    xi_m_star = {}
    x1_m_arr = {}
    n1_m_amse = {}
    k1_m = {}
    max_m_index1 = {}
    x2_m_arr = {}
    n2_m_amse = {}
    k2_m = {}
    max_m_index2 = {}
    for bias in iterative_bias:
        gamma_string = "{:.2f}".format(bias)
        k_m_arr[gamma_string] = []
        xi_m_arr[gamma_string] = []
        k_m_star[gamma_string] = []
        xi_m_star[gamma_string] = []
        x1_m_arr[gamma_string] = []
        n1_m_amse[gamma_string] = []
        k1_m[gamma_string] = []
        max_m_index1[gamma_string] = []
        x2_m_arr[gamma_string] = []
        n2_m_amse[gamma_string] = []
        k2_m[gamma_string]= []
        max_m_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1] + bias
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            moment_results = moments_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            k_m_arr[gamma_string].append(moment_results[0])
            xi_m_arr[gamma_string].append(moment_results[1])
            k_m_star[gamma_string].append(moment_results[2])
            xi_m_star[gamma_string].append(moment_results[3])
            x1_m_arr[gamma_string].append(moment_results[4])
            n1_m_amse[gamma_string].append(moment_results[5])
            k1_m[gamma_string].append(moment_results[6])
            max_m_index1[gamma_string].append(moment_results[7])
            x2_m_arr[gamma_string].append(moment_results[8])
            n2_m_amse[gamma_string].append(moment_results[9])
            k2_m[gamma_string].append(moment_results[10])
            max_m_index2[gamma_string].append(moment_results[11])
            
    # perform moments estimation
    k_k_arr = {}
    xi_k_arr = {}
    k_k_star = {}
    xi_k_star = {}
    x1_k_arr = {}
    n1_k_amse = {}
    k1_k = {}
    max_k_index1 = {}
    x2_k_arr = {}
    n2_k_amse = {}
    k2_k = {}
    max_k_index2 = {}
    for bias in iterative_bias:
        gamma_string = "{:.2f}".format(bias)
        k_k_arr[gamma_string] = []
        xi_k_arr[gamma_string] = []
        k_k_star[gamma_string] = []
        xi_k_star[gamma_string] = []
        x1_k_arr[gamma_string] = []
        n1_k_amse[gamma_string] = []
        k1_k[gamma_string] = []
        max_k_index1[gamma_string] = []
        x2_k_arr[gamma_string] = []
        n2_k_amse[gamma_string] = []
        k2_k[gamma_string]= []
        max_k_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1] + bias
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            kernel_results = kernel_type_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False,
                                            hsteps = hsteps)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            k_k_arr[gamma_string].append(kernel_results[0])
            xi_k_arr[gamma_string].append(kernel_results[1])
            k_k_star[gamma_string].append(kernel_results[2])
            xi_k_star[gamma_string].append(kernel_results[3])
            x1_k_arr[gamma_string].append(kernel_results[4])
            n1_k_amse[gamma_string].append(kernel_results[5])
            k1_k[gamma_string].append(kernel_results[6])
            max_k_index1[gamma_string].append(kernel_results[7])
            x2_k_arr[gamma_string].append(kernel_results[8])
            n2_k_amse[gamma_string].append(kernel_results[9])
            k2_k[gamma_string].append(kernel_results[10])
            max_k_index2[gamma_string].append(kernel_results[11])
            
            
    liste = []
    listh = []
    listk = []
    listm = []
    for key in xi_e_star.keys():
        liste.append(pd.DataFrame(np.array(xi_e_star[key]) - 1/3, columns=[f'Expectile']).assign(Index = key))
    for key in xi_h_star.keys():
        listh.append(pd.DataFrame(np.array(xi_h_star[key]) - 1/3, columns=[f'Hill']).assign(Index = key))
    for key in xi_k_star.keys():
        listk.append(pd.DataFrame(np.array(xi_k_star[key]) - 1/3, columns=[f'Kernel']).assign(Index = key))
    for key in xi_m_star.keys():
        listm.append(pd.DataFrame(np.array(xi_m_star[key]) - 1/3, columns=[f'Moments']).assign(Index = key))


    sns.set(rc={'figure.figsize':(20,80)})
    fig, axes = plt.subplots(8, 1)

    exp = pd.concat(liste)      
    hill = pd.concat(listh)  
    mom = pd.concat(listm)  
    kern = pd.concat(listk)  
    common_table = pd.concat([exp, hill, mom, kern])
    result = pd.melt(common_table, id_vars=['Index'], var_name=['Number'])
    exp_hill = pd.melt(pd.concat([exp, hill]), id_vars=['Index'], var_name=['Number'])
    exp_kern = pd.melt(pd.concat([exp, kern]), id_vars=['Index'], var_name=['Number'])
    exp_mom = pd.melt(pd.concat([exp, mom]), id_vars=['Index'], var_name=['Number'])


    sns.boxplot(x="Index", y="value", hue="Number", data=result, ax=axes[0])
    sns.boxplot(x="Index", y="Expectile", data=exp, ax=axes[1])
    sns.boxplot(x="Index", y="Hill", data=hill, ax=axes[2])
    sns.boxplot(x="Index", y="Moments", data=mom, ax=axes[3])
    sns.boxplot(x="Index", y="Kernel", data=kern, ax=axes[4])
    sns.boxplot(x="Index", y="value", hue="Number", data=exp_hill, ax=axes[5])
    sns.boxplot(x="Index", y="value", hue="Number", data=exp_mom, ax=axes[6])
    sns.boxplot(x="Index", y="value", hue="Number", data=exp_kern, ax=axes[7])
    for i in range(8):
        axes[i].set_xticklabels(rotation=45, labels = xi_e_star.keys())
    axes[0].title.set_text('Совокупное распределение оценок')
    axes[1].title.set_text('Экспектильная оценка')
    axes[2].title.set_text('Оценка Хилла')
    axes[3].title.set_text('Оценка моментов')
    axes[4].title.set_text('Ядерная оценка')
    axes[5].title.set_text('Экспектильная оценка и оценка Хилла')
    axes[6].title.set_text('Экспектильная оценка и оценка моментов')
    axes[7].title.set_text('Экспектильная и ядерная оценки')
    for i in range(8):
        sns.lineplot(x = [-0.5,len(xi_e_star.keys()) - 0.5], y = 0, color = "black", lw = 1, ax=axes[i])
    plt.savefig(f'Boxplots_translation_{distribution_name}.pdf')
    
    
def make_boxplots_sns(sample_size, iterations, output_file_path, given_type, distribution_name, gamma_grid, bootstrap_flag, t_bootstrap, r_bootstrap, diagn_plots, eps_stop, verbose):
    hsteps = int(sample_size/10)
    
    output_dir = os.path.dirname(os.path.realpath(output_file_path))
    output_name = os.path.splitext(os.path.basename(output_file_path))[0]
    
    # perform adjusted Hill estimation
    k_h_arr = {}
    xi_h_arr= {}
    k_h_star = {}
    xi_h_star = {}
    x1_h_arr = {}
    n1_h_amse = {}
    k1_h = {}
    max_h_index1 = {}
    x2_h_arr = {}
    n2_h_amse = {}
    k2_h = {}
    max_h_index2 = {}
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        k_h_arr[gamma_string] = []
        xi_h_arr[gamma_string] = []
        k_h_star[gamma_string] = []
        xi_h_star[gamma_string] = []
        x1_h_arr[gamma_string] = []
        n1_h_amse[gamma_string] = []
        k1_h[gamma_string] = []
        max_h_index1[gamma_string] = []
        x2_h_arr[gamma_string] = []
        n2_h_amse[gamma_string] = []
        k2_h [gamma_string]= []
        max_h_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1]
            if verbose:
                print(f"Calculating adjusted Hill... iteration {iteration}")
            t1 = time.time()
            hill_results = hill_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Hill), iteration {iteration}: {t2-t1}")
            k_h_arr[gamma_string].append(hill_results[0])
            xi_h_arr[gamma_string].append(hill_results[1])
            k_h_star[gamma_string].append(hill_results[2])
            xi_h_star[gamma_string].append(hill_results[3] - 1/gamma)
            x1_h_arr[gamma_string].append(hill_results[4])
            n1_h_amse[gamma_string].append(hill_results[5])
            k1_h[gamma_string].append(hill_results[6])
            max_h_index1[gamma_string].append(hill_results[7])
            x2_h_arr[gamma_string].append(hill_results[8])
            n2_h_amse[gamma_string].append(hill_results[9])
            k2_h[gamma_string].append(hill_results[10])
            max_h_index2[gamma_string].append(hill_results[11])
            
    # perform expectile estimation
    k_e_arr = {}
    xi_e_arr = {}
    k_e_star = {}
    xi_e_star = {}
    x1_e_arr = {}
    n1_e_amse = {}
    k1_e = {}
    max_e_index1 = {}
    x2_e_arr = {}
    n2_e_amse = {}
    k2_e = {}
    max_e_index2 = {}
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        k_e_arr[gamma_string] = []
        xi_e_arr[gamma_string] = []
        k_e_star[gamma_string] = []
        xi_e_star[gamma_string] = []
        x1_e_arr[gamma_string] = []
        n1_e_amse[gamma_string] = []
        k1_e[gamma_string] = []
        max_e_index1[gamma_string] = []
        x2_e_arr[gamma_string] = []
        n2_e_amse[gamma_string] = []
        k2_e [gamma_string]= []
        max_e_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1]
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            hill_results = expectile_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            k_e_arr[gamma_string].append(hill_results[0])
            xi_e_arr[gamma_string].append(hill_results[1])
            k_e_star[gamma_string].append(hill_results[2])
            xi_e_star[gamma_string].append(hill_results[3] - 1/gamma)
            x1_e_arr[gamma_string].append(hill_results[4])
            n1_e_amse[gamma_string].append(hill_results[5])
            k1_e[gamma_string].append(hill_results[6])
            max_e_index1[gamma_string].append(hill_results[7])
            x2_e_arr[gamma_string].append(hill_results[8])
            n2_e_amse[gamma_string].append(hill_results[9])
            k2_e[gamma_string].append(hill_results[10])
            max_e_index2[gamma_string].append(hill_results[11])
    
    # perform moments estimation
    k_m_arr = {}
    xi_m_arr = {}
    k_m_star = {}
    xi_m_star = {}
    x1_m_arr = {}
    n1_m_amse = {}
    k1_m = {}
    max_m_index1 = {}
    x2_m_arr = {}
    n2_m_amse = {}
    k2_m = {}
    max_m_index2 = {}
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        k_m_arr[gamma_string] = []
        xi_m_arr[gamma_string] = []
        k_m_star[gamma_string] = []
        xi_m_star[gamma_string] = []
        x1_m_arr[gamma_string] = []
        n1_m_amse[gamma_string] = []
        k1_m[gamma_string] = []
        max_m_index1[gamma_string] = []
        x2_m_arr[gamma_string] = []
        n2_m_amse[gamma_string] = []
        k2_m[gamma_string]= []
        max_m_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1]
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            moment_results = moments_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            k_m_arr[gamma_string].append(moment_results[0])
            xi_m_arr[gamma_string].append(moment_results[1])
            k_m_star[gamma_string].append(moment_results[2])
            xi_m_star[gamma_string].append(moment_results[3] - 1/gamma)
            x1_m_arr[gamma_string].append(moment_results[4])
            n1_m_amse[gamma_string].append(moment_results[5])
            k1_m[gamma_string].append(moment_results[6])
            max_m_index1[gamma_string].append(moment_results[7])
            x2_m_arr[gamma_string].append(moment_results[8])
            n2_m_amse[gamma_string].append(moment_results[9])
            k2_m[gamma_string].append(moment_results[10])
            max_m_index2[gamma_string].append(moment_results[11])
            
    # perform moments estimation
    k_k_arr = {}
    xi_k_arr = {}
    k_k_star = {}
    xi_k_star = {}
    x1_k_arr = {}
    n1_k_amse = {}
    k1_k = {}
    max_k_index1 = {}
    x2_k_arr = {}
    n2_k_amse = {}
    k2_k = {}
    max_k_index2 = {}
    for gamma in gamma_grid:
        gamma_string = "{:.2f}".format(gamma)
        k_k_arr[gamma_string] = []
        xi_k_arr[gamma_string] = []
        k_k_star[gamma_string] = []
        xi_k_star[gamma_string] = []
        x1_k_arr[gamma_string] = []
        n1_k_amse[gamma_string] = []
        k1_k[gamma_string] = []
        max_k_index1[gamma_string] = []
        x2_k_arr[gamma_string] = []
        n2_k_amse[gamma_string] = []
        k2_k[gamma_string]= []
        max_k_index2[gamma_string] = []
        for iteration in range(iterations):
            ordered_data = np.sort(given_type.rvs(gamma, size = sample_size))[::-1]
            if verbose:
                print(f"Calculating expectile estimator... iteration {iteration}")
            t1 = time.time()
            kernel_results = kernel_type_estimator(ordered_data,
                                            bootstrap = bootstrap_flag,
                                            t_bootstrap = t_bootstrap,
                                            r_bootstrap = r_bootstrap,
                                            diagn_plots = diagn_plots,
                                            eps_stop = eps_stop, 
                                            verbose = False,
                                            hsteps = hsteps)
            t2 =time.time()
            if verbose:
                print(f"Elapsed time (Expectile estimator), iteration {iteration}: {t2-t1}")
            k_k_arr[gamma_string].append(kernel_results[0])
            xi_k_arr[gamma_string].append(kernel_results[1])
            k_k_star[gamma_string].append(kernel_results[2])
            xi_k_star[gamma_string].append(kernel_results[3] - 1/gamma)
            x1_k_arr[gamma_string].append(kernel_results[4])
            n1_k_amse[gamma_string].append(kernel_results[5])
            k1_k[gamma_string].append(kernel_results[6])
            max_k_index1[gamma_string].append(kernel_results[7])
            x2_k_arr[gamma_string].append(kernel_results[8])
            n2_k_amse[gamma_string].append(kernel_results[9])
            k2_k[gamma_string].append(kernel_results[10])
            max_k_index2[gamma_string].append(kernel_results[11])
            
            
    liste = []
    listh = []
    listk = []
    listm = []
    for key in xi_e_star.keys():
        liste.append(pd.DataFrame(np.array(xi_e_star[key]), columns=[f'Expectile']).assign(Index = key))
        listh.append(pd.DataFrame(np.array(xi_h_star[key]), columns=[f'Hill']).assign(Index = key))
        listk.append(pd.DataFrame(np.array(xi_k_star[key]), columns=[f'Kernel']).assign(Index = key))
        listm.append(pd.DataFrame(np.array(xi_m_star[key]), columns=[f'Moments']).assign(Index = key))


    sns.set(rc={'figure.figsize':(20,80)})
    fig, axes = plt.subplots(8, 1)

    exp = pd.concat(liste)      
    hill = pd.concat(listh)  
    mom = pd.concat(listm)  
    kern = pd.concat(listk)  
    common_table = pd.concat([exp, hill, mom, kern])
    result = pd.melt(common_table, id_vars=['Index'], var_name=['Number'])
    exp_hill = pd.melt(pd.concat([exp, hill]), id_vars=['Index'], var_name=['Number'])
    exp_kern = pd.melt(pd.concat([exp, kern]), id_vars=['Index'], var_name=['Number'])
    exp_mom = pd.melt(pd.concat([exp, mom]), id_vars=['Index'], var_name=['Number'])


    sns.boxplot(x="Index", y="value", hue="Number", data=result, ax=axes[0])
    sns.boxplot(x="Index", y="Expectile", data=exp, ax=axes[1])
    sns.boxplot(x="Index", y="Hill", data=hill, ax=axes[2])
    sns.boxplot(x="Index", y="Moments", data=mom, ax=axes[3])
    sns.boxplot(x="Index", y="Kernel", data=kern, ax=axes[4])
    sns.boxplot(x="Index", y="value", hue="Number", data=exp_hill, ax=axes[5])
    sns.boxplot(x="Index", y="value", hue="Number", data=exp_mom, ax=axes[6])
    sns.boxplot(x="Index", y="value", hue="Number", data=exp_kern, ax=axes[7])
    for i in range(8):
        axes[i].set_xticklabels(rotation=45, labels = xi_e_star.keys())
    axes[0].title.set_text('Совокупное распределение оценок')
    axes[1].title.set_text('Экспектильная оценка')
    axes[2].title.set_text('Оценка Хилла')
    axes[3].title.set_text('Оценка моментов')
    axes[4].title.set_text('Ядерная оценка')
    axes[5].title.set_text('Экспектильная оценка и оценка Хилла')
    axes[6].title.set_text('Экспектильная оценка и оценка моментов')
    axes[7].title.set_text('Экспектильная и ядерная оценки')
    for i in range(8):
        sns.lineplot(x = [-0.5,len(xi_e_star.keys()) - 0.5], y = 0, color = "black", lw = 1, ax=axes[i])
    plt.savefig(f'Boxplots_{distribution_name}.pdf')
        
            

# ==========================
# ========== Main ==========
# ==========================

def main():
    #ignore warnings other than explicit ones
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description =
                                     "Script to compute tail index estimates\
                                     for a provided dataset.")
    parser.add_argument("sequence_file_path",
                        help = "Path to a data sequence.", type = str)
    parser.add_argument("output_file_path",
                        help = "Output path for plots. Use either PDF or\
                        PNG format.", type = str)
    parser.add_argument("--nbins",
                        help = "Number of bins for degree\
                        distribution (default = 30)", type = int,
                        default = 30)
    parser.add_argument("--rsmooth",
                        help = "Smoothing parameter for smooth Hill estimator\
                        (default = 2)",
                        type = int, default = 2)
    parser.add_argument("--alphakernel",
                        help = "Alpha parameter used for kernel-type estimator.\
                        Should be greater than 0.5 (default = 0.6).",
                        type = float, default = 0.6)
    parser.add_argument("--hsteps",
                        help = "Parameter to select number of bandwidth\
                        steps for kernel-type estimator, (default = 200).",
                        type = int, default = 200)
    parser.add_argument("--noise",
                        help = "Switch on/off uniform noise in range\
                        [-5*10^(-p), 5*10^(-p)] that is added to each\
                        data point. Used for integer-valued sequences\
                        with p = 1 (default = 1).",
                        type = int, default = 1)
    parser.add_argument("--pnoise",
                        help = "Uniform noise parameter corresponding to\
                        the rounding error of the data sequence. For integer\
                        values it equals to 1. (default = 1).",
                        type = int, default = 1)
    parser.add_argument("--bootstrap",
                        help = "Flag to switch on/off double-bootstrap\
                        algorithm for defining optimal order statistic\
                        of Hill, moments and kernel-type estimators.\
                        (default = 1)",
                        type = int, default = 1)
    parser.add_argument("--tbootstrap",
                        help = "Fraction of bootstrap samples in the 2nd\
                        bootstrap defined as n*tbootstrap, i.e., for\
                        n*0.5 a n/2 is the size of a single bootstrap\
                        sample (default = 0.5).",
                        type = float, default = 0.5)
    parser.add_argument("--rbootstrap",
                        help = "Number of bootstrap resamplings used in\
                        double-bootstrap. Note that each sample results\
                        are stored in an array, so be careful about the\
                        memory (default = 500).",
                        type = int, default = 500)
    parser.add_argument("--amseborder",
                        help = "Upper bound for order statistic to consider\
                        for double-bootstrap AMSE minimizer.\
                        Entries that are smaller or equal to the border value\
                        are ignored during AMSE minimization (default = 1).",
                        type = float, default = 1.0)
    parser.add_argument("--theta1",
                        help = "Lower bound of plotting range, defined as\
                        k_min = ceil(n^theta1), (default = 0.01).\
                        Overwritten if plots behave badly within the range.",
                        type = float, default = 0.01)
    parser.add_argument("--theta2",
                        help = "Upper bound of plotting range, defined as\
                        k_max = floor(n^theta2), (default = 0.99).\
                        Overwritten if plots behave badly within the range.",
                        type = float, default = 0.99)
    parser.add_argument("--diagplots",
                        help = "Flag to switch on/off plotting AMSE statistics\
                        for Hill/moments/kernel-type double-bootstrap algorithm.\
                        Used for diagnostics when double-bootstrap provides unstable\
                        results. Can be used to find proper amseborder parameter.\
                        (default = 0).",
                        type = int, default = 0)
    parser.add_argument("--verbose",
                        help = "Verbosity of bootstrap procedure.\
                        (default = 0).",
                        type = int, default = 0)
    parser.add_argument("--savedata",
                        help = "Flag to save data files in the directory\
                        with plots.\
                        (default = 0)",
                        type = int, default = 0)
    parser.add_argument("--delimiter",
                        help = "Delimiter used in the input file.\
                        Options are: whitespace, tab, comma, semicolon.",
                        type = str, default = "whitespace")
    args = parser.parse_args()

    # check arguments for consistency
    if args.nbins <= 0:
        parser.error("Number of bins should be greater than 0.")
    if args.rsmooth < 2:
        parser.error("r_smooth should be greater than 1.")
    if args.alphakernel <= 0.5:
        parser.error("alpha of kernel estimator should be grater than 0.5.")
    if args.hsteps <= 0:
        parser.error("hsteps should be greater than 0.")
    if args.noise != 0 and args.noise != 1:
        parser.error("noise flag should be 0 or 1.")
    if args.pnoise < 0:
        parser.error("pnoise parameter should be greater or equal to 0.")
    if args.bootstrap != 0 and args.bootstrap != 1:
        parser.error("bootstrap flag should be 0 or 1.")
    if args.tbootstrap <= 0.0 or args.tbootstrap >= 1.0:
        parser.error("tbootstrap should be in range (0, 1).")
    if args.rbootstrap <= 0:
        parser.error("Number of bootstrap resamples should be greater than 0.")
    if args.amseborder <= 0.0:
        parser.error("amseborder should be greater than 0.")
    if args.diagplots != 0 and args.diagplots != 1:
        parser.error("diagplots flag should be 0 or 1.")
    if args.verbose != 0 and args.verbose != 1:
        parser.error("verbose flag should be 0 or 1.")
    if args.savedata != 0 and args.savedata != 1:
        parser.error("savedata flag should be 0 or 1.")
    if args.theta1 < 0.0 or args.theta1 > 1.0:
        parser.error("Theta parameters should be in [0,1] range, where theta1 < theta2.")
    if args.theta2 < 0.0 or args.theta2 > 1.0:
        parser.error("Theta parameters should be in [0,1] range, where theta1 < theta2.")
    if args.theta2 <= args.theta1:
        parser.error("Theta parameters should be in [0,1] range, where theta1 < theta2.")
    if args.delimiter not in set(['whitespace', 'tab', 'comma', 'semicolon']):
        parser.error("Delimiter provided is not supported.")

    number_of_bins = args.nbins
    r_smooth = args.rsmooth
    alpha = args.alphakernel
    hsteps = args.hsteps
    if args.noise == 1:
        noise_flag = True
    else:
        noise_flag = False
    p_noise = args.pnoise
    if args.bootstrap == 1:
        bootstrap_flag = True
    else:
        bootstrap_flag = False
    t_bootstrap = args.tbootstrap
    r_bootstrap = args.rbootstrap
    amse_border = args.amseborder
    if args.diagplots == 1:
        diagnostic_plots_flag = True
    else:
        diagnostic_plots_flag = False

    if args.verbose == 1:
        verbose = True
    else:
        verbose = False

    if args.delimiter == "whitespace":
        delimiter = " "
    elif args.delimiter == "tab":
        delimiter = "\t"
    elif args.delimiter == "comma":
        delimiter = ","
    elif args.delimiter == "semicolon":
        delimiter = ";"

    # check for number of entries
    N = 0
    with open(args.sequence_file_path, "r") as f:
        for line in f:
            degree, count = line.strip().split(delimiter)
            N += int(count)
    print("========== Tail Index Estimation ==========")
    print("Number of data entries: %i" % N)
    ordered_data = np.zeros(N)
    current_index = 0      
    with open(args.sequence_file_path, "r") as f:
        for line in f:
            degree, count = line.strip().split(delimiter)
            ordered_data[current_index:current_index + int(count)] = float(degree)
            current_index += int(count)

    #enforce minimization boundary to the order statistics larger than border value
    eps_stop = 1 - float(len(ordered_data[np.where(ordered_data <= amse_border)]))\
                   /len(ordered_data)
    print("========================")
    print("Selected AMSE border value: %0.4f"%amse_border) 
    print("Selected fraction of order statistics boundary for AMSE minimization: %0.4f"%eps_stop)
    print("========================")
    make_plots(ordered_data, args.output_file_path, number_of_bins,
               r_smooth, alpha, hsteps, bootstrap_flag, t_bootstrap,
               r_bootstrap, diagnostic_plots_flag, eps_stop, 
               args.theta1, args.theta2, verbose, noise_flag,
               p_noise, args.savedata)


if __name__ == '__main__':
    
    t1 = time.time()
    main()
    t2 = time.time()
    print("Elapsed time (total):", t2-t1)


    

    
