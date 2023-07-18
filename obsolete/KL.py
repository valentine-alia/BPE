import sys
minp = sys.float_info.min
import numpy as np
import scipy.stats as scs

# how peaked to make gaussian
sigma_small = 0.0001
# gaussian to act like dirac delta - change to whatever dimension you need, and the mean should be the true values of params
true_posterior = scs.multivariate_normal(mean = np.array([0.6,10]), cov = np.array([[sigma_small,0],[0,sigma_small]]))
# not sure how to do this for non "delta" like thing 

## These adapted from PyVBMC source code
def truth_first_KL(true_posterior,vp2, N = int(1e4)):  # KL[true posterior || vp2]
    truth_samples = true_posterior.rvs(size = N)
    truth_densities = true_posterior.pdf(truth_samples)
    vp2_samples = vp2.pdf(truth_samples)
    truth_densities[truth_densities == 0 | np.isinf(truth_densities)] = 1.0
    vp2_samples[vp2_samples == 0 | np.isinf(vp2_samples)] = minp
    KL = -np.mean(np.log(vp2_samples) - np.log(truth_densities))
    return np.maximum(0, KL)

def truth_second_KL(true_posterior,vp2, N = int(1e4)): # KL[vp2 || true posterior] (DON'T USE THIS ONE)
    vp2_samples, _ = vp2.sample(N, True, True)
    truth_densities = true_posterior.pdf(vp2_samples)
    vp2_samples = vp2.pdf(vp2_samples, True)
    truth_densities[truth_densities == 0 | np.isinf(truth_densities)] = minp
    vp2_samples[vp2_samples == 0 | np.isinf(vp2_samples)] = 1.0
    KL = -np.mean(np.log(truth_densities) - np.log(vp2_samples))
    return np.maximum(0, KL)

def KL(vp1,vp2): # KL[vp1 || vp2]
    vp1_samples, _ = vp1.sample(N, True, True)
    vp1_densities = vp1.pdf(vp1_samples, True)
    vp2_samples = vp2.pdf(vp1_samples)
    vp1_densities[vp1_densities == 0 | np.isinf(vp1_densities)] = 1.0
    vp2_samples[vp2_samples == 0 | np.isinf(vp2_samples)] = minp
    KL = -np.mean(np.log(vp2_samples) - np.log(vp1_densities))
    return np.maximum(0,KL)

print(truth_first_KL(true_posterior,two_body_vp1),truth_second_KL(true_posterior,two_body_vp1)) #,KL(two_body_vp1, two_body_vp2),KL(two_body_vp2, two_body_vp1))

# alternate way to calculate KL that assumes everything is gaussian but is way faster
from pyvbmc.stats import kl_div_mvn
N = int(1e5)
truth_densitiesmu, truth_densitiessigma = np.atleast_2d(true_posterior.mean), true_posterior.cov
vp2_samplesmu, vp2_samplessigma = two_body_vp1.moments(N,orig_flag=True, cov_flag=True)

kls = kl_div_mvn(truth_densitiesmu, truth_densitiessigma, vp2_samplesmu, vp2_samplessigma) # ASSUMES NORMAL DISTRIBUTION

print("KL[true posterior||vp2] = ",truth_first_KL(true_posterior,two_body_vp1), "KL[true posterior||vp2], using gaussian assumption =", kls[0])