# Reproduces Figure 1 (top row) from "Estimating Heterogeneous Treatment Effects by Combining Weak Instruments and Observational Data"

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18

matplotlib.use('Agg')

# Make figures directory
if not os.path.exists("figures"):
    os.makedirs("figures")

##################
# DGP Parameters #
##################
def sigma(x):
    return 1/(1+np.exp(-x))

def y_func(x, a, u):
    return (1 + a + x + 2*a*x + 0.5*x**2 + 0.75*a*x**2 + u)

def true_tau(x):
    return 0.75*x**2 + 2*x + 1

def run_experiment(i, n, X_test):
    np.random.seed(i)
    n_O = n
    n_E = n
    # Observational data
    X_O = np.random.normal(size=n_O, scale=1)
    A_O = np.random.binomial(1, 0.5, size=n_O)
    U_O = np.random.normal(X_O*(A_O-0.5), np.sqrt(1-(A_O-0.5)**2))
    eps_O = np.random.normal(size=n_O)/2
    Y_O = np.array([y_func(X_O[i], A_O[i], U_O[i]) for i in range(n_O)]) + eps_O
    
    # Experimental data
    X_E = np.random.normal(size=n_E, scale=1)
    Z_E = np.random.binomial(1, 0.5, size=n_E)
    gamma_X = np.vectorize(sigma)(2*X_E)
    C = np.random.binomial(1, gamma_X)
    A_star = np.random.binomial(1, 0.5, size=n_E)
    A_E = C * Z_E + (1-C) * A_star
    U_E = C*np.random.normal(size=n_E) + (1-C)*np.random.normal(X_E*(A_E-0.5), np.sqrt(1-(A_E-0.5)**2))
    eps_E = np.random.normal(size=n_E)/2
    Y_E = np.array([y_func(X_E[i], A_E[i], U_E[i]) for i in range(n_E)]) + eps_E
    
    # Learn observational tau
    mu1_model =  RandomForestRegressor(max_depth=5, min_samples_leaf=5)
    mu1_model.fit(X_O[A_O==1].reshape(-1,1), Y_O[A_O==1])
    mu0_model = RandomForestRegressor(max_depth=5, min_samples_leaf=5)
    mu0_model.fit(X_O[A_O==0].reshape(-1,1), Y_O[A_O==0])
    tau_O = mu1_model.predict(X_test.reshape(-1, 1)) - mu0_model.predict(X_test.reshape(-1, 1))
    tau_O_mse = mu1_model.predict(X_mse.reshape(-1, 1)) - mu0_model.predict(X_mse.reshape(-1, 1))
    
    # Learn experimental tau
    # Z models
    mu1_z_model =  RandomForestRegressor(max_depth=5, min_samples_leaf=5)
    mu1_z_model.fit(X_E[Z_E==1].reshape(-1,1), Y_E[Z_E==1])
    mu0_z_model = RandomForestRegressor(max_depth=5, min_samples_leaf=5)
    mu0_z_model.fit(X_E[Z_E==0].reshape(-1,1), Y_E[Z_E==0])
    # A models
    pi1_z_model = RandomForestClassifier(max_depth=3, min_samples_leaf=50)
    pi1_z_model.fit(X_E[Z_E==1].reshape(-1,1), A_E[Z_E==1])
    pi0_z_model = RandomForestClassifier(max_depth=3, min_samples_leaf=50)
    pi0_z_model.fit(X_E[Z_E==0].reshape(-1,1), A_E[Z_E==0])
    ### X_test
    delta_Y_X_test = mu1_z_model.predict(X_test.reshape(-1, 1)) - mu0_z_model.predict(X_test.reshape(-1, 1))
    gamma_X_test = np.maximum(np.maximum(pi1_z_model.predict_proba(X_test.reshape(-1, 1))[:, 1] - pi0_z_model.predict_proba(X_test.reshape(-1, 1))[:, 1], 0), 
                      np.ones(X_test.size)*0.1)
    tau_E = delta_Y_X_test / gamma_X_test
    
    # Learn extension
    tau_O_hat = mu1_model.predict(X_E.reshape(-1, 1)) - mu0_model.predict(X_E.reshape(-1, 1))
    gamma_X_hat = np.maximum(pi1_z_model.predict_proba(X_E.reshape(-1, 1))[:, 1] - pi0_z_model.predict_proba(X_E.reshape(-1, 1))[:, 1], 0)
    tilde_Y = 2*Y_E*Z_E - 2*Y_E*(1-Z_E) - gamma_X_hat*tau_O_hat
    tilde_X = X_E * gamma_X_hat
    lr = LinearRegression().fit(tilde_X.reshape(-1, 1), tilde_Y)
    
    return tau_O, tau_E, tau_O + lr.predict(X_test.reshape(-1, 1))

###################
# Run experiments #
###################
n_iter = 100
X_test = np.arange(-3, 3, 0.01)
X_mse = np.random.normal(size=1000, scale=1)
tau_O_results = np.empty((n_iter, X_test.size))
tau_E_results = np.empty((n_iter, X_test.size))
tau_corrected = np.empty((n_iter, X_test.size))
results = Parallel(n_jobs=-1, verbose=1)(delayed(run_experiment)(i, 5000, X_test) for i in range(n_iter))

################
# Plot results #
################
tau_O_results = np.array([result[0] for result in results])
tau_E_results = np.array([result[1] for result in results])
tau_corrected = np.array([result[2] for result in results])

# \tau_O plot
plt.figure(figsize=(6, 3))
plt.plot(X_test, tau_O_results.mean(axis=0), label=r"$\widehat{\tau}^O(x)$"+" \xb1 SE", color="C0", zorder=10)
plt.fill_between(X_test, tau_O_results.mean(axis=0) + np.std(tau_O_results, axis=0), 
                 tau_O_results.mean(axis=0) - np.std(tau_O_results, axis=0), color="C0", alpha=0.3)
plt.plot(X_test, np.vectorize(true_tau)(X_test), label=r"$\tau(x)$", color='black', ls='--', lw=1)
plt.xlabel("x")
plt.ylabel("Effect")
plt.ylim(-3.5, 16)
plt.xlim(-3, 3)
plt.legend()
plt.savefig("figures/tau_O_sim_parametric_extrapolation.pdf", dpi=200, bbox_inches="tight")

# \tau_E plot
plt.figure(figsize=(6, 3))
plt.plot(X_test, tau_E_results.mean(axis=0), label=r"$\widehat{\tau}^E(x)$"+ " \xb1 SE", color="C0", zorder=10)
plt.fill_between(X_test, tau_E_results.mean(axis=0) + np.std(tau_E_results, axis=0), 
                 tau_E_results.mean(axis=0) - np.std(tau_E_results, axis=0), color="C0", alpha=0.3)
plt.plot(X_test, np.vectorize(true_tau)(X_test), label=r"$\tau(x)$", color='black', ls='--', lw=1)
plt.xlabel("x")
plt.ylabel("Effect")
plt.ylim(-3.5, 16)
plt.xlim(-3, 3)
plt.legend()
plt.savefig("figures/tau_E_hat_sim_parametric_extrapolation.pdf", dpi=200, bbox_inches="tight")

# Algorithm 1 plot
plt.figure(figsize=(6, 3))
plt.plot(X_test, tau_corrected.mean(axis=0), label=r"$\widehat{\tau}(x)$"+ " \xb1 SE", color="C0", zorder=10)
plt.fill_between(X_test, tau_corrected.mean(axis=0) + np.std(tau_corrected, axis=0), 
                 tau_corrected.mean(axis=0) - np.std(tau_corrected, axis=0), color="C0", alpha=0.3)
plt.plot(X_test, np.vectorize(true_tau)(X_test), label=r"$\tau(x)$", color='black', ls='--', lw=1)
plt.xlabel("x")
plt.ylabel("Effect")
plt.ylim(-3.5, 16)
plt.xlim(-3, 3)
plt.legend()
plt.savefig("figures/tau_hat_sim_parametric_extrapolation.pdf", dpi=200, bbox_inches="tight")