"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma  

def generate_data(mu=1.0, tau=0.5, N=100) -> np.ndarray:
    D = np.random.normal(loc=mu, scale=1/np.sqrt(tau), size=N)
    return D

def calc_log_joint(x: np.ndarray, mu: float, tau: float,
              mu_0: float, beta_0: float, a_0: float, b_0: float) -> float:
    EPS = 1e-12
    #tau = max(tau, EPS)
    
    N = len(x)
    log_likelihood = N * 1/2 * np.log(tau) - N * 1/2 * np.log(2*np.pi) - tau/2 * np.sum((x - mu)**2)
    prior_mu = 1/2 * np.log(beta_0*tau) - 1/2 * np.log(2*np.pi) - beta_0*tau/2 * (mu - mu_0)**2
    prior_tau = a_0 * np.log(b_0) - gammaln(a_0) + (a_0 - 1) * np.log(tau) - b_0 * tau

    return log_likelihood + prior_mu + prior_tau

def calc_log_q_mu(mu: float, mean: float, log_var: float) -> float:
    sigma2 = np.exp(log_var)
    log_q_mu = -1/2 * np.log(2*np.pi*sigma2) - (mu - mean)**2 / (2 * sigma2)
    return log_q_mu

def calc_log_q_tau(a: float, b: float, tau: float) -> float:
    EPS = 1e-12
    #a = max(a, EPS)
    #b = max(b, EPS)
    #tau = max(tau, EPS)
    log_q_tau = a * np.log(b) - gammaln(a) + (a - 1) * np.log(tau) - b*tau
    return log_q_tau

def calc_score_q_mu(mu_sample: float, log_var: float, mean: float) -> tuple[float, float]:
    sigma2 = np.exp(log_var)
    score_m = (mu_sample - mean) / sigma2
    score_var = -1/2 + (mu_sample - mean)**2 / (2*sigma2) 
    return score_m, score_var

def calc_score_q_tau(a: float, b: float, tau_sample: float) -> tuple[float, float]:
    EPS = 1e-12
    #a = max(a, EPS)
    #b = max(b, EPS)
    #tau = max(tau_sample, EPS)
    score_a = np.log(b) - digamma(a) + np.log(tau_sample)
    score_b = a / b - tau_sample
    return score_a, score_b

def calc_robbins_monroe(t: int, alpha_start: float = 0.01, alpha_decay: float = 0.6) -> float:
    return alpha_start / (t + 1)**alpha_decay

def plot_elbo_expectation_mu_tau(elbo_list: list[float], E_mu: list[float], E_tau: list[float], save=False) -> None:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(elbo_list)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO over iterations (BBVI)")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(E_mu, label="E_q[mu]")
    plt.plot(E_tau, label="E_q[tau]")
    plt.xlabel("Iteration")
    plt.ylabel("Expectation")
    plt.legend()
    plt.title("E_q[mu] and E_q[tau] over iterations")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig("bbvi_results.png", dpi=300)
    plt.show()

def bbvi(x: np.ndarray, debug=False):
    # hyperparams
    max_iter = 1000
    tol = 1e-4
    S = 50  # Increased sample size for lower variance

    # prior parameters
    mu_0 = 1.0
    beta_0 = 0.1
    a_0 = 1.0
    b_0 = 2.0

    # Initialize with reasonable values
    mean = 1.0 #np.mean(x)  # Better initialization
    log_var = 0.0 #np.log(np.var(x))
    a = 1.0
    b = 1.0

    # save history
    elbo_list = []
    E_mu_list = []
    E_tau_list = []

    for t in range(max_iter):
        # Draw S samples
        std = np.exp(0.5 * log_var)
        mu_samples = np.random.normal(loc=mean, scale=std, size=S)
        tau_samples = np.random.gamma(shape=a, scale=1/b, size=S)

        # Initialize gradients
        grad_mean = 0.0
        grad_log_var = 0.0
        grad_a = 0.0
        grad_b = 0.0
        elbo_est = 0.0

        # Accumulate over samples
        for s in range(S):
            mu_s = mu_samples[s]
            tau_s = tau_samples[s]

            log_joint = calc_log_joint(x, mu_s, tau_s, mu_0, beta_0, a_0, b_0)
            log_q = calc_log_q_mu(mu_s, mean, log_var) + calc_log_q_tau(a, b, tau_s)

            f = log_joint - log_q

            score_mean, score_log_var = calc_score_q_mu(mu_s, log_var, mean)
            score_a, score_b = calc_score_q_tau(a, b, tau_s)

            # FIX: Accumulate gradients correctly
            grad_mean += score_mean * f
            grad_log_var += score_log_var * f
            grad_a += score_a * f
            grad_b += score_b * f
            elbo_est += f
        
        # Average over samples
        grad_mean /= S
        grad_log_var /= S
        grad_a /= S
        grad_b /= S
        elbo_est /= S
        
        # Get learning rate
        rho = calc_robbins_monroe(t)

        # Save old parameters
        lmbda_old = np.array([mean, log_var, a, b])

        # Update with gradient ascent
        mean += rho * grad_mean
        log_var += rho * grad_log_var
        
        # FIX: Enforce stricter constraints on Gamma parameters
        #a = max(a + rho * grad_a, 0.1)  # Keep a away from 0
        #b = max(b + rho * grad_b, 0.1)  # Keep b away from 0

        a += rho * grad_a
        b += rho * grad_b
        lmbda = np.array([mean, log_var, a, b])
        delta = np.max(np.abs(lmbda_old - lmbda))

        # Save history
        elbo_list.append(elbo_est)
        E_mu_list.append(mean)
        E_tau_list.append(a / b)

        if delta < tol and t > 100:
            print(f"Converged at iteration {t}, delta {delta:.6f}")
            break

        if debug and t % 100 == 0:
            print(f"Iter {t}: ELBO≈{elbo_est:.3f}, E[mu]≈{mean:.3f}, E[tau]≈{a/b:.3f}, step={rho:.5f}")

    history = [elbo_list, E_mu_list, E_tau_list]
    return mean, log_var, a, b, history


if __name__ == "__main__":
    #np.random.seed(42)
    
    # Generate data with known parameters
    true_mu = 1.0
    true_tau = 0.5
    x = generate_data(mu=true_mu, tau=true_tau, N=100)
    
    print(f"True parameters: mu={true_mu}, tau={true_tau}")
    print(f"Data: mean={np.mean(x):.3f}, var={np.var(x):.3f}")
    print("\nRunning BBVI...")
    
    mean, log_var, a, b, history = bbvi(x, debug=True)
    
    print(f"\nFinal results:")
    print(f"E[mu] = {mean:.3f}")
    print(f"Var[mu] = {np.exp(log_var):.3f}")
    print(f"E[tau] = {a/b:.3f}")
    print(f"Std[tau] = {np.sqrt(a/b**2):.3f}")
    
    plot_elbo_expectation_mu_tau(history[0], history[1], history[2])

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma  

EPS = 1e-12 # min value
MAX_VAL = 1e12   # max value

def generate_data(mu=1.0, tau=0.5, N=100) -> np.ndarray:
    # generate N normal data points with mean mu and var 1/tau
    D = np.random.normal(loc=mu, scale=1/np.sqrt(tau), size=N)
    return D

def calc_log_joint(x: np.ndarray, mu: float, tau: float,
              mu_0: float, beta_0: float, a_0: float, b_0: float) -> float:
    tau = max(tau, EPS)
    
    N = len(x)

    # compute the joint from the conditionals
    log_likelihood = N * 1/2 * np.log(tau) - N *  1/2 * np.log(2*np.pi) - tau/2 * np.sum((x - mu)**2)
    prior_mu = 1/2 * np.log(beta_0*tau) - 1/2 * np.log(2*np.pi) - beta_0*tau/2 * (mu - mu_0)**2
    prior_tau = a_0 * np.log(b_0) - gammaln(a_0) + (a_0 - 1) * np.log(tau) - b_0 * tau

    return log_likelihood + prior_mu + prior_tau

def calc_log_q_mu(mu: float, mean: float, log_var: float) -> float:

    sigma2 = np.exp(log_var)
    log_q_mu = -1/2 * np.log(2*np.pi*sigma2) -  (mu - mean)**2 / (2 * sigma2)

    return log_q_mu

def calc_log_q_tau(a: float, b: float, tau: float) -> float:
    a = max(a, EPS)
    b = max(b, EPS)
    tau = max(tau, EPS)

    log_q_tau = a * np.log(b) - gammaln(a) + (a - 1) * np.log(tau) - b*tau
    return log_q_tau

def calc_score_q_mu(mu_sample: float, log_var: float, mean: float) -> tuple[float, float]:

    sigma2 = np.exp(log_var)
    score_m = (mu_sample - mean) / sigma2
    score_var = - 1/2 + (mu_sample - mean)**2 / (2*sigma2) 
    return score_m, score_var

def calc_score_q_tau(a: float, b: float, tau_sample: float) -> tuple[float, float]:
    EPS = 1e-12
    a = max(a, EPS)
    b = max(b, EPS)

    score_a = np.log(b) - digamma(a) + np.log(tau_sample)
    score_b = a / b - tau_sample
    return score_a, score_b

def calc_robbins_monroe(t: int, alpha_start: float = 0.01, alpha_decay: float = 0.6) -> float:
    return alpha_start / (t + 1)**alpha_decay

def get_test_data(mu: float, tau: float):
    mu = 1
    tau = 0.5

    dataset_1 = generate_data(mu, tau, 10)
    dataset_2 = generate_data(mu, tau, 100)
    dataset_3 = generate_data(mu, tau, 1000)

    # Visulaize the datasets via histograms
    plt.figure(1)
    plt.hist(dataset_1)
    plt.title("Dataset 1 (N=10)")
    plt.savefig("dataset_1_hist.png", dpi=300)

    plt.figure(2)
    plt.hist(dataset_2)
    plt.title("Dataset 2 (N=100)")
    plt.savefig("dataset_2_hist.png", dpi=300)

    plt.figure(3)
    plt.hist(dataset_3)
    plt.title("Dataset 3 (N=1000)")
    plt.savefig("dataset_3_hist.png", dpi=300)

    # show figures
    plt.show()

def plot_elbo_expectation_mu_tau(elbo_list: list[float], E_mu: list[float], E_tau: list[float], save=False) -> None:

    plt.figure()
    plt.plot(elbo_list)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO over iterations (BBVI)")
    plt.tight_layout()
    plt.grid()

    if save:
        plt.savefig("elbo_bbvi.png", dpi=300)

    plt.show()

    plt.figure()
    plt.plot(E_mu, label="E_q[mu]")
    plt.plot(E_tau, label="E_q[tau]")
    plt.xlabel("Iteration")
    plt.ylabel("Expectation")
    plt.legend()
    plt.title("E_q[mu] and E_q[tau] over iterations")
    plt.tight_layout()
    plt.grid()

    if save:
        plt.savefig("eq_mu_tau_bbvi.png", dpi=300)
    plt.show()

def bbvi(x: np.ndarray, S = 10, max_iter = 2000, alpha_start = 0.01, alpha_decay=0.7, debug=False):

    # hyperparams
    tol = 1e-4

    # prior parameters
    mu_0 = 1.0
    beta_0 = 0.1
    a_0 = 1.0
    b_0 = 2.0

    # intialize mean, log_var, a, b
    mean = np.mean(x)
    log_var = np.log(np.var(x))
    a = a_0
    b = b_0

    # save history
    elbo_list = []
    E_mu_list = []
    E_tau_list = []

    for t in range(max_iter):
        # draw S samples
        std = np.exp(0.5 * log_var)
        mu_samples = np.random.normal(loc=mean, scale=std, size=S)
        tau_samples = np.random.gamma(shape=a, scale=1/b, size=S)

        # initalize gradients
        grad_mean = 0.0
        grad_log_var = 0.0
        grad_a = 0.0
        grad_b = 0.0
        elbo_est = 0.0

        for s in range(S):
            mu_s = mu_samples[s]
            tau_s = tau_samples[s]

            log_joint = calc_log_joint(x, mu_s, tau_s, mu_0, beta_0, a_0, b_0)
            log_q = calc_log_q_mu(mu_s, mean, log_var) + calc_log_q_tau(a, b, tau_s)

            f = log_joint - log_q

            score_mean, score_log_var = calc_score_q_mu(mu_s, log_var, mean)
            score_a, score_b = calc_score_q_tau(a, b, tau_s)

            grad_mean += score_mean * f
            grad_log_var += score_log_var * f
            grad_a += score_a * f
            grad_b += score_b * f
            elbo_est += f
        
        grad_mean /= S
        grad_log_var /= S
        grad_a /= S
        grad_b /= S
        elbo_est /= S
        
        rho = calc_robbins_monroe(t, alpha_start, alpha_decay)

        lmbda_old = np.array([mean, log_var, a, b])

        mean += rho * grad_mean
        log_var += rho * grad_log_var
        a = max(a + rho * grad_a, 1e-6)
        b = max(b + rho * grad_b, 1e-6)

        lmbda = np.array([mean, log_var, a, b])

        delta = np.max(np.abs(lmbda_old - lmbda))

        elbo_list.append(elbo_est)
        E_mu_list.append(mean)
        E_tau_list.append(a / b)

        if delta < tol:
            print(f"Converged at iteration {t}, delta {delta:.4f}")
            break

        if debug and t % 100 == 0:
            print(f"Iter {t}: ELBO≈{elbo_est:.3f}, E[mean]≈{mean:.3f}, E[tau]≈{a/b:.3f}, step={rho:.4f}")
        if np.isnan(elbo_est):
            return mean, log_var, a, b, history

        history = [elbo_list, E_mu_list, E_tau_list]

    return mean, log_var, a, b, history

if __name__ == "__main__":
    np.random.seed(10)
    # Generate data with known parameters
    true_mu = 1.0
    true_tau = 0.5
    x = generate_data(mu=true_mu, tau=true_tau, N=100)
    
    print(f"True parameters: mu={true_mu}, tau={true_tau}")    

    # safe params
    alpha_start = 0.001
    alpha_decay = 0.6
    S = 20
    max_iter = 1000

    print("running BBVI...")
    mean, log_var, a, b, history = bbvi(x, S, max_iter, alpha_start, alpha_decay, debug=True)  

    print(f"\nFinal results:")
    print(f"E[mu] = {mean:.3f}")
    #print(f"Var[mu] = {np.exp(log_var):.3f}")
    print(f"E[tau] = {a/b:.3f}")
    #print(f"Std[tau] = {np.sqrt(a/b**2):.3f}")

    plot_elbo_expectation_mu_tau(history[0], history[1], history[2], save=False)

#"""