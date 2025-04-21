import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import factorial

def calc_distribution_theorique(lambda_, mu, m, max_k):
    rho = lambda_ / (m * mu)
    sum1 = sum((lambda_/mu)**k / factorial(k) for k in range(m))
    sum2 = (lambda_/mu)**m / (factorial(m) * (1 - rho))
    pi0 = 1.0 / (sum1 + sum2)

    pi = []
    for k in range(max_k+1):
        if k < m:
            pik = ( (lambda_/mu)**k / factorial(k) ) * pi0
        else:
            pik = ( (lambda_/mu)**m / factorial(m) ) * (rho**(k-m)) * pi0
        pi.append(pik)
    return np.array(pi)

def simuler_mm_m(lambda_, mu, m, n_transitions, seed=42):
    """
    Simule une file M/M/m en temps continu.
    Retourne deux listes de longueur n_transitions+1 :
      - times : instants des transitions (t_0=0, ..., t_n)
      - states: nombre de clients après chaque transition
    """
    np.random.seed(seed)
    t = 0.0
    etat = 0
    times = [t]
    states = [etat]

    for _ in range(n_transitions):
        mu_i = mu * min(etat, m)
        rate_total = lambda_ + mu_i
        # temps de séjour
        dt = np.random.exponential(1 / rate_total)
        t += dt
        # saut
        if np.random.rand() < lambda_ / rate_total:
            etat += 1
        elif etat > 0:
            etat -= 1
        times.append(t)
        states.append(etat)

    return times, states

def main():
    # — Entrée des paramètres
    lambda_ = 3
    mu       = 2
    m        = 2
    n_trans  = 10000
    burn_in  = 5000

    rho = lambda_ / (m * mu)
    if rho >= 1:
        raise ValueError(f"Système instable : ρ = {rho:.2f} ≥ 1")

    # — Simulation
    times, states = simuler_mm_m(lambda_, mu, m, n_trans)

    # — Trajectoire temporelle
    plt.figure(figsize=(10,4))
    plt.step(times, states, where='post')
    plt.xlabel("Temps")
    plt.ylabel("Nombre de clients")
    plt.title(f"Trajectoire M/M/{m} – λ={lambda_}, µ={mu}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # — Distribution empirique sur les derniers burn_in états
    sample = states[-burn_in:]
    counts = Counter(sample)
    max_k_emp = max(counts.keys())
    empirique = np.array([counts[k] / burn_in for k in range(max_k_emp+1)])

    # — Distribution théorique
    max_k = max(max_k_emp, m+10)
    theorique = calc_distribution_theorique(lambda_, mu, m, max_k)

    # — Comparaison des distributions
    ks_emp = np.arange(len(empirique))
    ks_th  = np.arange(len(theorique))

    plt.figure(figsize=(10,5))
    plt.bar(ks_emp, empirique, width=0.4, alpha=0.6,
            label="Empirique (simulation)")
    plt.plot(ks_th, theorique, 'ro-', lw=2,
             label="Théorique (formule)", markersize=4)
    plt.xlabel("Nombre de clients k")
    plt.ylabel("Probabilité πₖ")
    plt.title(f"Distribution asymptotique – M/M/{m}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # — Comparaison des moyennes
    E_emp = sum(k * p for k,p in enumerate(empirique))
    E_th  = sum(k * p for k,p in enumerate(theorique[:len(empirique)]))
    print("----- Moyennes en régime asymptotique -----")
    print(f"Moyenne empirique (sur {burn_in} états) : {E_emp:.4f}")
    print(f"Moyenne théorique (∑ k πₖ) : {E_th:.4f}")

if __name__ == "__main__":
    main()
