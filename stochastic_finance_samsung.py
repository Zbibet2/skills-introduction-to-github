"""
Calcul Stochastique Appliqué à la Finance - Samsung Electronics
Stochastic Calculus Applied to Finance for Samsung Electronics

Ce module implémente des modèles de calcul stochastique pour l'analyse financière
des actions Samsung Electronics, incluant:
- Modèle Black-Scholes pour le pricing d'options
- Simulations Monte Carlo pour la prédiction des prix
- Analyse de volatilité stochastique
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class BlackScholesModel:
    """
    Modèle Black-Scholes pour le pricing d'options européennes
    Black-Scholes model for European option pricing
    """
    
    def __init__(self, S0, K, T, r, sigma):
        """
        Parameters:
        -----------
        S0 : float
            Prix actuel de l'action (Current stock price)
        K : float
            Prix d'exercice (Strike price)
        T : float
            Temps jusqu'à l'échéance en années (Time to maturity in years)
        r : float
            Taux sans risque (Risk-free rate)
        sigma : float
            Volatilité (Volatility)
        """
        if S0 <= 0:
            raise ValueError("Stock price S0 must be positive")
        if K <= 0:
            raise ValueError("Strike price K must be positive")
        if T <= 0:
            raise ValueError("Time to maturity T must be positive")
        if sigma <= 0:
            raise ValueError("Volatility sigma must be positive")
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def d1(self):
        """Calcul de d1 dans la formule Black-Scholes"""
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        """Calcul de d2 dans la formule Black-Scholes"""
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        """Prix d'une option d'achat européenne (European call option price)"""
        d1 = self.d1()
        d2 = self.d2()
        return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def put_price(self):
        """Prix d'une option de vente européenne (European put option price)"""
        d1 = self.d1()
        d2 = self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
    
    def delta_call(self):
        """Delta d'une option d'achat (Call option delta)"""
        return norm.cdf(self.d1())
    
    def delta_put(self):
        """Delta d'une option de vente (Put option delta)"""
        return -norm.cdf(-self.d1())
    
    def gamma(self):
        """Gamma de l'option (Option gamma)"""
        return norm.pdf(self.d1()) / (self.S0 * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """Vega de l'option (Option vega)"""
        return self.S0 * norm.pdf(self.d1()) * np.sqrt(self.T)
    
    def theta_call(self):
        """Theta d'une option d'achat (Call option theta)"""
        d1 = self.d1()
        d2 = self.d2()
        term1 = -(self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return term1 + term2


class MonteCarloSimulation:
    """
    Simulation Monte Carlo pour la prédiction des prix d'actions
    Monte Carlo simulation for stock price prediction
    """
    
    def __init__(self, S0, mu, sigma, T, dt, n_simulations):
        """
        Parameters:
        -----------
        S0 : float
            Prix initial de l'action (Initial stock price)
        mu : float
            Rendement attendu (Expected return)
        sigma : float
            Volatilité (Volatility)
        T : float
            Horizon temporel en années (Time horizon in years)
        dt : float
            Pas de temps (Time step)
        n_simulations : int
            Nombre de simulations (Number of simulations)
        """
        if S0 <= 0:
            raise ValueError("Initial stock price S0 must be positive")
        if sigma < 0:
            raise ValueError("Volatility sigma must be non-negative")
        if T <= 0:
            raise ValueError("Time horizon T must be positive")
        if dt <= 0:
            raise ValueError("Time step dt must be positive")
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.n_simulations = n_simulations
        self.n_steps = int(T / dt)
    
    def simulate_paths(self):
        """
        Simule les trajectoires de prix selon un mouvement brownien géométrique
        Simulate price paths using geometric Brownian motion
        
        dS = μS dt + σS dW
        """
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(1, self.n_steps + 1):
            Z = np.random.standard_normal(self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * np.sqrt(self.dt) * Z
            )
        
        return paths
    
    def price_european_call(self, K, r):
        """
        Prix d'une option d'achat européenne par Monte Carlo
        European call option price using Monte Carlo
        """
        paths = self.simulate_paths()
        final_prices = paths[:, -1]
        payoffs = np.maximum(final_prices - K, 0)
        discounted_payoff = np.exp(-r * self.T) * payoffs
        return np.mean(discounted_payoff), np.std(discounted_payoff)
    
    def calculate_var(self, confidence_level=0.95):
        """
        Calcule la Value at Risk (VaR)
        Calculate Value at Risk
        """
        paths = self.simulate_paths()
        returns = (paths[:, -1] - self.S0) / self.S0
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var


class SamsungStochasticAnalysis:
    """
    Analyse stochastique spécifique pour Samsung Electronics
    Stochastic analysis specific to Samsung Electronics
    """
    
    def __init__(self, current_price):
        """
        Parameters:
        -----------
        current_price : float
            Prix actuel de l'action Samsung (Current Samsung stock price)
        """
        self.current_price = current_price
        self.company_name = "Samsung Electronics"
    
    def analyze_options(self, strike_price, days_to_expiry, volatility=0.25, risk_free_rate=0.03):
        """
        Analyse complète des options pour Samsung
        Complete options analysis for Samsung
        """
        T = days_to_expiry / 365.0
        bs = BlackScholesModel(
            S0=self.current_price,
            K=strike_price,
            T=T,
            r=risk_free_rate,
            sigma=volatility
        )
        
        results = {
            "Prix Call": bs.call_price(),
            "Prix Put": bs.put_price(),
            "Delta Call": bs.delta_call(),
            "Delta Put": bs.delta_put(),
            "Gamma": bs.gamma(),
            "Vega": bs.vega(),
            "Theta Call": bs.theta_call()
        }
        
        return results
    
    def monte_carlo_forecast(self, days=252, n_simulations=10000, mu=0.08, sigma=0.25):
        """
        Prévision par Monte Carlo pour Samsung
        Monte Carlo forecast for Samsung
        """
        T = days / 365.0
        dt = 1 / 252  # Daily time step
        
        mc = MonteCarloSimulation(
            S0=self.current_price,
            mu=mu,
            sigma=sigma,
            T=T,
            dt=dt,
            n_simulations=n_simulations
        )
        
        paths = mc.simulate_paths()
        
        results = {
            "Prix Moyen Final": np.mean(paths[:, -1]),
            "Prix Médian Final": np.median(paths[:, -1]),
            "Écart-type": np.std(paths[:, -1]),
            "Percentile 5%": np.percentile(paths[:, -1], 5),
            "Percentile 95%": np.percentile(paths[:, -1], 95),
            "VaR 95%": mc.calculate_var(0.95),
            "Paths": paths
        }
        
        return results
    
    def plot_monte_carlo(self, results, n_paths_to_plot=100):
        """
        Visualise les simulations Monte Carlo
        Visualize Monte Carlo simulations
        """
        paths = results["Paths"]
        n_paths, n_steps = paths.shape
        
        plt.figure(figsize=(12, 6))
        
        # Plot subset of paths
        for i in range(min(n_paths_to_plot, n_paths)):
            plt.plot(paths[i], alpha=0.3, linewidth=0.5)
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        plt.plot(mean_path, color='red', linewidth=2, label='Trajectoire Moyenne')
        
        plt.axhline(y=self.current_price, color='black', linestyle='--', 
                   label=f'Prix Initial: {self.current_price:.2f}')
        plt.xlabel('Jours de Trading')
        plt.ylabel('Prix de l\'Action (KRW)')
        plt.title(f'Simulation Monte Carlo - {self.company_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt


def exemple_samsung():
    """
    Exemple d'utilisation pour l'analyse de Samsung Electronics
    Example usage for Samsung Electronics analysis
    """
    print("=" * 70)
    print("ANALYSE STOCHASTIQUE - SAMSUNG ELECTRONICS")
    print("STOCHASTIC ANALYSIS - SAMSUNG ELECTRONICS")
    print("=" * 70)
    print()
    
    # Prix actuel approximatif de Samsung Electronics (en KRW)
    # Approximate current price of Samsung Electronics (in KRW)
    prix_samsung = 71000  # Prix d'exemple / Example price
    
    # Créer l'analyseur
    analyzer = SamsungStochasticAnalysis(prix_samsung)
    
    print(f"Prix actuel de l'action: {prix_samsung:,.0f} KRW")
    print()
    
    # 1. Analyse des options
    print("-" * 70)
    print("1. ANALYSE BLACK-SCHOLES DES OPTIONS")
    print("-" * 70)
    
    strike_price = 75000  # Prix d'exercice / Strike price
    days_to_expiry = 90  # 3 mois / 3 months
    
    options_results = analyzer.analyze_options(
        strike_price=strike_price,
        days_to_expiry=days_to_expiry,
        volatility=0.30,  # 30% volatilité annuelle
        risk_free_rate=0.035  # 3.5% taux sans risque
    )
    
    print(f"Prix d'exercice: {strike_price:,.0f} KRW")
    print(f"Jours jusqu'à l'échéance: {days_to_expiry}")
    print()
    
    for key, value in options_results.items():
        print(f"{key:20s}: {value:12.4f}")
    
    print()
    
    # 2. Simulation Monte Carlo
    print("-" * 70)
    print("2. SIMULATION MONTE CARLO (1 AN)")
    print("-" * 70)
    
    mc_results = analyzer.monte_carlo_forecast(
        days=252,  # 1 année de trading
        n_simulations=10000,
        mu=0.10,  # 10% rendement attendu annuel
        sigma=0.30  # 30% volatilité annuelle
    )
    
    print(f"Nombre de simulations: 10,000")
    print(f"Horizon: 1 an (252 jours de trading)")
    print()
    print(f"Prix Moyen Final:      {mc_results['Prix Moyen Final']:12,.2f} KRW")
    print(f"Prix Médian Final:     {mc_results['Prix Médian Final']:12,.2f} KRW")
    print(f"Écart-type:            {mc_results['Écart-type']:12,.2f} KRW")
    print(f"Intervalle de confiance 90%:")
    print(f"  - Percentile 5%:     {mc_results['Percentile 5%']:12,.2f} KRW")
    print(f"  - Percentile 95%:    {mc_results['Percentile 95%']:12,.2f} KRW")
    print(f"VaR 95% (perte max):   {mc_results['VaR 95%']:12.2%}")
    print()
    
    # 3. Rendement attendu
    rendement_attendu = (mc_results['Prix Moyen Final'] - prix_samsung) / prix_samsung
    print(f"Rendement attendu sur 1 an: {rendement_attendu:+.2%}")
    print()
    
    print("=" * 70)
    print("Note: Ces valeurs sont calculées à des fins de démonstration")
    print("Note: These values are calculated for demonstration purposes")
    print("=" * 70)
    
    return analyzer, mc_results


if __name__ == "__main__":
    # Exécuter l'exemple
    analyzer, mc_results = exemple_samsung()
    
    # Optionnel: créer un graphique si matplotlib est disponible
    try:
        print("\nGénération du graphique des simulations Monte Carlo...")
        plt_obj = analyzer.plot_monte_carlo(mc_results, n_paths_to_plot=100)
        plt_obj.savefig('samsung_monte_carlo_simulation.png', dpi=150, bbox_inches='tight')
        print("Graphique sauvegardé: samsung_monte_carlo_simulation.png")
    except Exception as e:
        print(f"Impossible de créer le graphique: {e}")
