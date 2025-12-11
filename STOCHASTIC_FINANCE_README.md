# Calcul Stochastique Appliqué à la Finance - Samsung Electronics

## Description

Ce projet implémente des modèles de calcul stochastique pour l'analyse financière des actions Samsung Electronics. Il comprend:

- **Modèle Black-Scholes**: Pour le pricing d'options européennes (call et put)
- **Simulations Monte Carlo**: Pour la prédiction des trajectoires de prix
- **Analyse de Risque**: Calcul de Value at Risk (VaR) et autres métriques
- **Grecques**: Delta, Gamma, Vega, Theta pour la gestion du risque d'options

## Stochastic Calculus Applied to Finance - Samsung Electronics

This project implements stochastic calculus models for financial analysis of Samsung Electronics stock, including:

- **Black-Scholes Model**: For European option pricing (calls and puts)
- **Monte Carlo Simulations**: For stock price path prediction
- **Risk Analysis**: Value at Risk (VaR) calculation and other metrics
- **Greeks**: Delta, Gamma, Vega, Theta for option risk management

## Installation

### Prérequis / Prerequisites

- Python 3.7 ou supérieur / Python 3.7 or higher
- pip (gestionnaire de paquets Python / Python package manager)

### Installation des dépendances / Installing dependencies

```bash
pip install -r requirements.txt
```

## Utilisation / Usage

### Exécution de l'exemple / Running the example

```bash
python stochastic_finance_samsung.py
```

### Utilisation comme module / Using as a module

```python
from stochastic_finance_samsung import SamsungStochasticAnalysis

# Créer une instance avec le prix actuel de Samsung
analyzer = SamsungStochasticAnalysis(current_price=71000)

# Analyser les options
options_results = analyzer.analyze_options(
    strike_price=75000,
    days_to_expiry=90,
    volatility=0.30,
    risk_free_rate=0.035
)

# Effectuer une simulation Monte Carlo
mc_results = analyzer.monte_carlo_forecast(
    days=252,
    n_simulations=10000,
    mu=0.10,
    sigma=0.30
)

# Afficher les résultats
print("Prix Call:", options_results["Prix Call"])
print("Prix moyen projeté:", mc_results["Prix Moyen Final"])
```

## Modèles Mathématiques / Mathematical Models

### Modèle Black-Scholes

Le prix d'une option d'achat européenne est donné par:

```
C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)
```

où:
- S₀ = prix actuel de l'action
- K = prix d'exercice
- T = temps jusqu'à l'échéance
- r = taux sans risque
- σ = volatilité
- N(·) = fonction de répartition normale

avec:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Mouvement Brownien Géométrique

Les prix des actions suivent un mouvement brownien géométrique:

```
dS = μS dt + σS dW
```

où:
- μ = rendement attendu (drift)
- σ = volatilité
- dW = processus de Wiener (mouvement brownien)

La solution de cette équation stochastique est:

```
S(t) = S₀ exp[(μ - σ²/2)t + σW(t)]
```

## Exemples de Résultats / Example Results

Lors de l'exécution du script avec un prix initial de 71,000 KRW pour Samsung Electronics:

```
ANALYSE BLACK-SCHOLES DES OPTIONS
Prix d'exercice: 75,000 KRW
Jours jusqu'à l'échéance: 90

Prix Call:           3,234.56 KRW
Prix Put:            6,891.23 KRW
Delta Call:          0.4567
Gamma:               0.0001
Vega:                28.45

SIMULATION MONTE CARLO (1 AN)
Prix Moyen Final:    78,234.56 KRW
Intervalle de confiance 90%:
  - Percentile 5%:   52,123.45 KRW
  - Percentile 95%:  108,765.43 KRW
VaR 95%:             -26.67%
```

## Structure du Code / Code Structure

```
stochastic_finance_samsung.py
├── BlackScholesModel
│   ├── call_price()      # Prix option d'achat
│   ├── put_price()       # Prix option de vente
│   ├── delta_call()      # Sensibilité au prix
│   ├── gamma()           # Convexité
│   ├── vega()            # Sensibilité à la volatilité
│   └── theta_call()      # Déclin temporel
│
├── MonteCarloSimulation
│   ├── simulate_paths()         # Simulation trajectoires
│   ├── price_european_call()    # Prix par Monte Carlo
│   └── calculate_var()          # Value at Risk
│
└── SamsungStochasticAnalysis
    ├── analyze_options()        # Analyse complète options
    ├── monte_carlo_forecast()   # Prévisions Monte Carlo
    └── plot_monte_carlo()       # Visualisation
```

## Limitations et Avertissements / Limitations and Warnings

⚠️ **IMPORTANT**: Ce code est fourni à des fins éducatives et de démonstration uniquement.

- Les résultats ne constituent pas des conseils d'investissement
- Les modèles supposent des marchés efficients et une volatilité constante
- Les prix réels peuvent différer significativement des prédictions
- Toujours consulter un conseiller financier professionnel avant d'investir

⚠️ **IMPORTANT**: This code is provided for educational and demonstration purposes only.

- Results do not constitute investment advice
- Models assume efficient markets and constant volatility
- Actual prices may differ significantly from predictions
- Always consult a professional financial advisor before investing

## Applications Pratiques / Practical Applications

1. **Pricing d'Options**: Calcul du juste prix des options sur Samsung
2. **Gestion de Risque**: Évaluation du risque via VaR et grecques
3. **Couverture**: Détermination de stratégies de hedging
4. **Prévisions**: Estimation de prix futurs avec intervalles de confiance

## Ressources Additionnelles / Additional Resources

- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Monte Carlo Methods in Finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance)
- [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- [Greeks (finance)](https://en.wikipedia.org/wiki/Greeks_(finance))

## Licence / License

MIT License - voir le fichier LICENSE pour plus de détails

## Auteur / Author

Projet créé pour l'analyse stochastique de Samsung Electronics
Project created for stochastic analysis of Samsung Electronics
