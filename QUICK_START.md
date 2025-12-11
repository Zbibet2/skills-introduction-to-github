# Quick Start - Projet Calcul Stochastique Samsung

## Démarrage Rapide / Quick Start

Ce projet contient un système complet d'analyse stochastique pour Samsung Electronics.

This project contains a complete stochastic analysis system for Samsung Electronics.

### Installation Rapide / Quick Installation

```bash
# 1. Installer les dépendances / Install dependencies
pip install -r requirements.txt

# 2. Exécuter le programme / Run the program
python stochastic_finance_samsung.py
```

### Ce que fait le programme / What the program does

Le programme effectue automatiquement:

1. **Analyse Black-Scholes** - Calcul du prix des options pour Samsung
2. **Simulation Monte Carlo** - Prévision des prix futurs sur 1 an
3. **Analyse de Risque** - Calcul de la VaR (Value at Risk)
4. **Visualisation** - Génération d'un graphique `samsung_monte_carlo_simulation.png`

The program automatically performs:

1. **Black-Scholes Analysis** - Calculate option prices for Samsung
2. **Monte Carlo Simulation** - Forecast future prices over 1 year
3. **Risk Analysis** - Calculate VaR (Value at Risk)
4. **Visualization** - Generate a chart `samsung_monte_carlo_simulation.png`

### Résultat Attendu / Expected Output

```
======================================================================
ANALYSE STOCHASTIQUE - SAMSUNG ELECTRONICS
======================================================================

Prix actuel de l'action: 71,000 KRW

----------------------------------------------------------------------
1. ANALYSE BLACK-SCHOLES DES OPTIONS
----------------------------------------------------------------------
Prix d'exercice: 75,000 KRW
Jours jusqu'à l'échéance: 90

Prix Call           :    2842.75 KRW
Prix Put            :    6198.27 KRW
Delta Call          :       0.41
...

----------------------------------------------------------------------
2. SIMULATION MONTE CARLO (1 AN)
----------------------------------------------------------------------
Prix Moyen Final:         76,055.37 KRW
Intervalle de confiance 90%:
  - Percentile 5%:        49,220.08 KRW
  - Percentile 95%:      111,279.95 KRW
VaR 95%:                    -31.00%
...
```

### Utilisation Personnalisée / Custom Usage

Pour analyser avec vos propres paramètres / To analyze with your own parameters:

```python
from stochastic_finance_samsung import SamsungStochasticAnalysis

# Créer l'analyseur avec votre prix / Create analyzer with your price
analyzer = SamsungStochasticAnalysis(current_price=80000)

# Analyser des options / Analyze options
results = analyzer.analyze_options(
    strike_price=85000,
    days_to_expiry=60,
    volatility=0.35,
    risk_free_rate=0.04
)

# Simulation sur 6 mois / 6-month simulation
forecast = analyzer.monte_carlo_forecast(
    days=126,  # ~6 mois / ~6 months
    n_simulations=5000,
    mu=0.12,
    sigma=0.28
)

print("Prix Call:", results["Prix Call"])
print("Prix projeté:", forecast["Prix Moyen Final"])
```

### Fichiers du Projet / Project Files

- `stochastic_finance_samsung.py` - Code principal / Main code
- `requirements.txt` - Dépendances Python / Python dependencies
- `STOCHASTIC_FINANCE_README.md` - Documentation complète / Full documentation
- `QUICK_START.md` - Ce fichier / This file

### Documentation Complète / Full Documentation

Pour plus de détails sur les modèles mathématiques et les options avancées, consultez:

For more details on mathematical models and advanced options, see:

📖 [STOCHASTIC_FINANCE_README.md](STOCHASTIC_FINANCE_README.md)

### Support

Pour toute question sur les modèles financiers ou l'implémentation, veuillez consulter la documentation complète.

For any questions about the financial models or implementation, please refer to the full documentation.
