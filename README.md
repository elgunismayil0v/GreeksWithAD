# GreeksWithAD 📈🧮

A PyTorch-based Monte Carlo simulation engine for pricing European options and computing Greeks (Delta, Gamma, Vega) using automatic differentiation.

## ✨ Features

- ✅ Monte Carlo simulation of GBM (Geometric Brownian Motion)
- ✅ Pricing of European call/put options
- ✅ Automatic computation of Greeks using PyTorch autograd:
  - Delta (∂Price/∂S₀)
  - Gamma (∂²Price/∂S₀²)
  - Vega  (∂Price/∂σ)
- ✅ Modular and testable design (Open/Closed Principle)
- ✅ Easy GPU acceleration (via PyTorch)
- ✅ CI-ready structure (e.g., GitHub Actions, pytest)

---

## 🏗️ Project Structure

GreeksWithAD/
├── MonteCarlo/
│ ├── init.py
│ ├── GBM.py # Geometric Brownian Motion model
│ ├── MonteCarloSimulator.py # Path simulation engine
│ ├── Option.py # Option payoff definitions (e.g., EuropeanCall)
│ └── Pricer.py # Monte Carlo pricer
├── greeks.py # compute_all_greeks() function
├── main.py # Entry point for pricing & greeks
├── tests/
│ └── test_pricer.py # Unit tests
├── requirements.txt
└── README.md