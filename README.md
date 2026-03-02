# 📊 Portfolio Optimization & Efficient Frontier (Python)

## 📌 Overview

This project implements a **Modern Portfolio Theory (MPT)** framework in Python to construct and evaluate an optimized investment portfolio.

The objective is to **maximize the Sharpe Ratio** under long-only constraints and compare performance against a benchmark (**SPY**).

The project includes:

- Portfolio return and volatility modeling
- Sharpe Ratio maximization
- Efficient Frontier construction (Monte Carlo simulation)
- Minimum Volatility portfolio
- Benchmark comparison (SPY)
- Risk metrics including Maximum Drawdown

---

## 🧠 Methodology

### 1️⃣ Data Collection

Historical daily adjusted closing prices were downloaded using `yfinance` for the following assets:

| Ticker | Description             |
|--------|-------------------------|
| AAPL   | Apple Inc.              |
| MSFT   | Microsoft Corp.         |
| JNJ    | Johnson & Johnson       |
| XOM    | Exxon Mobil Corp.       |
| SPY    | S&P 500 ETF (benchmark) |

**Time period:** 2018 – Present

---

### 2️⃣ Return & Risk Estimation

- Logarithmic daily returns
- Annualized mean returns
- Annualized covariance matrix (252 trading days)

$$R_p = w^T \mu$$

$$\sigma_p = \sqrt{w^T \Sigma w}$$

---

### 3️⃣ Portfolio Optimization

**Objective:**

$$\max \frac{E[R_p] - R_f}{\sigma_p}$$

**Constraints:**

- Fully invested: $\sum w = 1$
- Long-only: $0 \leq w \leq 1$

**Optimization method:** Sequential Least Squares Programming (SLSQP)

---

### 4️⃣ Efficient Frontier

- 5,000 random portfolios generated
- Risk-return space visualized
- Maximum Sharpe portfolio identified
- Minimum Volatility portfolio identified

---

### 5️⃣ Benchmark Comparison

The optimized portfolio was compared against **SPY** using:

- Annual Return
- Annual Volatility
- Sharpe Ratio
- Maximum Drawdown
- Cumulative Return growth

---

## 📈 Results Summary

| Metric            | Optimized Portfolio | SPY    |
|-------------------|---------------------|--------|
| Annual Return     | 18.1%               | 13.0%  |
| Annual Volatility | 21.7%               | 19.4%  |
| Sharpe Ratio      | 0.74                | 0.57   |
| Max Drawdown      | -32.6%              | -35.7% |

### Key Insights

- ✅ Higher risk-adjusted performance than SPY
- ✅ Improved Sharpe Ratio
- ✅ Lower maximum drawdown
- ⚠️ Slightly higher volatility

---

## ⚠️ Important Considerations

- Results are **in-sample**
- No transaction costs included
- No taxes considered
- Static allocation (no dynamic rebalancing)

**Future improvements:**

- Out-of-sample validation
- Sortino Ratio implementation
- Beta and CAPM analysis
- Rolling optimization
- Transaction cost modeling

---

## 🛠 Tech Stack

| Library    | Purpose                         |
|------------|---------------------------------|
| Python     | Core language                   |
| NumPy      | Numerical computing             |
| Pandas     | Data manipulation               |
| SciPy      | Optimization (SLSQP)            |
| Matplotlib | Data visualization              |
| yfinance   | Historical price data download  |

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the notebook:

```bash
jupyter notebook notebook/portfolio_analysis.ipynb
```

3. Run all cells.

---

## 📌 Author

**Paolo César Salazar**  
Engineering Student – Data & Quantitative Analysis

GitHub: [https://github.com/paolosalazarp](https://github.com/paolosalazarp)
