# Asset Management Project

This repository contains an analysis and implementation of portfolio allocation strategies within the S&P 500 investment universe.

## 📂 Overview
- The project analyzes **sector** and **factor indices** of the S&P 500, focusing on portfolio construction and quantitative evaluation of strategies.
- **Historical data** includes prices and market capitalizations of 11 sector indices and 5 factor indices.
- Key metrics include **Sharpe Ratio**, **volatility**, **maximum drawdown**, and **diversification ratio**.

## ⚙️ Methods
1. **Markowitz Efficient Frontier**:
   - Computed portfolios with Minimum Variance and Maximum Sharpe Ratio.
2. **Enhanced Constraints**:
   - Custom constraints for sector weights to manage risk and diversification.
3. **Resampled Frontiers**:
   - Used Monte Carlo simulations to improve robustness under dynamic market conditions.
4. **Black-Litterman Model**:
   - Incorporated investor views into the Markowitz framework for stable allocations.
5. **Alternative Optimization**:
   - Portfolios optimized for diversification (entropy, diversification ratio) and VaR-modified Sharpe Ratio.

## 🏆 Results
- Portfolios maximizing the Sharpe Ratio showed high returns but increased concentration in sectors like **Information Technology** and **Momentum**.
- Minimum Variance portfolios offered more stability, ideal for risk-averse investors.
- Resampling and advanced models, such as **Black-Litterman**, improved adaptability to real-world data.

This project was completed as part of the **Computational Finance** course.
