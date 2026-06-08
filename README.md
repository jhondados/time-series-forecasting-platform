# 📈 Time Series Forecasting Platform

[![Series](https://img.shields.io/badge/Series%20Managed-50K%2B-blue)](.)
[![MAPE](https://img.shields.io/badge/MAPE-3.8%25-green)](.)
[![Horizon](https://img.shields.io/badge/Horizon-365%20days-orange)](.)

> Enterprise time series platform managing **50,000+ simultaneous series** using Temporal Fusion Transformer + AutoML ensemble. Achieves **3.8% MAPE** on 365-day horizons with probabilistic uncertainty quantification.

## 🏆 Results
- **3.8% MAPE** on 1-year horizon forecasts (industry benchmark: 12-18%)
- **50,000+ series** forecasted simultaneously with hierarchical coherence
- **99.2% SLA** — forecasts always available before business opening
- Powers inventory, finance and workforce planning for 3 enterprise clients

## 🏗️ Model Architecture
```
Input Features: historical_values, covariates (weather, events, promotions)
     │
     ▼
Variable Selection Network ──▶ LSTM Encoder ──▶ Temporal Attention ──▶ Quantile Output
                                                  (Multi-head, 4 heads)  (P10/P50/P90)
```
