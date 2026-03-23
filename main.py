import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


# =========================================================
# 1. DESCARGA DE DATOS
# =========================================================
def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker}.")

    # Si viene con multi-index, normalizamos
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_cols = ["Close"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"La columna '{col}' no está disponible en los datos.")

    data = data[["Close"]].copy()
    data.dropna(inplace=True)
    return data


# =========================================================
# 2. RENDIMIENTOS LOGARÍTMICOS
# =========================================================
def compute_log_returns(prices: pd.Series) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


# =========================================================
# 3. MÁXIMA VEROSIMILITUD PARA GBM
#    Bajo GBM:
#    log(S_t/S_{t-1}) ~ N((mu - 0.5*sigma^2)dt, sigma^2 dt)
# =========================================================
def estimate_gbm_mle(log_returns: pd.Series, dt: float = 1 / 252) -> tuple[float, float]:
    """
    Regresa:
    mu_hat     : estimación anual del drift
    sigma_hat  : estimación anual de la volatilidad
    """
    m = log_returns.mean()
    s2 = log_returns.var(ddof=0)

    sigma_hat = math.sqrt(s2 / dt)
    mu_hat = (m / dt) + 0.5 * sigma_hat**2

    return mu_hat, sigma_hat


# =========================================================
# 4. BLACK-SCHOLES
# =========================================================
def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S0, K, sigma y T deben ser positivos.")

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price


# =========================================================
# 5. MONTE CARLO ESTÁNDAR
#    S_T = S_0 exp((r - 0.5 sigma^2)T + sigma sqrt(T) Z)
# =========================================================
def monte_carlo_call_standard(
    S0: float, K: float, r: float, sigma: float, T: float, n_sim: int, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sim)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    price = discounted_payoffs.mean()
    variance = discounted_payoffs.var(ddof=1)

    return price, variance


# =========================================================
# 6. VARIABLES ANTITÉTICAS
# =========================================================
def monte_carlo_call_antithetic(
    S0: float, K: float, r: float, sigma: float, T: float, n_sim: int, seed: int = 42
) -> tuple[float, float]:
    if n_sim % 2 != 0:
        n_sim += 1  # forzamos par

    rng = np.random.default_rng(seed)
    half = n_sim // 2
    Z = rng.standard_normal(half)

    ST_plus = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ST_minus = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)

    payoff_plus = np.maximum(ST_plus - K, 0.0)
    payoff_minus = np.maximum(ST_minus - K, 0.0)

    paired_payoff = 0.5 * (payoff_plus + payoff_minus)
    discounted_payoff = np.exp(-r * T) * paired_payoff

    price = discounted_payoff.mean()
    variance = discounted_payoff.var(ddof=1)

    return price, variance


# =========================================================
# 7. VARIABLE DE CONTROL
#    Usamos S_T como variable de control.
#    E[e^{-rT} S_T] = S_0
# =========================================================
def monte_carlo_call_control_variate(
    S0: float, K: float, r: float, sigma: float, T: float, n_sim: int, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sim)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    X = np.exp(-r * T) * np.maximum(ST - K, 0.0)   # variable objetivo
    Y = np.exp(-r * T) * ST                        # variable de control
    EY = S0                                        # valor esperado exacto

    cov_xy = np.cov(X, Y, ddof=1)[0, 1]
    var_y = np.var(Y, ddof=1)

    b_opt = cov_xy / var_y if var_y > 0 else 0.0
    X_cv = X - b_opt * (Y - EY)

    price = X_cv.mean()
    variance = X_cv.var(ddof=1)

    return price, variance


# =========================================================
# 8. FUNCIÓN AUXILIAR PARA TIEMPO
# =========================================================
def timed_method(method, *args, **kwargs):
    start = time.perf_counter()
    price, variance = method(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return price, variance, elapsed


# =========================================================
# 9. GRÁFICAS
# =========================================================
def plot_prices(data: pd.DataFrame, ticker: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"])
    plt.title(f"Precio histórico ajustado de {ticker}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_log_returns_hist(log_returns: pd.Series, ticker: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(log_returns, bins=30, edgecolor="black")
    plt.title(f"Histograma de rendimientos logarítmicos de {ticker}")
    plt.xlabel("Rendimiento logarítmico")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# 10. COMPARACIÓN DE MÉTODOS
# =========================================================
def compare_methods(
    S0: float, K: float, r: float, sigma: float, T: float, n_sim: int, bs_price: float
) -> pd.DataFrame:
    results = []

    methods = {
        "Monte Carlo estándar": monte_carlo_call_standard,
        "Antitéticas": monte_carlo_call_antithetic,
        "Variable de control": monte_carlo_call_control_variate,
    }

    for name, method in methods.items():
        price, variance, elapsed = timed_method(method, S0, K, r, sigma, T, n_sim)
        abs_error = abs(price - bs_price)

        results.append({
            "Método": name,
            "Precio estimado": price,
            "Error absoluto vs BS": abs_error,
            "Varianza del estimador": variance,
            "Tiempo (s)": elapsed
        })

    df_results = pd.DataFrame(results)
    return df_results


# =========================================================
# 11. PROGRAMA PRINCIPAL
# =========================================================
def main():
    # -----------------------------------------------------
    # Parámetros del problema
    # -----------------------------------------------------
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2025-01-01"

    K = 200.0       # strike
    T = 1.0         # 1 año
    r = 0.10        # tasa libre de riesgo anual (ejemplo)
    n_sim = 100000  # número de simulaciones

    # -----------------------------------------------------
    # Descarga de datos
    # -----------------------------------------------------
    data = download_stock_data(ticker, start_date, end_date)
    prices = data["Close"]
    S0 = float(prices.iloc[-1])

    # -----------------------------------------------------
    # Rendimientos y estimación MLE
    # -----------------------------------------------------
    log_returns = compute_log_returns(prices)
    mu_hat, sigma_hat = estimate_gbm_mle(log_returns)

    # -----------------------------------------------------
    # Precio Black-Scholes
    # -----------------------------------------------------
    bs_price = black_scholes_call(S0, K, r, sigma_hat, T)

    # -----------------------------------------------------
    # Comparación Monte Carlo
    # -----------------------------------------------------
    results_df = compare_methods(S0, K, r, sigma_hat, T, n_sim, bs_price)

    # -----------------------------------------------------
    # Mostrar resultados
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("PARÁMETROS ESTIMADOS POR MÁXIMA VEROSIMILITUD")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Precio actual S0: {S0:.4f}")
    print(f"Media anual estimada mu_hat: {mu_hat:.6f}")
    print(f"Volatilidad anual estimada sigma_hat: {sigma_hat:.6f}")

    print("\n" + "=" * 60)
    print("PRECIO BLACK-SCHOLES")
    print("=" * 60)
    print(f"Call europea: {bs_price:.6f}")

    print("\n" + "=" * 60)
    print("COMPARACIÓN DE MÉTODOS")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # -----------------------------------------------------
    # Gráficas
    # -----------------------------------------------------
    plot_prices(data, ticker)
    plot_log_returns_hist(log_returns, ticker)


if __name__ == "__main__":
    main()

# Guardar resultados
results_df.to_csv("results.csv", index=False)

print("\nResultados guardados en results.csv")
plt.savefig("prices.png")
plt.savefig("returns.png")
