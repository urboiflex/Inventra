"""
Inventra analytics engine.

Pure calculation layer — operates on pandas DataFrames and plain numbers, with no
Django imports — so it can be unit-tested in isolation. The formulas implement the
DSS spec exactly (see CLAUDE.md): CSV cleaning, demand statistics, SES / moving-average
forecasting, safety stock, reorder point, replenishment, simulation and financials.
"""
import math

import numpy as np
import pandas as pd

# Required columns a user's CSV must contain.
REQUIRED_COLUMNS = ['transaction_date', 'product_name', 'quantity', 'final_amount', 'unit_price']

# Service-level (%) -> Z score mapping.
Z_MAP = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}
DEFAULT_Z = 1.65

SES_ALPHAS = [0.2, 0.5, 0.8]
MA_WINDOW = 4
STATIC_THRESHOLD = 10  # static reorder policy threshold (units)

HOLDING_RATE = 0.25  # annual holding cost as a fraction of unit price


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe(value, default=0.0):
    """Coerce a number to a finite float, falling back to `default` for NaN/inf."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def z_from_service_level(service_level):
    """Map a service-level percentage to its Z score (nearest known key)."""
    if service_level in Z_MAP:
        return Z_MAP[service_level]
    # tolerate floats like 97.5 vs 98; pick the closest configured level
    closest = min(Z_MAP, key=lambda k: abs(k - service_level))
    return Z_MAP[closest]


# --------------------------------------------------------------------------- #
# CSV cleaning & validation
# --------------------------------------------------------------------------- #
def validate_csv(file_path, min_rows=10):
    """
    Validate an uploaded CSV. Returns (ok: bool, message: str, dataframe_or_None).
    Checks required columns, parseable dates and a minimum row count.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        return False, f"Could not read the CSV file: {exc}", None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}.", None

    if len(df) < min_rows:
        return False, f"CSV must contain at least {min_rows} rows of data (found {len(df)}).", None

    try:
        pd.to_datetime(df['transaction_date'], format='mixed', errors='raise')
    except Exception:  # noqa: BLE001
        return False, "Column 'transaction_date' contains values that could not be parsed as dates.", None

    return True, "CSV is valid.", df


def clean_dataframe(df):
    """
    Apply the spec's cleaning steps:
      - drop rows missing store_name (when the column exists)
      - remove non-positive final_amount
      - parse and sort by transaction_date
    """
    df = df.copy()
    if 'store_name' in df.columns:
        df = df.dropna(subset=['store_name'])
    df = df[df['final_amount'] > 0]
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='mixed')
    df = df.sort_values('transaction_date')
    return df


def weekly_demand_series(df_product):
    """
    Weekly demand for one product: resample quantity to weekly sums, then treat zero
    weeks as gaps and forward-fill them. Leading gaps that cannot be filled are dropped.
    """
    if df_product.empty:
        return pd.Series(dtype='float64')
    weekly = df_product.set_index('transaction_date')['quantity'].resample('W').sum()
    weekly = weekly.replace(0, np.nan).ffill().dropna()
    return weekly


def weekly_revenue_series(df_product):
    """Weekly revenue for one product (sum of final_amount per calendar week)."""
    if df_product.empty:
        return pd.Series(dtype='float64')
    return df_product.groupby(pd.Grouper(key='transaction_date', freq='W'))['final_amount'].sum()


# --------------------------------------------------------------------------- #
# Demand statistics
# --------------------------------------------------------------------------- #
def demand_stats(weekly):
    """Return (avg_weekly_demand, std_weekly_demand, daily_velocity)."""
    avg = _safe(weekly.mean()) if len(weekly) else 0.0
    std = _safe(weekly.std()) if len(weekly) > 1 else 0.0
    velocity = avg / 7 if avg else 0.0
    return avg, std, velocity


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def ses_forecast(demand, alpha):
    """Single exponential smoothing forecast series (same length as demand)."""
    demand = list(demand)
    if not demand:
        return []
    forecasts = [demand[0]]
    for t in range(1, len(demand)):
        forecasts.append(alpha * demand[t - 1] + (1 - alpha) * forecasts[t - 1])
    return forecasts


def ma_forecast(demand, window=MA_WINDOW):
    """Moving-average forecast series (uses an expanding mean until `window` is reached)."""
    demand = list(demand)
    forecasts = []
    for t in range(len(demand)):
        if t < window:
            forecasts.append(float(np.mean(demand[:t + 1])))
        else:
            forecasts.append(float(np.mean(demand[t - window:t])))
    return forecasts


def mae(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if actual.size == 0:
        return 0.0
    return _safe(np.mean(np.abs(actual - forecast)))


def rmse(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if actual.size == 0:
        return 0.0
    return _safe(math.sqrt(np.mean((actual - forecast) ** 2)))


def best_ses(weekly):
    """
    Try each candidate alpha and return the one with the lowest MAE.
    Returns dict: {alpha, forecast, mae, rmse}.
    """
    demand = list(weekly)
    best = None
    for alpha in SES_ALPHAS:
        fc = ses_forecast(demand, alpha)
        m = mae(demand, fc)
        if best is None or m < best['mae']:
            best = {'alpha': alpha, 'forecast': fc, 'mae': m, 'rmse': rmse(demand, fc)}
    if best is None:
        best = {'alpha': SES_ALPHAS[0], 'forecast': [], 'mae': 0.0, 'rmse': 0.0}
    return best


def compare_algorithms(weekly):
    """
    Compare best-SES against the 4-week moving average.
    Returns a dict with per-method MAE/RMSE, forecasts, the chosen alpha and best method.
    """
    demand = list(weekly)
    ses = best_ses(weekly)
    ma_fc = ma_forecast(demand)
    mae_ma, rmse_ma = mae(demand, ma_fc), rmse(demand, ma_fc)
    best_method = 'SES' if ses['mae'] <= mae_ma else 'MA-4'
    return {
        'weeks': demand,
        'best_alpha': ses['alpha'],
        'ses_forecast': ses['forecast'],
        'ma_forecast': ma_fc,
        'mae_ses': ses['mae'],
        'rmse_ses': ses['rmse'],
        'mae_ma': mae_ma,
        'rmse_ma': rmse_ma,
        'best_method': best_method,
    }


# --------------------------------------------------------------------------- #
# Inventory metrics
# --------------------------------------------------------------------------- #
def safety_stock(z, std_weekly_demand, lead_time):
    return _safe(z * std_weekly_demand * math.sqrt(max(lead_time, 0)))


def reorder_point(avg_weekly_demand, lead_time, ss):
    return _safe(avg_weekly_demand * lead_time + ss)


def sells_out_in(current_stock, daily_velocity):
    """Days until stock is depleted at current velocity. Infinite (None) when no velocity."""
    if daily_velocity <= 0:
        return None
    return round(current_stock / daily_velocity)


def stock_status(current_stock, rop, ss):
    """Return one of 'Safe', 'Order Soon', 'Critical'."""
    if current_stock > rop:
        return 'Safe'
    if current_stock > ss:
        return 'Order Soon'
    return 'Critical'


def replenishment_qty(rop, current_stock, ss):
    """Suggested order quantity to bring stock back above ROP plus a safety buffer."""
    return max(0.0, _safe(rop - current_stock + ss))


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate(weekly_demand, threshold, order_quantity, initial_stock):
    """
    Run a single reordering policy over the historical weekly demand.
    Returns dict: {stockouts, service_level, inventory_levels}.
    """
    demand = list(weekly_demand)
    total_weeks = len(demand)
    stock = float(initial_stock)
    stockouts = 0
    inventory_levels = []
    for d in demand:
        stock -= d
        if stock < 0:
            stockouts += 1
            stock = 0
        inventory_levels.append(stock)
        if stock <= threshold:
            stock += order_quantity
    service_level = ((total_weeks - stockouts) / total_weeks * 100) if total_weeks else 100.0
    return {
        'stockouts': stockouts,
        'service_level': _safe(service_level),
        'inventory_levels': inventory_levels,
    }


def run_simulation(weekly_demand, rop, order_quantity, initial_stock):
    """
    Compare the static threshold policy (reorder at <= 10) against the forecast-based
    ROP policy. Returns both result sets plus stockouts eliminated.
    """
    static = simulate(weekly_demand, STATIC_THRESHOLD, order_quantity, initial_stock)
    dss = simulate(weekly_demand, rop, order_quantity, initial_stock)
    return {
        'static': static,
        'dss': dss,
        'stockouts_eliminated': max(0, static['stockouts'] - dss['stockouts']),
    }


# --------------------------------------------------------------------------- #
# Financials
# --------------------------------------------------------------------------- #
def financials(current_stock, unit_price, ss, weekly_revenue, stockouts_static, stockouts_dss):
    """Compute the financial summary block (inventory value, holding/stockout costs, savings)."""
    avg_weekly_revenue = _safe(weekly_revenue.mean()) if len(weekly_revenue) else 0.0
    max_revenue = _safe(weekly_revenue.max()) if len(weekly_revenue) else 0.0
    min_revenue = _safe(weekly_revenue.min()) if len(weekly_revenue) else 0.0

    holding_cost_annual = _safe(ss * unit_price * HOLDING_RATE)
    holding_cost_weekly = holding_cost_annual / 52

    stockout_cost_static = stockouts_static * avg_weekly_revenue
    stockout_cost_dss = stockouts_dss * avg_weekly_revenue

    return {
        'inventory_value': _safe(current_stock * unit_price),
        'avg_weekly_revenue': avg_weekly_revenue,
        'max_revenue': max_revenue,
        'min_revenue': min_revenue,
        'holding_cost_weekly': holding_cost_weekly,
        'holding_cost_annual': holding_cost_annual,
        'stockout_cost_per_occurrence': avg_weekly_revenue,
        'stockout_cost_static': _safe(stockout_cost_static),
        'stockout_cost_dss': _safe(stockout_cost_dss),
        'dss_savings': _safe(stockout_cost_static - stockout_cost_dss),
    }


# --------------------------------------------------------------------------- #
# High-level product analysis
# --------------------------------------------------------------------------- #
def per_product_stats(df, product_name):
    """
    Compute the persisted demand statistics for one product from a cleaned dataframe.
    Returns dict: {avg_weekly_demand, std_weekly_demand, daily_velocity, weeks}.
    """
    df_product = df[df['product_name'] == product_name]
    weekly = weekly_demand_series(df_product)
    avg, std, velocity = demand_stats(weekly)
    return {
        'avg_weekly_demand': avg,
        'std_weekly_demand': std,
        'daily_velocity': velocity,
        'weeks': len(weekly),
    }


def analyze(weekly, weekly_revenue, current_stock, unit_price, lead_time, z, order_quantity):
    """
    Full analysis for a product given its weekly demand/revenue history and settings.
    Bundles forecasting, inventory metrics, simulation and financials for the dashboard.
    """
    avg, std, velocity = demand_stats(weekly)
    ss = safety_stock(z, std, lead_time)
    rop = reorder_point(avg, lead_time, ss)
    status = stock_status(current_stock, rop, ss)

    comparison = compare_algorithms(weekly)
    sim = run_simulation(weekly, rop, order_quantity, current_stock)
    fin = financials(
        current_stock, unit_price, ss, weekly_revenue,
        sim['static']['stockouts'], sim['dss']['stockouts'],
    )

    return {
        'avg_weekly_demand': avg,
        'std_weekly_demand': std,
        'daily_velocity': velocity,
        'safety_stock': ss,
        'reorder_point': rop,
        'sells_out_in': sells_out_in(current_stock, velocity),
        'status': status,
        'replenishment_qty': replenishment_qty(rop, current_stock, ss),
        'forecast': comparison,
        'simulation': sim,
        'financials': fin,
    }
