"""
Inventra analytics engine.

Pure calculation layer — operates on pandas DataFrames and plain numbers, with no
Django imports — so it can be unit-tested in isolation. The formulas implement the
DSS spec exactly (see CLAUDE.md): CSV cleaning, demand statistics, SES / moving-average
forecasting, safety stock, reorder point, replenishment, simulation and financials.
"""
import math
from datetime import date

import numpy as np
import pandas as pd

# Required columns a user's CSV must contain (internal / native format).
REQUIRED_COLUMNS = ['transaction_date', 'product_name', 'quantity', 'final_amount', 'unit_price']

# UCI Online Retail dataset column signature (subset sufficient for detection).
_UCI_SIGNATURE = {'InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice'}

# UCI rows whose Description starts with these strings are admin/postage, not products.
_UCI_NOISE_PREFIXES = ('POST', 'DOT', 'CRUK', 'BANK', 'MANUAL', 'AMAZONFEE', 'PADS', 'ADJUST')

# Service-level (%) -> Z score mapping.
Z_MAP = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}
DEFAULT_Z = 1.65

SES_ALPHAS = [0.2, 0.5, 0.8]
MA_WINDOW = 4

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
# CSV normalisation & validation
# --------------------------------------------------------------------------- #
def normalize_csv_format(df):
    """
    Auto-detect and normalise an uploaded CSV to Inventra's internal schema.

    Supports two formats:
      - Native Inventra  : transaction_date, product_name, quantity,
                           final_amount, unit_price  (+ optional extras)
      - UCI Online Retail: InvoiceNo, StockCode, Description, Quantity,
                           InvoiceDate, UnitPrice, CustomerID, Country

    Returns a dataframe with at least the REQUIRED_COLUMNS present.
    Non-UCI files are returned unchanged.
    """
    if not _UCI_SIGNATURE.issubset(set(df.columns)):
        return df  # already in native format (or unknown — let validate_csv catch it)

    df = df.copy()
    df = df.rename(columns={
        'Description': 'product_name',
        'InvoiceDate': 'transaction_date',
        'Quantity':    'quantity',
        'UnitPrice':   'unit_price',
    })
    df['final_amount']    = df['quantity'] * df['unit_price']
    df['discount_amount'] = 0.0
    df['store_name']      = df['Country'] if 'Country' in df.columns else 'Online'

    # Remove returns (negative qty), cancelled invoices, and admin/postage rows.
    df = df[df['quantity'] > 0]
    df = df[df['unit_price'] > 0]
    noise = df['product_name'].str.upper().str.startswith(_UCI_NOISE_PREFIXES, na=False)
    df = df[~noise]

    return df


def validate_csv(file_path, min_rows=10):
    """
    Validate an uploaded CSV. Returns (ok: bool, message: str, dataframe_or_None).
    Accepts Inventra native format and UCI Online Retail format (auto-detected).
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not read the CSV file: {exc}", None
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not read the CSV file: {exc}", None

    df = normalize_csv_format(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, (
            f"Missing required column(s): {', '.join(missing)}. "
            "Accepted: Inventra native (transaction_date, product_name, quantity, "
            "final_amount, unit_price) or UCI Online Retail "
            "(InvoiceNo, Description, Quantity, InvoiceDate, UnitPrice)."
        ), None

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


def detect_promotional_weeks(df_product):
    """
    Identify calendar weeks with abnormally high discounting.

    A week is flagged as promotional when its total discount_amount exceeds
    mean + 1 standard deviation across all weeks.  Returns a boolean Series
    indexed by week-end date.  Returns an all-False series when the
    discount_amount column is absent or variance is zero.
    """
    if 'discount_amount' not in df_product.columns or df_product.empty:
        return pd.Series(dtype=bool)

    weekly_disc = (
        df_product
        .set_index('transaction_date')['discount_amount']
        .resample('W').sum()
    )
    std = weekly_disc.std()
    if std == 0 or pd.isna(std):
        return pd.Series([False] * len(weekly_disc), index=weekly_disc.index)

    threshold = weekly_disc.mean() + std
    return weekly_disc > threshold


# ---------------------------------------------------------------------------
# Malaysian Public Holidays — federal + major state holidays, 2023–2026.
# Islamic and Hindu holidays follow lunar calendars; dates marked (≈) are
# confirmed official or government-gazette dates.
# ---------------------------------------------------------------------------
MY_PUBLIC_HOLIDAYS: frozenset = frozenset([
    # ── 2023 ──────────────────────────────────────────────────────────────
    date(2023, 1,  1),              # New Year's Day
    date(2023, 1, 22), date(2023, 1, 23),  # Chinese New Year (Rabbit)
    date(2023, 2,  1),              # Federal Territory Day
    date(2023, 2,  4),              # Thaipusam
    date(2023, 4, 21), date(2023, 4, 22),  # Hari Raya Aidilfitri
    date(2023, 5,  1),              # Labour Day
    date(2023, 5,  4),              # Wesak Day
    date(2023, 6,  5),              # Yang di-Pertuan Agong Birthday
    date(2023, 6, 28), date(2023, 6, 29),  # Hari Raya Aidiladha
    date(2023, 7, 19),              # Awal Muharram
    date(2023, 8, 31),              # National Day (Merdeka)
    date(2023, 9, 16),              # Malaysia Day
    date(2023, 9, 27),              # Prophet Muhammad's Birthday
    date(2023, 11, 12),             # Deepavali
    date(2023, 12, 25),             # Christmas
    # ── 2024 ──────────────────────────────────────────────────────────────
    date(2024, 1,  1),              # New Year's Day
    date(2024, 2,  1),              # Federal Territory Day
    date(2024, 2, 10), date(2024, 2, 11),  # Chinese New Year (Dragon)
    date(2024, 2, 24),              # Thaipusam
    date(2024, 4, 10), date(2024, 4, 11),  # Hari Raya Aidilfitri
    date(2024, 5,  1),              # Labour Day
    date(2024, 5, 22),              # Wesak Day
    date(2024, 6,  3),              # Yang di-Pertuan Agong Birthday
    date(2024, 6, 17),              # Hari Raya Aidiladha
    date(2024, 7,  7),              # Awal Muharram
    date(2024, 8, 31),              # National Day
    date(2024, 9, 16),              # Malaysia Day / Prophet's Birthday (combined)
    date(2024, 10, 31),             # Deepavali
    date(2024, 12, 25),             # Christmas
    # ── 2025 ──────────────────────────────────────────────────────────────
    date(2025, 1,  1),              # New Year's Day
    date(2025, 1, 29), date(2025, 1, 30),  # Chinese New Year (Snake)
    date(2025, 2,  1),              # Federal Territory Day
    date(2025, 2, 11),              # Thaipusam
    date(2025, 3, 30), date(2025, 3, 31),  # Hari Raya Aidilfitri
    date(2025, 5,  1),              # Labour Day
    date(2025, 5, 12),              # Wesak Day
    date(2025, 6,  2),              # Yang di-Pertuan Agong Birthday
    date(2025, 6,  6), date(2025, 6,  7),  # Hari Raya Aidiladha
    date(2025, 6, 26),              # Awal Muharram
    date(2025, 8, 31),              # National Day
    date(2025, 9,  5),              # Prophet Muhammad's Birthday
    date(2025, 9, 16),              # Malaysia Day
    date(2025, 10, 20),             # Deepavali
    date(2025, 12, 25),             # Christmas
    # ── 2026 (≈ lunar-based dates estimated from Hijri/lunisolar calendars) ─
    date(2026, 1,  1),              # New Year's Day
    date(2026, 2,  1),              # Federal Territory Day
    date(2026, 2, 17), date(2026, 2, 18),  # Chinese New Year (Horse)
    date(2026, 3, 20), date(2026, 3, 21),  # Hari Raya Aidilfitri (≈)
    date(2026, 5,  1),              # Labour Day
    date(2026, 5, 27), date(2026, 5, 28),  # Hari Raya Aidiladha (≈)
    date(2026, 6,  1),              # Yang di-Pertuan Agong Birthday
    date(2026, 6, 16),              # Awal Muharram (≈)
    date(2026, 8, 25),              # Prophet Muhammad's Birthday (≈)
    date(2026, 8, 31),              # National Day
    date(2026, 9, 16),              # Malaysia Day
    date(2026, 11,  8),             # Deepavali (≈)
    date(2026, 12, 25),             # Christmas
])


def detect_holiday_weeks(weekly_index):
    """
    Flag weeks that overlap at least one Malaysian public holiday.

    `weekly_index` is a DatetimeIndex of week-end dates produced by
    pandas resample('W').  A week is flagged when any of its 7 calendar
    days (Mon–Sun) falls on a date in MY_PUBLIC_HOLIDAYS.

    Returns a boolean Series indexed by weekly_index.
    """
    result = []
    for week_end in weekly_index:
        week_start = week_end - pd.Timedelta(days=6)
        days = pd.date_range(week_start.normalize(), week_end.normalize(), freq='D')
        result.append(any(d.date() in MY_PUBLIC_HOLIDAYS for d in days))
    return pd.Series(result, index=weekly_index)


def _compute_baseline(df_product):
    """
    Internal helper: compute the demand baseline and return all intermediate masks.

    Returns (baseline, promo_count, holiday_count, promo_aligned, holiday_aligned)
    where the masks are boolean Series indexed by the weekly DatetimeIndex.
    Use weekly_demand_baseline() for callers that don't need the masks, and
    weekly_demand_baseline_with_masks() when chart annotation requires them.
    """
    weekly = weekly_demand_series(df_product)
    _empty_mask = pd.Series(dtype=bool)
    if weekly.empty:
        return weekly, 0, 0, _empty_mask, _empty_mask

    promo_mask   = detect_promotional_weeks(df_product)
    holiday_mask = detect_holiday_weeks(weekly.index)

    promo_aligned   = promo_mask.reindex(weekly.index, fill_value=False)
    holiday_aligned = holiday_mask.reindex(weekly.index, fill_value=False)
    combined_mask   = promo_aligned | holiday_aligned

    promo_count   = int(promo_aligned.sum())
    holiday_count = int(holiday_aligned.sum())

    if not combined_mask.any() or not (~combined_mask).any():
        return weekly, promo_count, holiday_count, promo_aligned, holiday_aligned

    baseline_median = weekly[~combined_mask].median()
    baseline = weekly.copy()
    baseline[combined_mask] = baseline_median
    return baseline, promo_count, holiday_count, promo_aligned, holiday_aligned


def weekly_demand_baseline(df_product):
    """
    Compute a promotion- and holiday-adjusted weekly demand series for SES/MA training.

    Promotional weeks (high discount) and holiday weeks (Malaysian public holidays)
    are replaced with the median demand of unaffected weeks so that forecasting
    algorithms are not biased by temporary demand spikes.

    Returns:
        baseline      (pd.Series) – adjusted weekly demand (same index as raw)
        promo_weeks   (int)       – promotional weeks replaced
        holiday_weeks (int)       – public-holiday weeks replaced
    """
    baseline, promo_count, holiday_count, _, _ = _compute_baseline(df_product)
    return baseline, promo_count, holiday_count


def weekly_demand_baseline_with_masks(df_product):
    """
    Same as weekly_demand_baseline() but also returns the boolean masks so callers
    can identify which specific weeks were adjusted (e.g. for chart annotation).

    Returns:
        baseline        (pd.Series)  – adjusted weekly demand
        promo_count     (int)        – promotional weeks replaced
        holiday_count   (int)        – holiday weeks replaced
        promo_aligned   (pd.Series)  – bool mask, True = promotional week
        holiday_aligned (pd.Series)  – bool mask, True = holiday week
    """
    return _compute_baseline(df_product)


# --------------------------------------------------------------------------- #
# Cold-start prior blending
# --------------------------------------------------------------------------- #
PRIOR_FULL_TRUST_WEEKS = 4  # weeks of real data needed before prior is ignored


def blend_with_prior(computed_avg, prior_avg, data_weeks):
    """
    Blend a transaction-computed demand average with a user-supplied prior estimate.

    Weight shifts linearly from pure prior (0 real weeks) to pure computed
    (>= PRIOR_FULL_TRUST_WEEKS weeks).  Inspired by Bayesian prior updating:
    as evidence accumulates, the prior loses influence.

    Args:
        computed_avg: avg_weekly_demand calculated from real StockTransactions
        prior_avg:    user-entered estimate (prior_avg_demand on the Product)
        data_weeks:   number of calendar weeks with at least one recorded sale

    Returns:
        blended float — the demand estimate to use for analysis
    """
    if prior_avg <= 0:
        return computed_avg  # no prior set, use computed as-is
    if data_weeks >= PRIOR_FULL_TRUST_WEEKS:
        return computed_avg  # enough real data, ignore prior completely
    weight = data_weeks / PRIOR_FULL_TRUST_WEEKS  # 0.0 → 1.0
    return weight * computed_avg + (1.0 - weight) * prior_avg


# --------------------------------------------------------------------------- #
# Demand statistics
# --------------------------------------------------------------------------- #
def demand_stats(weekly):
    """Return (avg_weekly_demand, std_weekly_demand, daily_velocity)."""
    avg = _safe(weekly.mean()) if len(weekly) else 0.0
    std = _safe(weekly.std()) if len(weekly) > 1 else 0.0
    velocity = avg / 7 if avg else 0.0
    return avg, std, velocity


def demand_stats_from_transactions(sale_transactions):
    """
    Recalculate demand stats from a StockTransaction queryset (SALE only).
    Groups sales by week and feeds into demand_stats().
    Returns (avg_weekly_demand, std_weekly_demand, daily_velocity).
    """
    records = list(sale_transactions.values('date', 'quantity'))
    if not records:
        return 0.0, 0.0, 0.0
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df = df.set_index('date').sort_index()
    weekly = df['quantity'].resample('W').sum()
    weekly = weekly.replace(0, np.nan).ffill().dropna()
    return demand_stats(weekly)


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
    Returns a dict with per-method MAE/RMSE, forecasts, the chosen alpha, best method,
    date labels for the chart x-axis, and a 4-week ahead projection for each method.
    """
    demand = list(weekly)
    ses = best_ses(weekly)
    ma_fc = ma_forecast(demand)
    mae_ma, rmse_ma = mae(demand, ma_fc), rmse(demand, ma_fc)
    best_method = 'SES' if ses['mae'] <= mae_ma else 'MA-4'

    # Date x-axis labels — only available when weekly carries a DatetimeIndex.
    if isinstance(getattr(weekly, 'index', None), pd.DatetimeIndex) and len(weekly) > 0:
        week_dates   = [d.strftime('%Y-%m-%d') for d in weekly.index]
        last_date    = weekly.index[-1]
        future_dates = [
            (last_date + pd.Timedelta(weeks=i + 1)).strftime('%Y-%m-%d')
            for i in range(4)
        ]
    else:
        week_dates   = []
        future_dates = []

    # 4-week ahead out-of-sample forecast (flat SES level; MA extended with own preds).
    ses_future = _ses_forecast_ahead(demand, ses['alpha'], 4)
    ma_future  = _ma_forecast_ahead(demand, 4)

    return {
        'weeks':        demand,
        'week_dates':   week_dates,
        'best_alpha':   ses['alpha'],
        'ses_forecast': ses['forecast'],
        'ma_forecast':  ma_fc,
        'mae_ses':      ses['mae'],
        'rmse_ses':     ses['rmse'],
        'mae_ma':       mae_ma,
        'rmse_ma':      rmse_ma,
        'best_method':  best_method,
        'future_dates': future_dates,
        'ses_future':   [round(v, 2) for v in ses_future],
        'ma_future':    [round(v, 2) for v in ma_future],
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


def run_simulation(weekly_demand, rop, order_quantity, initial_stock, lead_time=2):
    """
    Compare two reordering policies over the historical weekly demand series.

    Policy 1 — Naive ROP (no safety buffer):
        Reorder when stock <= avg_weekly_demand × lead_time with zero safety stock.
        Represents intuition-based ordering with no statistical buffer.
        Standard academic baseline for safety-stock studies
        (Chopra & Meindl, Supply Chain Management, Ch. 11).

    Policy 2 — DSS Forecast-based ROP (with safety stock):
        Reorder when stock <= calculated ROP (includes safety stock).
    """
    demand_series = pd.Series(list(weekly_demand))
    avg, _, _ = demand_stats(demand_series) if len(demand_series) else (0.0, 0.0, 0.0)
    naive_rop = reorder_point(avg, lead_time, 0)  # ss=0 → no buffer

    static = simulate(weekly_demand, naive_rop, order_quantity, initial_stock)
    dss    = simulate(weekly_demand, rop,       order_quantity, initial_stock)
    return {
        'static':               static,
        'dss':                  dss,
        'naive_rop':            round(naive_rop, 1),
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
    Compute demand statistics for a single product. Prefer all_product_stats()
    when processing multiple products — it is significantly faster.
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


def all_product_stats(df):
    """
    Compute demand statistics for every product in a single groupby pass.
    ~50-100x faster than calling per_product_stats() in a loop for large datasets.
    Returns dict keyed by product_name.
    """
    results = {}
    # Single groupby+resample — avoids O(n_products × n_rows) repeated filtering.
    grouped = (
        df.set_index('transaction_date')
        .groupby('product_name', sort=False)['quantity']
    )
    for name, series in grouped:
        weekly = series.resample('W').sum()
        weekly = weekly.replace(0, np.nan).ffill().dropna()
        avg, std, velocity = demand_stats(weekly)
        results[name] = {
            'avg_weekly_demand': avg,
            'std_weekly_demand': std,
            'daily_velocity': velocity,
            'weeks': len(weekly),
        }
    return results


def analyze(weekly, weekly_revenue, current_stock, unit_price, lead_time, z,
            order_quantity, promo_weeks_count=0, holiday_weeks_count=0):
    """
    Full analysis for a product given its weekly demand/revenue history and settings.
    Bundles forecasting, inventory metrics, simulation and financials for the dashboard.
    promo_weeks_count:   promotional weeks excluded from the baseline (informational only).
    holiday_weeks_count: public-holiday weeks excluded from the baseline (informational only).
    """
    avg, std, velocity = demand_stats(weekly)
    ss = safety_stock(z, std, lead_time)
    rop = reorder_point(avg, lead_time, ss)
    status = stock_status(current_stock, rop, ss)

    comparison = compare_algorithms(weekly)
    sim = run_simulation(weekly, rop, order_quantity, current_stock, lead_time)
    fin = financials(
        current_stock, unit_price, ss, weekly_revenue,
        sim['static']['stockouts'], sim['dss']['stockouts'],
    )

    return {
        'avg_weekly_demand':    avg,
        'std_weekly_demand':    std,
        'daily_velocity':       velocity,
        'safety_stock':         ss,
        'reorder_point':        rop,
        'sells_out_in':         sells_out_in(current_stock, velocity),
        'status':               status,
        'replenishment_qty':    replenishment_qty(rop, current_stock, ss),
        'forecast':             comparison,
        'simulation':           sim,
        'financials':           fin,
        'data_weeks':           len(weekly),
        'promo_weeks_count':    promo_weeks_count,
        'holiday_weeks_count':  holiday_weeks_count,
    }


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
def _ses_forecast_ahead(train, alpha, n_steps):
    """
    SES n-step-ahead out-of-sample forecast.
    For SES (no trend/seasonality), the optimal forecast for all horizons
    is the last estimated level — a flat line forward.
    """
    if not train:
        return [0.0] * n_steps
    fc = ses_forecast(list(train), alpha)
    return [_safe(fc[-1])] * n_steps


def _ma_forecast_ahead(train, n_steps, window=MA_WINDOW):
    """
    MA n-step-ahead out-of-sample forecast.
    Extends with its own forecasts once the look-back window overlaps the future.
    """
    history = list(train)
    preds = []
    for _ in range(n_steps):
        available = history + preds
        w = available[-window:] if len(available) >= window else available
        preds.append(float(np.mean(w)) if w else 0.0)
    return preds


def backtest_product(weekly, test_weeks=4):
    """
    Hold-out backtest for a single product's weekly demand series.
    Splits into train (all but last test_weeks) and test (last test_weeks).
    Returns a result dict, or None if there is insufficient data
    (requires at least 4 training weeks + test_weeks).
    """
    demand = list(weekly)
    if len(demand) < test_weeks + 4:
        return None

    train = demand[:-test_weeks]
    test  = demand[-test_weeks:]

    ses        = best_ses(pd.Series(train))
    ses_preds  = _ses_forecast_ahead(train, ses['alpha'], test_weeks)
    ma_preds   = _ma_forecast_ahead(train, test_weeks)

    mae_ses_v  = mae(test, ses_preds)
    rmse_ses_v = rmse(test, ses_preds)
    mae_ma_v   = mae(test, ma_preds)
    rmse_ma_v  = rmse(test, ma_preds)

    return {
        'train_weeks':  len(train),
        'test_weeks':   test_weeks,
        'actual':       [round(v, 2) for v in test],
        'ses_forecast': [round(v, 2) for v in ses_preds],
        'ma_forecast':  [round(v, 2) for v in ma_preds],
        'best_alpha':   ses['alpha'],
        'mae_ses':      round(mae_ses_v, 2),
        'rmse_ses':     round(rmse_ses_v, 2),
        'mae_ma':       round(mae_ma_v, 2),
        'rmse_ma':      round(rmse_ma_v, 2),
        'best_method':  'SES' if mae_ses_v <= mae_ma_v else 'MA-4',
    }


def backtest_all_products(df, test_weeks=4):
    """
    Run hold-out backtest for every product in the dataframe.
    Uses the promotion-adjusted baseline series (same as the live forecast).
    Returns (results_list, summary_dict).
    results_list: one dict per product (includes 'product' key).
    summary_dict: aggregate MAE/RMSE, win counts, overall best method.
    """
    results = []
    for name, group in df.groupby('product_name', sort=True):
        weekly, _, _ = weekly_demand_baseline(group)
        bt = backtest_product(weekly, test_weeks)
        if bt is None:
            continue
        bt['product'] = name
        results.append(bt)

    if not results:
        return [], {}

    mae_ses_vals  = [r['mae_ses']  for r in results]
    mae_ma_vals   = [r['mae_ma']   for r in results]
    rmse_ses_vals = [r['rmse_ses'] for r in results]
    rmse_ma_vals  = [r['rmse_ma']  for r in results]
    ses_wins = sum(1 for r in results if r['best_method'] == 'SES')

    summary = {
        'n_products':   len(results),
        'test_weeks':   test_weeks,
        'avg_mae_ses':  round(float(np.mean(mae_ses_vals)),  2),
        'avg_mae_ma':   round(float(np.mean(mae_ma_vals)),   2),
        'avg_rmse_ses': round(float(np.mean(rmse_ses_vals)), 2),
        'avg_rmse_ma':  round(float(np.mean(rmse_ma_vals)),  2),
        'ses_wins':     ses_wins,
        'ma_wins':      len(results) - ses_wins,
        'overall_best': 'SES' if np.mean(mae_ses_vals) <= np.mean(mae_ma_vals) else 'MA-4',
    }

    return results, summary
