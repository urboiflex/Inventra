"""
Service layer: bridges the pure analytics engine (`analytics.py`) to Django models.

Responsibilities:
  - store an uploaded CSV/XLSX under media/user_csvs/<user_id>/data.csv
  - process the CSV into per-user Product rows with demand statistics
  - load the cleaned dataframe back for analysis (with in-memory cache)
  - assemble the full analysis context for a single product
"""
import os
import tempfile

import pandas as pd
from django.conf import settings
from django.db import transaction

from . import analytics
from .models import InventorySettings, Product, StockTransaction, UserCSV

# Maximum products to import from a single CSV. Caps very large datasets (e.g. UCI
# Online Retail with 4000+ SKUs) to the top N by total revenue so the UI stays usable.
MAX_PRODUCTS = 100

# ---------------------------------------------------------------------------
# In-process dataframe cache: keyed by (user_id, file_mtime).
# Avoids re-parsing the CSV on every product switch.  Invalidated automatically
# whenever the file changes (new upload updates mtime).
# ---------------------------------------------------------------------------
_df_cache: dict = {}


def user_csv_path(user):
    """Absolute path where this user's canonical CSV lives."""
    return os.path.join(settings.MEDIA_ROOT, 'user_csvs', str(user.id), 'data.csv')


def store_uploaded_csv(user, uploaded_file):
    """
    Persist an uploaded CSV or XLSX file to the user's canonical CSV path.
    XLSX files are read into a dataframe and written out as CSV so the rest
    of the pipeline always sees a plain CSV at data.csv.
    Clears the dataframe cache for this user.
    Returns the destination path.
    """
    dest = user_csv_path(user)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
        suffix = '.xlsx' if uploaded_file.name.lower().endswith('.xlsx') else '.xls'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        try:
            df = pd.read_excel(tmp_path, engine='openpyxl')
            df.to_csv(dest, index=False)
        finally:
            os.unlink(tmp_path)
    else:
        with open(dest, 'wb') as fh:
            for chunk in uploaded_file.chunks():
                fh.write(chunk)

    _df_cache.pop(user.id, None)  # invalidate stale cache
    return dest


def load_clean_dataframe(user):
    """
    Read, normalise and clean the user's stored CSV.
    Result is cached in-process by file mtime — repeated calls within the same
    server process are near-instant after the first load.
    Returns a dataframe or None if the file is absent or unreadable.
    """
    path = user_csv_path(user)
    if not os.path.exists(path):
        _df_cache.pop(user.id, None)
        return None

    mtime = os.path.getmtime(path)
    cached = _df_cache.get(user.id)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        try:
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        df = analytics.normalize_csv_format(df)
        df = analytics.clean_dataframe(df)
        _df_cache[user.id] = (mtime, df)
        return df
    except Exception:  # noqa: BLE001
        return None


@transaction.atomic
def process_csv_for_user(user, file_path, original_filename):
    """
    Replace the user's product set with products derived from the CSV.

    Optimisations vs the naive loop:
      - all_product_stats() computes demand stats in a single groupby pass
        instead of one filter+resample per product (50-100x faster on large data)
      - Products capped to MAX_PRODUCTS top by revenue so the UI stays usable
        for large datasets like UCI Online Retail (4000+ SKUs)
      - bulk_create() inserts all products in one SQL statement

    Returns the number of products created.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
    df = analytics.normalize_csv_format(df)
    df = analytics.clean_dataframe(df)

    # Cap to top MAX_PRODUCTS by total revenue so the UI stays usable.
    all_names = df['product_name'].dropna().unique()
    if len(all_names) > MAX_PRODUCTS:
        top_names = (
            df.groupby('product_name')['final_amount'].sum()
            .nlargest(MAX_PRODUCTS)
            .index
        )
        df = df[df['product_name'].isin(top_names)]

    # Wipe prior data (FK cascades remove transactions & POs).
    Product.objects.filter(user=user).delete()
    _df_cache.pop(user.id, None)

    # Compute stats for all products in a single groupby pass.
    all_stats = analytics.all_product_stats(df)

    # Collect last-seen unit price per product in one vectorised operation.
    last_prices = (
        df.sort_values('transaction_date')
        .groupby('product_name')['unit_price']
        .last()
    )

    products_to_create = []
    for name, stats in all_stats.items():
        unit_price = analytics._safe(last_prices.get(name, 10.0), 10.0) or 10.0
        products_to_create.append(Product(
            user=user,
            name=name,
            unit_price=unit_price,
            avg_weekly_demand=stats['avg_weekly_demand'],
            std_weekly_demand=stats['std_weekly_demand'],
            daily_velocity=stats['daily_velocity'],
            has_csv_history=stats['weeks'] > 0,
        ))

    Product.objects.bulk_create(products_to_create, batch_size=500)
    created = len(products_to_create)

    UserCSV.objects.update_or_create(
        user=user,
        defaults={
            'original_filename': original_filename,
            'file_path':         file_path,
            'is_processed':      True,
            'product_count':     created,
        },
    )
    return created


def get_settings(user):
    """Return (creating if needed) the user's InventorySettings."""
    obj, _ = InventorySettings.objects.get_or_create(user=user)
    return obj


def analyze_product(user, product, df=None):
    """
    Build the full analysis context for one product, combining its stored fields,
    the user's inventory settings and the historical CSV (when available).
    df can be pre-loaded and passed in to avoid redundant cache lookups when
    analysing multiple products in a single request.
    """
    inv = get_settings(user)
    z = inv.service_level_z
    lead_time = product.lead_time or inv.lead_time
    order_quantity = inv.order_quantity

    if df is None:
        df = load_clean_dataframe(user)

    if df is not None and product.has_csv_history:
        df_product = df[df['product_name'] == product.name]
        weekly, promo_count, holiday_count = analytics.weekly_demand_baseline(df_product)
        weekly_revenue      = analytics.weekly_revenue_series(df_product)
        prior_active        = False
    else:
        # Manual product: compute actual data weeks from transaction history.
        sale_qs    = StockTransaction.objects.filter(product=product, transaction_type='SALE')
        data_weeks = _count_sale_weeks(sale_qs)

        # Blend computed avg with user prior when data is sparse.
        blended_avg = analytics.blend_with_prior(
            product.avg_weekly_demand, product.prior_avg_demand, data_weeks
        )
        prior_active = (
            product.prior_avg_demand > 0
            and data_weeks < analytics.PRIOR_FULL_TRUST_WEEKS
        )

        effective_avg  = blended_avg if blended_avg > 0 else 0
        weekly         = pd.Series([effective_avg] * 8) if effective_avg else pd.Series(dtype='float64')
        weekly_revenue = pd.Series([effective_avg * product.unit_price] * 8) if effective_avg else pd.Series(dtype='float64')
        promo_count    = 0
        holiday_count  = 0

    result = analytics.analyze(
        weekly=weekly,
        weekly_revenue=weekly_revenue,
        current_stock=product.current_stock,
        unit_price=product.unit_price,
        lead_time=lead_time,
        z=z,
        order_quantity=order_quantity,
        promo_weeks_count=promo_count,
        holiday_weeks_count=holiday_count,
    )
    if not product.has_csv_history:
        result['data_weeks']    = data_weeks
        result['prior_active']  = prior_active
        result['prior_avg']     = product.prior_avg_demand
    else:
        result['prior_active']  = False
        result['prior_avg']     = 0
    result['product']        = product
    result['lead_time']      = lead_time
    result['service_level_z'] = z
    return result


def _count_sale_weeks(sale_qs):
    """Return the number of distinct calendar weeks that have at least one sale."""
    records = list(sale_qs.values('date', 'quantity'))
    if not records:
        return 0
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    weekly = df.set_index('date')['quantity'].resample('W').sum()
    return int((weekly > 0).sum())


def products_summary(user):
    """Tally products by status for summary widgets. Returns counts + per-product status list."""
    inv = get_settings(user)
    status_key = {'Safe': 'safe', 'Order Soon': 'order_soon', 'Critical': 'critical'}
    counts = {'total': 0, 'safe': 0, 'order_soon': 0, 'critical': 0}
    rows = []
    for product in Product.objects.filter(user=user):
        ss  = analytics.safety_stock(inv.service_level_z, product.std_weekly_demand,
                                     product.lead_time or inv.lead_time)
        rop = analytics.reorder_point(product.avg_weekly_demand,
                                      product.lead_time or inv.lead_time, ss)
        status = analytics.stock_status(product.current_stock, rop, ss)
        counts['total'] += 1
        counts[status_key[status]] += 1
        rows.append({
            'product':       product,
            'status':        status,
            'reorder_point': rop,
            'safety_stock':  ss,
            'sells_out_in':  analytics.sells_out_in(product.current_stock, product.daily_velocity),
        })
    return counts, rows
