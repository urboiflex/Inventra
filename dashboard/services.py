"""
Service layer: bridges the pure analytics engine (`analytics.py`) to Django models.

Responsibilities:
  - store an uploaded CSV under media/user_csvs/<user_id>/data.csv
  - process the CSV into per-user Product rows with demand statistics
  - load the cleaned dataframe back for analysis
  - assemble the full analysis context for a single product
"""
import os

import pandas as pd
from django.conf import settings
from django.db import transaction

from . import analytics
from .models import InventorySettings, Product, UserCSV


def user_csv_path(user):
    """Absolute path where this user's canonical CSV lives."""
    return os.path.join(settings.MEDIA_ROOT, 'user_csvs', str(user.id), 'data.csv')


def store_uploaded_csv(user, uploaded_file):
    """Persist an uploaded file object to the user's canonical CSV path. Returns the path."""
    dest = user_csv_path(user)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as fh:
        for chunk in uploaded_file.chunks():
            fh.write(chunk)
    return dest


def load_clean_dataframe(user):
    """Read and clean the user's stored CSV. Returns a dataframe or None if absent/unreadable."""
    path = user_csv_path(user)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return analytics.clean_dataframe(df)
    except Exception:  # noqa: BLE001 - a corrupt CSV should not crash the dashboard
        return None


@transaction.atomic
def process_csv_for_user(user, file_path, original_filename):
    """
    Replace the user's product set with products derived from the CSV.

    Wipes existing products (and their related transactions/orders via cascade),
    builds one Product per distinct product_name with computed demand statistics,
    and records/updates the UserCSV row. Returns the number of products created.
    """
    df = pd.read_csv(file_path)
    df = analytics.clean_dataframe(df)

    # Wipe prior data for a clean rebuild (FK cascades remove txns & POs).
    Product.objects.filter(user=user).delete()

    product_names = sorted(df['product_name'].dropna().unique())
    created = 0
    for name in product_names:
        stats = analytics.per_product_stats(df, name)
        df_product = df[df['product_name'] == name]
        # Use the most recent unit price seen for this product, falling back to default.
        unit_price = analytics._safe(df_product['unit_price'].iloc[-1], 10.0) if len(df_product) else 10.0
        Product.objects.create(
            user=user,
            name=name,
            unit_price=unit_price or 10.0,
            avg_weekly_demand=stats['avg_weekly_demand'],
            std_weekly_demand=stats['std_weekly_demand'],
            daily_velocity=stats['daily_velocity'],
            has_csv_history=stats['weeks'] > 0,
        )
        created += 1

    UserCSV.objects.update_or_create(
        user=user,
        defaults={
            'original_filename': original_filename,
            'file_path': file_path,
            'is_processed': True,
            'product_count': created,
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
    """
    inv = get_settings(user)
    z = inv.service_level_z
    lead_time = product.lead_time or inv.lead_time
    order_quantity = inv.order_quantity

    if df is None:
        df = load_clean_dataframe(user)

    if df is not None and product.has_csv_history:
        df_product = df[df['product_name'] == product.name]
        weekly = analytics.weekly_demand_series(df_product)
        weekly_revenue = analytics.weekly_revenue_series(df_product)
    else:
        # Manual product with no CSV history: synthesize a flat demand series so the
        # metrics still compute from the product's stored averages.
        weekly = pd.Series([product.avg_weekly_demand] * 8) if product.avg_weekly_demand else pd.Series(dtype='float64')
        weekly_revenue = pd.Series([product.avg_weekly_demand * product.unit_price] * 8) if product.avg_weekly_demand else pd.Series(dtype='float64')

    result = analytics.analyze(
        weekly=weekly,
        weekly_revenue=weekly_revenue,
        current_stock=product.current_stock,
        unit_price=product.unit_price,
        lead_time=lead_time,
        z=z,
        order_quantity=order_quantity,
    )
    result['product'] = product
    result['lead_time'] = lead_time
    result['service_level_z'] = z
    return result


def products_summary(user):
    """Tally products by status for summary widgets. Returns counts + per-product status list."""
    inv = get_settings(user)
    # template-friendly keys (no spaces) keyed off the analytics status strings
    status_key = {'Safe': 'safe', 'Order Soon': 'order_soon', 'Critical': 'critical'}
    counts = {'total': 0, 'safe': 0, 'order_soon': 0, 'critical': 0}
    rows = []
    for product in Product.objects.filter(user=user):
        ss = analytics.safety_stock(inv.service_level_z, product.std_weekly_demand,
                                     product.lead_time or inv.lead_time)
        rop = analytics.reorder_point(product.avg_weekly_demand,
                                      product.lead_time or inv.lead_time, ss)
        status = analytics.stock_status(product.current_stock, rop, ss)
        counts['total'] += 1
        counts[status_key[status]] += 1
        rows.append({
            'product': product,
            'status': status,
            'reorder_point': rop,
            'safety_stock': ss,
            'sells_out_in': analytics.sells_out_in(product.current_stock, product.daily_velocity),
        })
    return counts, rows
