"""Inventra views: authentication, the tabbed dashboard, products, what-if, admin and settings."""
import json

from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import get_object_or_404, redirect, render

from . import analytics, services
from .forms import CSVUploadForm, InventorySettingsForm, ProductForm, SignupForm
from .models import Product, PurchaseOrder, StockTransaction, UserCSV


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _has_csv(user):
    return UserCSV.objects.filter(user=user, is_processed=True).exists()


def _selected_product(user, request):
    """Resolve which product the dashboard is analysing (GET param, saved setting, or first).
    Persists the choice to InventorySettings so it survives page navigation."""
    qs = Product.objects.filter(user=user)
    pid = request.GET.get('product')
    if pid:
        product = qs.filter(pk=pid).first()
        if product:
            inv = services.get_settings(user)
            if inv.selected_product_id != product.pk:
                inv.selected_product = product
                inv.save(update_fields=['selected_product'])
            return product
    inv = services.get_settings(user)
    if inv.selected_product and inv.selected_product.user_id == user.id:
        return inv.selected_product
    return qs.first()


def _forecast_chart(forecast, promo_dates=None, holiday_dates=None):
    """
    Build the data payload for the SES vs MA forecast Plotly chart.
    Uses real date labels on the x-axis when available (CSV products), falling
    back to integer week numbers for manual products without date history.
    Includes a 4-week forward projection and optional week-type markers.
    """
    week_dates = forecast.get('week_dates', [])
    x_axis = week_dates if week_dates else list(range(1, len(forecast['weeks']) + 1))
    return {
        'weeks':        x_axis,
        'actual':       [round(v, 2) for v in forecast['weeks']],
        'ses':          [round(v, 2) for v in forecast['ses_forecast']],
        'ma':           [round(v, 2) for v in forecast['ma_forecast']],
        'future_dates': forecast.get('future_dates', []),
        'ses_future':   forecast.get('ses_future', []),
        'ma_future':    forecast.get('ma_future', []),
        'promo_dates':  promo_dates or [],
        'holiday_dates': holiday_dates or [],
    }


def _simulation_chart(sim):
    static_levels = sim['static']['inventory_levels']
    return {
        'weeks': list(range(1, len(static_levels) + 1)),
        'static': static_levels,
        'dss': sim['dss']['inventory_levels'],
    }


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def signup_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Welcome! Upload your CSV to get started.")
            return redirect('settings')
    else:
        form = SignupForm()
    return render(request, 'dashboard/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            if not request.POST.get('remember_me'):
                request.session.set_expiry(0)  # expire at browser close
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'dashboard/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


# --------------------------------------------------------------------------- #
# Main dashboard (tabbed)
# --------------------------------------------------------------------------- #
@login_required
def index(request):
    if not _has_csv(request.user) and not Product.objects.filter(user=request.user).exists():
        messages.info(request, "Upload your CSV to get started.")
        return redirect('settings')

    products = Product.objects.filter(user=request.user)
    selected = _selected_product(request.user, request)
    orders = PurchaseOrder.objects.filter(user=request.user)

    context = {
        'products': products,
        'selected': selected,
        'summary': services.products_summary(request.user)[0],
        'recent_transactions': StockTransaction.objects.filter(user=request.user)[:5],
        'orders': orders,
        'order_counts': {
            'suggested': orders.filter(status='Suggested').count(),
            'ordered': orders.filter(status='Ordered').count(),
            'received': orders.filter(status='Received').count(),
        },
    }

    if selected:
        analysis = services.analyze_product(request.user, selected)
        context['analysis'] = analysis
        context['forecast_chart'] = _forecast_chart(
            analysis['forecast'],
            promo_dates=analysis.get('promo_week_dates', []),
            holiday_dates=analysis.get('holiday_week_dates', []),
        )
        context['simulation_chart'] = _simulation_chart(analysis['simulation'])
        df = services.load_clean_dataframe(request.user)
        if df is not None and selected.has_csv_history:
            rev = analytics.weekly_revenue_series(df[df['product_name'] == selected.name])
            context['revenue_chart'] = {
                'weeks': [d.strftime('%Y-%m-%d') for d in rev.index],
                'values': [round(v, 2) for v in rev.tolist()],
            }
    return render(request, 'dashboard/index.html', context)


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #
@login_required
def products_view(request):
    counts, rows = services.products_summary(request.user)
    bar = {
        'names': [r['product'].name for r in rows],
        'stock': [r['product'].current_stock for r in rows],
        'rop': [round(r['reorder_point'], 1) for r in rows],
    }
    critical = sorted(
        [r for r in rows if r['status'] == 'Critical'],
        key=lambda r: r['product'].current_stock,
    )[:5]
    context = {
        'counts': counts,
        'rows': rows,
        'donut': {
            'safe': counts['safe'],
            'order_soon': counts['order_soon'],
            'critical': counts['critical'],
        },
        'bar_chart': bar,
        'critical_chart': {
            'names': [r['product'].name for r in critical],
            'stock': [r['product'].current_stock for r in critical],
        },
    }
    return render(request, 'dashboard/products.html', context)


@login_required
def product_detail(request, pk):
    product = get_object_or_404(Product, pk=pk, user=request.user)
    analysis = services.analyze_product(request.user, product)
    context = {
        'product': product,
        'analysis': analysis,
        'forecast_chart': _forecast_chart(
            analysis['forecast'],
            promo_dates=analysis.get('promo_week_dates', []),
            holiday_dates=analysis.get('holiday_week_dates', []),
        ),
        'simulation_chart': _simulation_chart(analysis['simulation']),
    }
    df = services.load_clean_dataframe(request.user)
    if df is not None and product.has_csv_history:
        rev = analytics.weekly_revenue_series(df[df['product_name'] == product.name])
        context['revenue_chart'] = {
            'weeks': [d.strftime('%Y-%m-%d') for d in rev.index],
            'values': [round(v, 2) for v in rev.tolist()],
        }
    # Stock history chart — all transactions for this product, ordered oldest first.
    txns = list(StockTransaction.objects.filter(
        product=product, user=request.user,
    ).order_by('date'))
    if txns:
        context['stock_history'] = {
            'dates':  [t.date.strftime('%Y-%m-%d') for t in txns],
            'levels': [t.stock_after for t in txns],
            'types':  [t.transaction_type for t in txns],
            'rop':    round(analysis['reorder_point'], 1),
            'ss':     round(analysis['safety_stock'], 1),
        }
    return render(request, 'dashboard/product_detail.html', context)


# --------------------------------------------------------------------------- #
# What-If tool
# --------------------------------------------------------------------------- #
@login_required
def whatif_view(request):
    selected = _selected_product(request.user, request)
    inv = services.get_settings(request.user)
    context = {
        'products': Product.objects.filter(user=request.user),
        'selected': selected,
        'defaults': {
            'current_stock': selected.current_stock if selected else 25,
            'lead_time': inv.lead_time,
            'avg_weekly_demand': selected.avg_weekly_demand if selected else 0,
            'std_weekly_demand': selected.std_weekly_demand if selected else 3.05,
            'daily_velocity': selected.daily_velocity if selected else 0,
        },
    }
    return render(request, 'dashboard/whatif.html', context)


# --------------------------------------------------------------------------- #
# Transactions & orders (POST actions)
# --------------------------------------------------------------------------- #
@login_required
def record_transaction(request):
    if request.method != 'POST':
        return redirect('index')
    product = get_object_or_404(Product, pk=request.POST.get('product'), user=request.user)
    ttype = request.POST.get('transaction_type')
    nxt = request.POST.get('next', 'index')
    try:
        qty = float(request.POST.get('quantity'))
    except (TypeError, ValueError):
        messages.error(request, "Enter a valid quantity.")
        return redirect(nxt)
    if qty <= 0:
        messages.error(request, "Quantity must be greater than zero.")
        return redirect(nxt)

    if ttype == 'SALE':
        product.current_stock = max(0, product.current_stock - qty)
    elif ttype == 'RESTOCK':
        product.current_stock += qty
    else:
        messages.error(request, "Unknown transaction type.")
        return redirect(nxt)
    product.save()
    StockTransaction.objects.create(
        user=request.user, product=product, transaction_type=ttype,
        quantity=qty, stock_after=product.current_stock,
        notes=request.POST.get('notes', ''),
    )
    if ttype == 'SALE':
        sale_txns = StockTransaction.objects.filter(product=product, transaction_type='SALE')
        avg, std, velocity = analytics.demand_stats_from_transactions(sale_txns)
        product.avg_weekly_demand = avg
        product.std_weekly_demand = std
        product.daily_velocity = velocity
        product.save(update_fields=['avg_weekly_demand', 'std_weekly_demand', 'daily_velocity'])
    messages.success(request, f"Recorded {ttype.lower()} of {qty:g} {product.name}.")
    return redirect(nxt)


@login_required
def create_order(request):
    if request.method != 'POST':
        return redirect('index')
    product = get_object_or_404(Product, pk=request.POST.get('product'), user=request.user)
    nxt = request.POST.get('next', 'index')
    try:
        qty = float(request.POST.get('quantity'))
    except (TypeError, ValueError):
        messages.error(request, "Enter a valid order quantity.")
        return redirect(nxt)
    PurchaseOrder.objects.create(
        user=request.user, product=product, quantity=qty,
        status='Suggested', notes=request.POST.get('notes', ''),
    )
    messages.success(request, f"Created purchase order for {qty:g} {product.name}.")
    return redirect(nxt)


@login_required
def order_action(request, pk):
    if request.method != 'POST':
        return redirect('index')
    order = get_object_or_404(PurchaseOrder, pk=pk, user=request.user)
    action = request.POST.get('action')
    nxt = request.POST.get('next', 'index')
    if action == 'order':
        order.status = 'Ordered'
        order.save()
        messages.success(request, "Order marked as Ordered.")
    elif action == 'receive':
        order.status = 'Received'
        order.save()
        product = order.product
        product.current_stock += order.quantity
        product.save()
        StockTransaction.objects.create(
            user=request.user, product=product, transaction_type='RESTOCK',
            quantity=order.quantity, stock_after=product.current_stock,
            notes=f"Received PO #{order.pk}",
        )
        messages.success(request, f"Received {order.quantity:g} {product.name}; stock updated.")
    elif action == 'delete':
        order.delete()
        messages.success(request, "Order removed.")
    return redirect(nxt)


# --------------------------------------------------------------------------- #
# Admin panel
# --------------------------------------------------------------------------- #
@login_required
def admin_panel(request):
    context = {
        'products': Product.objects.filter(user=request.user),
        'orders': PurchaseOrder.objects.filter(user=request.user),
        'transactions': StockTransaction.objects.filter(user=request.user)[:20],
        'product_form': ProductForm(),
    }
    return render(request, 'dashboard/admin_panel.html', context)


@login_required
def product_create(request):
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            product = form.save(commit=False)
            product.user = request.user
            product.daily_velocity = (product.avg_weekly_demand or 0) / 7
            product.has_csv_history = False
            product.save()
            messages.success(request, f"Added product '{product.name}'.")
        else:
            messages.error(request, "Could not add product. Check the form values.")
    return redirect('admin_panel')


@login_required
def product_edit(request, pk):
    product = get_object_or_404(Product, pk=pk, user=request.user)
    if request.method == 'POST':
        form = ProductForm(request.POST, instance=product)
        if form.is_valid():
            product = form.save(commit=False)
            product.daily_velocity = (product.avg_weekly_demand or 0) / 7
            product.save()
            messages.success(request, f"Updated '{product.name}'.")
            return redirect('admin_panel')
    else:
        form = ProductForm(instance=product)
    return render(request, 'dashboard/product_edit.html', {'form': form, 'product': product})


@login_required
def product_delete(request, pk):
    product = get_object_or_404(Product, pk=pk, user=request.user)
    if request.method == 'POST':
        name = product.name
        product.delete()
        messages.success(request, f"Deleted '{name}'.")
    return redirect('admin_panel')


@login_required
def clear_history(request):
    if request.method == 'POST':
        StockTransaction.objects.filter(user=request.user).delete()
        messages.success(request, "Transaction history cleared.")
    return redirect('admin_panel')


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
@login_required
def backtest_view(request):
    df = services.load_clean_dataframe(request.user)
    if df is None:
        messages.info(request, "Upload a CSV to run the backtest.")
        return redirect('settings')

    try:
        test_weeks = int(request.GET.get('test_weeks', 4))
    except (TypeError, ValueError):
        test_weeks = 4
    test_weeks = max(2, min(test_weeks, 8))

    results, summary = analytics.backtest_all_products(df, test_weeks=test_weeks)

    # Build Plotly grouped-bar chart data — serialised as JSON so the template
    # can inject it directly into JS without Python→JS repr issues.
    mae_chart_json = None
    if results:
        sorted_results = sorted(results, key=lambda r: r['mae_ses'], reverse=True)
        mae_chart_json = json.dumps({
            'products': [r['product'] for r in sorted_results],
            'mae_ses':  [r['mae_ses']  for r in sorted_results],
            'mae_ma':   [r['mae_ma']   for r in sorted_results],
        })

    return render(request, 'dashboard/backtest.html', {
        'results':        results,
        'summary':        summary,
        'mae_chart_json': mae_chart_json,
        'test_weeks':     test_weeks,
    })


# --------------------------------------------------------------------------- #
# Settings (CSV upload + inventory settings)
# --------------------------------------------------------------------------- #
@login_required
def settings_view(request):
    user_csv = UserCSV.objects.filter(user=request.user).first()
    inv = services.get_settings(request.user)

    upload_form = CSVUploadForm()
    settings_form = InventorySettingsForm(instance=inv)

    if request.method == 'POST':
        if 'upload_csv' in request.POST:
            upload_form = CSVUploadForm(request.POST, request.FILES)
            if upload_form.is_valid():
                f = upload_form.cleaned_data['csv_file']
                path = services.store_uploaded_csv(request.user, f)
                ok, msg, _ = analytics.validate_csv(path)
                if not ok:
                    messages.error(request, msg)
                else:
                    count = services.process_csv_for_user(request.user, path, f.name)
                    messages.success(request, f"CSV processed — {count} products loaded.")
                    return redirect('index')
        elif 'save_settings' in request.POST:
            settings_form = InventorySettingsForm(request.POST, instance=inv)
            if settings_form.is_valid():
                settings_form.save()
                messages.success(request, "Inventory settings saved.")
                return redirect('settings')

    return render(request, 'dashboard/settings.html', {
        'user_csv': user_csv,
        'upload_form': upload_form,
        'settings_form': settings_form,
    })
