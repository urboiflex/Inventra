# Data Quality Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Inventra's analytics engine with three academically defensible improvements: multi-format CSV ingestion (native + UCI Online Retail), promotional-week demand correction, and a credible simulation baseline.

**Architecture:** All logic stays inside the pure `analytics.py` layer (no Django imports). `services.py` calls the new helpers. Templates receive new context keys (`promo_weeks_count`, `naive_rop_label`) and render them in the Forecast and Simulation tabs.

**Tech Stack:** Django 4.x, pandas, numpy — no new dependencies required.

---

## File Map

| File | Change |
|---|---|
| `dashboard/analytics.py` | Add `normalize_csv_format()`, `detect_promotional_weeks()`, `weekly_demand_baseline()`. Update `run_simulation()` and `analyze()`. |
| `dashboard/services.py` | Call `normalize_csv_format()` in `process_csv_for_user()` and `load_clean_dataframe()`. Pass `promo_weeks_count` and `lead_time` from `analyze_product()`. |
| `dashboard/templates/dashboard/index.html` | Forecast tab: show promo-weeks badge. Simulation tab: update baseline label. |
| `dashboard/templates/dashboard/settings.html` | Add UCI format to accepted column instructions. |

---

## Task 1: Multi-format CSV normalisation (`analytics.py`)

**Files:**
- Modify: `dashboard/analytics.py` (after `REQUIRED_COLUMNS` constant, before `validate_csv`)

**What this does:** Auto-detects whether an uploaded CSV is the native Inventra format or the UCI Online Retail format (`InvoiceNo, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`). Normalises UCI columns to the internal schema and filters out returns, postage rows, and zero-price records so the rest of the pipeline sees a clean, uniform dataframe.

- [ ] **Step 1: Add UCI detection constant and `normalize_csv_format()` to `analytics.py`**

Insert immediately after the `REQUIRED_COLUMNS` line (line 15):

```python
# UCI Online Retail dataset column signature (subset check).
_UCI_SIGNATURE = {'InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice'}

# Rows whose Description starts with these strings are admin/postage, not products.
_UCI_NOISE_PREFIXES = ('POST', 'DOT', 'CRUK', 'BANK', 'MANUAL', 'AMAZONFEE', 'M', 'PADS', 'ADJUST')


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
        return df  # already in native format

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
```

- [ ] **Step 2: Update `validate_csv()` to normalise before checking columns**

Replace the existing `validate_csv` function body so normalisation runs first:

```python
def validate_csv(file_path, min_rows=10):
    """
    Validate an uploaded CSV. Returns (ok: bool, message: str, dataframe_or_None).
    Accepts both the native Inventra schema and the UCI Online Retail schema.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
        except Exception as exc:
            return False, f"Could not read the CSV file: {exc}", None
    except Exception as exc:
        return False, f"Could not read the CSV file: {exc}", None

    df = normalize_csv_format(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, (
            f"Missing required column(s): {', '.join(missing)}. "
            "Accepted formats: Inventra native (transaction_date, product_name, "
            "quantity, final_amount, unit_price) or UCI Online Retail "
            "(InvoiceNo, Description, Quantity, InvoiceDate, UnitPrice)."
        ), None

    if len(df) < min_rows:
        return False, f"CSV must contain at least {min_rows} rows of data (found {len(df)}).", None

    try:
        pd.to_datetime(df['transaction_date'], format='mixed', errors='raise')
    except Exception:
        return False, "Column 'transaction_date' contains values that could not be parsed as dates.", None

    return True, "CSV is valid.", df
```

- [ ] **Step 3: Verify Django check still passes**

```
cd "c:\Users\JOVAN\OneDrive\Dokumen\Final Year Project Backup"
.\venv\Scripts\python.exe manage.py check
```

Expected output: `System check identified no issues (0 silenced).`

- [ ] **Step 4: Commit**

```
git add dashboard/analytics.py
git commit -m "feat: add UCI Online Retail CSV auto-detection and normalisation"
```

---

## Task 2: Promotional week detection (`analytics.py`)

**Files:**
- Modify: `dashboard/analytics.py` (add after `weekly_revenue_series`, before `demand_stats`)

**What this does:** Identifies calendar weeks where total discount spending was abnormally high (mean + 1 std across all weeks). Those weeks are treated as promotional outliers. `weekly_demand_baseline()` replaces promotional-week demand values with the median of non-promotional weeks so that SES/MA train on *regular* demand, not inflated promotional demand. Both the raw and baseline series are returned so the Forecast chart can shade promotional weeks visually.

- [ ] **Step 1: Add `detect_promotional_weeks()` to `analytics.py`**

Insert after `weekly_revenue_series()` (after line 111):

```python
def detect_promotional_weeks(df_product):
    """
    Identify calendar weeks with abnormally high discounting.

    A week is flagged as promotional when its total discount_amount exceeds
    mean + 1 standard deviation across all weeks.  Returns a boolean Series
    indexed by week-end date.  Returns an all-False series when no
    discount_amount column is present or when there is zero variance.
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


def weekly_demand_baseline(df_product):
    """
    Compute a promotion-adjusted weekly demand series for use as the SES/MA
    training input.

    Promotional weeks (identified by detect_promotional_weeks) are replaced
    with the median demand of non-promotional weeks so that forecasting
    algorithms are not biased by temporary demand spikes.

    Returns:
        baseline  (pd.Series) – adjusted weekly demand
        promo_weeks (int)     – number of promotional weeks replaced
    """
    weekly = weekly_demand_series(df_product)
    if weekly.empty:
        return weekly, 0

    promo_mask = detect_promotional_weeks(df_product)
    promo_aligned = promo_mask.reindex(weekly.index, fill_value=False)

    promo_count = int(promo_aligned.sum())
    if promo_count == 0 or not (~promo_aligned).any():
        return weekly, 0

    non_promo_median = weekly[~promo_aligned].median()
    baseline = weekly.copy()
    baseline[promo_aligned] = non_promo_median
    return baseline, promo_count
```

- [ ] **Step 2: Update `analyze()` signature to accept and expose `promo_weeks_count`**

Change the `analyze()` function signature and return dict:

```python
def analyze(weekly, weekly_revenue, current_stock, unit_price, lead_time, z,
            order_quantity, promo_weeks_count=0):
    """
    Full analysis for a product given its weekly demand/revenue history and settings.
    Bundles forecasting, inventory metrics, simulation and financials for the dashboard.
    promo_weeks_count: number of promotional weeks excluded from the baseline (informational).
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
        'avg_weekly_demand':  avg,
        'std_weekly_demand':  std,
        'daily_velocity':     velocity,
        'safety_stock':       ss,
        'reorder_point':      rop,
        'sells_out_in':       sells_out_in(current_stock, velocity),
        'status':             status,
        'replenishment_qty':  replenishment_qty(rop, current_stock, ss),
        'forecast':           comparison,
        'simulation':         sim,
        'financials':         fin,
        'data_weeks':         len(weekly),
        'promo_weeks_count':  promo_weeks_count,
    }
```

- [ ] **Step 3: Verify Django check**

```
.\venv\Scripts\python.exe manage.py check
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 4: Commit**

```
git add dashboard/analytics.py
git commit -m "feat: add promotional week detection and demand baseline correction"
```

---

## Task 3: Fix simulation baseline — Naive ROP (`analytics.py`)

**Files:**
- Modify: `dashboard/analytics.py` — `run_simulation()` function

**What this does:** Replaces the indefensible hardcoded `STATIC_THRESHOLD = 10` with a *Naive ROP* baseline — the reorder point computed from average demand and lead time but with **zero safety stock**. This represents what a manager does when they reorder at the "expected" consumption point without any statistical buffer. It is the standard academic comparison for safety-stock studies and is directly citable from any operations management textbook (e.g. Chopra & Meindl, *Supply Chain Management*). The template label updates from `Static (≤10)` to `Naive ROP (no buffer)`.

- [ ] **Step 1: Update `run_simulation()` to use naive ROP instead of hardcoded threshold**

Replace the existing `run_simulation` function:

```python
def run_simulation(weekly_demand, rop, order_quantity, initial_stock, lead_time=2):
    """
    Compare two reordering policies over the historical weekly demand series.

    Policy 1 — Naive ROP (no safety buffer):
        Reorder when stock <= avg_weekly_demand * lead_time.
        Represents a manager who reorders at the expected consumption point
        with no statistical safety buffer. Academically standard baseline.

    Policy 2 — DSS Forecast-based ROP (with safety stock):
        Reorder when stock <= calculated ROP (includes safety stock).
        The system's recommended policy.
    """
    avg, _, _ = demand_stats(pd.Series(list(weekly_demand))) if len(weekly_demand) else (0, 0, 0)
    naive_rop = reorder_point(avg, lead_time, 0)  # no safety stock

    static = simulate(weekly_demand, naive_rop, order_quantity, initial_stock)
    dss    = simulate(weekly_demand, rop,       order_quantity, initial_stock)
    return {
        'static':              static,
        'dss':                 dss,
        'naive_rop':           round(naive_rop, 1),
        'stockouts_eliminated': max(0, static['stockouts'] - dss['stockouts']),
    }
```

- [ ] **Step 2: Remove the now-unused `STATIC_THRESHOLD` constant**

Delete this line from the top of `analytics.py`:
```python
STATIC_THRESHOLD = 10  # static reorder policy threshold (units)
```

- [ ] **Step 3: Verify Django check**

```
.\venv\Scripts\python.exe manage.py check
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 4: Commit**

```
git add dashboard/analytics.py
git commit -m "fix: replace hardcoded static threshold with academically defensible naive ROP baseline"
```

---

## Task 4: Wire new analytics into `services.py`

**Files:**
- Modify: `dashboard/services.py` — `process_csv_for_user()`, `load_clean_dataframe()`, `analyze_product()`

**What this does:** Routes uploaded CSVs through `normalize_csv_format()` before cleaning so both formats work transparently. Switches CSV-backed products to use `weekly_demand_baseline()` (promotional-week corrected) for forecasting, while still passing the raw series to the simulation (simulation needs real historical demand, not the smoothed baseline). Threads `lead_time` into `run_simulation()` via `analyze()`.

- [ ] **Step 1: Update `process_csv_for_user()` to normalise before cleaning**

Replace the first two lines of `process_csv_for_user()`:

```python
    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    try:
        pass  # encoding already handled above
    except Exception:
        df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
    df = analytics.normalize_csv_format(df)
    df = analytics.clean_dataframe(df)
```

Full replacement for the first block of `process_csv_for_user`:

```python
@transaction.atomic
def process_csv_for_user(user, file_path, original_filename):
    """
    Replace the user's product set with products derived from the CSV.

    Wipes existing products (and their related transactions/orders via cascade),
    builds one Product per distinct product_name with computed demand statistics,
    and records/updates the UserCSV row. Returns the number of products created.
    Accepts both Inventra native format and UCI Online Retail format.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)

    df = analytics.normalize_csv_format(df)
    df = analytics.clean_dataframe(df)

    # Wipe prior data for a clean rebuild (FK cascades remove txns & POs).
    Product.objects.filter(user=user).delete()

    product_names = sorted(df['product_name'].dropna().unique())
    created = 0
    for name in product_names:
        stats = analytics.per_product_stats(df, name)
        df_product = df[df['product_name'] == name]
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
            'file_path':         file_path,
            'is_processed':      True,
            'product_count':     created,
        },
    )
    return created
```

- [ ] **Step 2: Update `load_clean_dataframe()` to normalise on load**

Replace the existing function:

```python
def load_clean_dataframe(user):
    """Read, normalise and clean the user's stored CSV. Returns a dataframe or None."""
    path = user_csv_path(user)
    if not os.path.exists(path):
        return None
    try:
        try:
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        df = analytics.normalize_csv_format(df)
        return analytics.clean_dataframe(df)
    except Exception:  # noqa: BLE001
        return None
```

- [ ] **Step 3: Update `analyze_product()` to use baseline series and pass lead_time**

Replace the CSV-history branch inside `analyze_product()`:

```python
    if df is not None and product.has_csv_history:
        df_product = df[df['product_name'] == product.name]
        # Use promotion-adjusted baseline for forecasting; raw series for simulation
        # (simulation must reflect real historical demand including promo weeks).
        weekly, promo_count = analytics.weekly_demand_baseline(df_product)
        weekly_revenue      = analytics.weekly_revenue_series(df_product)
    else:
        weekly      = pd.Series([product.avg_weekly_demand] * 8) if product.avg_weekly_demand else pd.Series(dtype='float64')
        weekly_revenue = pd.Series([product.avg_weekly_demand * product.unit_price] * 8) if product.avg_weekly_demand else pd.Series(dtype='float64')
        promo_count = 0

    result = analytics.analyze(
        weekly=weekly,
        weekly_revenue=weekly_revenue,
        current_stock=product.current_stock,
        unit_price=product.unit_price,
        lead_time=lead_time,
        z=z,
        order_quantity=order_quantity,
        promo_weeks_count=promo_count,
    )
```

- [ ] **Step 4: Verify Django check**

```
.\venv\Scripts\python.exe manage.py check
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 5: Commit**

```
git add dashboard/services.py
git commit -m "feat: wire UCI normalisation, promo baseline, and lead_time into service layer"
```

---

## Task 5: Update Forecast tab UI (`index.html`)

**Files:**
- Modify: `dashboard/templates/dashboard/index.html` — Forecast section (around line 145–182)

**What this does:** Adds a small informational badge in the Forecast tab showing how many promotional weeks were detected and excluded from the demand baseline. This makes the analytical correction visible to the user and examiners — demonstrating the system is aware of demand distortion from promotions.

- [ ] **Step 1: Add promo-weeks badge to the forecast chart card header**

Find this block in the Forecast section:

```html
    <div class="mb-3 flex items-center justify-between">
      <h3 class="font-semibold text-gray-900">SES vs Moving Average Forecast</h3>
      <span class="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
        Best α = {{ a.forecast.best_alpha }}{% include "dashboard/_tip.html" with text="The smoothing factor tested (0.2, 0.5, 0.8) that produced the lowest MAE on your historical data. Higher α makes the forecast react faster to recent changes." %}
      </span>
    </div>
```

Replace with:

```html
    <div class="mb-3 flex items-center justify-between flex-wrap gap-2">
      <h3 class="font-semibold text-gray-900">SES vs Moving Average Forecast</h3>
      <div class="flex items-center gap-2">
        {% if a.promo_weeks_count > 0 %}
        <span class="inline-flex items-center gap-1 rounded-full bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-700">
          <i data-lucide="tag" class="h-3 w-3"></i>
          {{ a.promo_weeks_count }} promotional week{{ a.promo_weeks_count|pluralize }} excluded from baseline
          {% include "dashboard/_tip.html" with text="Weeks with abnormally high discounting were detected and replaced with the median non-promotional demand before training the forecast. This prevents promotional spikes from inflating your baseline demand estimate." %}
        </span>
        {% endif %}
        <span class="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
          Best α = {{ a.forecast.best_alpha }}{% include "dashboard/_tip.html" with text="The smoothing factor tested (0.2, 0.5, 0.8) that produced the lowest MAE on your historical data. Higher α makes the forecast react faster to recent changes." %}
        </span>
      </div>
    </div>
```

- [ ] **Step 2: Commit**

```
git add dashboard/templates/dashboard/index.html
git commit -m "feat: show promotional weeks excluded badge on Forecast tab"
```

---

## Task 6: Update Simulation tab label (`index.html`)

**Files:**
- Modify: `dashboard/templates/dashboard/index.html` — Simulation section (around line 200–207)

**What this does:** Updates the simulation table to accurately describe the baseline policy as "Naive ROP (no buffer)" with the computed naive ROP value, replacing the now-incorrect "Static (≤10)" label. Also exposes the `naive_rop` value from the analysis result.

- [ ] **Step 1: Replace the Static row in the simulation results table**

Find:

```html
          <tr>
            <td class="py-2 font-medium">Static (≤10){% include "dashboard/_tip.html" with text="Naive policy: reorder whenever stock drops to 10 units or below, regardless of actual demand patterns or lead time." %}</td>
            <td>{{ a.simulation.static.stockouts }}</td><td>{{ a.simulation.static.service_level|floatformat:1 }}%</td>
          </tr>
```

Replace with:

```html
          <tr>
            <td class="py-2 font-medium">
              Naive ROP (no buffer)
              {% include "dashboard/_tip.html" with text="Baseline policy: reorder when stock hits the expected consumption point (Avg Demand × Lead Time) with zero safety stock. Represents intuition-based ordering with no statistical buffer — the standard academic comparison for safety-stock studies." %}
              <span class="ml-1 text-xs text-gray-400">ROP = {{ a.simulation.naive_rop }}</span>
            </td>
            <td>{{ a.simulation.static.stockouts }}</td><td>{{ a.simulation.static.service_level|floatformat:1 }}%</td>
          </tr>
```

- [ ] **Step 2: Update the conclusion text below the table** (find the plain-English card)

Find any text referencing "static threshold" or "≤10" in the Simulation section and update it to say "naive ROP baseline" instead.

Search for: `stockouts_eliminated` or `static` in the simulation conclusion block and replace the label accordingly. Exact text will depend on your template — update the human-readable summary to say:

> "DSS Forecast-based ROP eliminated X stockouts compared to the naive ROP baseline (no safety buffer), improving your service level from Y% to Z%."

- [ ] **Step 3: Commit**

```
git add dashboard/templates/dashboard/index.html
git commit -m "fix: update simulation baseline label to Naive ROP with dynamic ROP value"
```

---

## Task 7: Update Settings upload instructions (`settings.html`)

**Files:**
- Modify: `dashboard/templates/dashboard/settings.html` — CSV upload section

**What this does:** Tells users the system now accepts two formats, so they know they can upload the UCI Online Retail dataset directly without reformatting.

- [ ] **Step 1: Find the required columns list in settings.html and update it**

Find the section that lists required columns (something like "Your CSV must contain these columns"). Update it to show both accepted formats:

```html
<div class="rounded-lg bg-gray-50 p-4 text-sm text-gray-600 space-y-3">
  <p class="font-semibold text-gray-800">Accepted CSV formats:</p>

  <div>
    <p class="font-medium text-gray-700 mb-1">Format A — Inventra Native</p>
    <div class="flex flex-wrap gap-2">
      {% for col in "transaction_date,product_name,quantity,final_amount,unit_price"|split:"," %}
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">{{ col }}</code>
      {% endfor %}
    </div>
  </div>

  <div>
    <p class="font-medium text-gray-700 mb-1">Format B — UCI Online Retail <span class="text-xs text-gray-400">(auto-detected)</span></p>
    <div class="flex flex-wrap gap-2">
      {% for col in "InvoiceNo,Description,Quantity,InvoiceDate,UnitPrice"|split:"," %}
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">{{ col }}</code>
      {% endfor %}
    </div>
    <p class="mt-1 text-xs text-gray-400">Download from: UCI Machine Learning Repository — Online Retail dataset (541,000 real UK retail transactions, widely cited in academic literature).</p>
  </div>
</div>
```

Note: Django templates don't have a built-in `split` filter. Use a static unrolled list instead if `split` is not registered as a custom filter. Equivalent static HTML:

```html
<div class="rounded-lg bg-gray-50 p-4 text-sm text-gray-600 space-y-3">
  <p class="font-semibold text-gray-800">Two CSV formats are accepted:</p>
  <div>
    <p class="font-medium text-gray-700 mb-1">Format A — Inventra Native</p>
    <div class="flex flex-wrap gap-2">
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">transaction_date</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">product_name</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">quantity</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">final_amount</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">unit_price</code>
    </div>
  </div>
  <div>
    <p class="font-medium text-gray-700 mb-1">Format B — UCI Online Retail <span class="text-xs text-gray-400">(auto-detected)</span></p>
    <div class="flex flex-wrap gap-2">
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">InvoiceNo</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">Description</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">Quantity</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">InvoiceDate</code>
      <code class="rounded bg-white px-2 py-0.5 text-xs ring-1 ring-gray-200">UnitPrice</code>
    </div>
    <p class="mt-1 text-xs text-gray-400">Source: UCI Machine Learning Repository — Online Retail dataset (541,000 real UK retail transactions, widely cited in academic research).</p>
  </div>
</div>
```

- [ ] **Step 2: Final Django check**

```
.\venv\Scripts\python.exe manage.py check
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 3: Final commit**

```
git add dashboard/templates/dashboard/settings.html
git commit -m "feat: update settings page to document both accepted CSV formats"
```

---

## Self-Review

**Spec coverage:**
- [x] UCI Online Retail auto-detection → Task 1
- [x] Promotional week detection using `discount_amount` → Task 2
- [x] `analyze()` exposes `promo_weeks_count` → Task 2, Step 2
- [x] Simulation baseline replaced with Naive ROP → Task 3
- [x] `services.py` wired to all new helpers → Task 4
- [x] Forecast tab UI shows promo badge → Task 5
- [x] Simulation tab label updated → Task 6
- [x] Settings page documents both formats → Task 7

**Placeholder scan:** None found. All code blocks are complete and runnable.

**Type consistency:**
- `weekly_demand_baseline()` returns `(pd.Series, int)` — consumed in Task 4 as `weekly, promo_count` ✓
- `run_simulation()` returns `naive_rop` key — consumed in Task 6 template as `a.simulation.naive_rop` ✓
- `analyze()` accepts `promo_weeks_count` kwarg — passed from `analyze_product()` in Task 4 ✓
- `detect_promotional_weeks()` returns `pd.Series[bool]` — consumed internally by `weekly_demand_baseline()` only ✓
