# Inventra — Full Rebuild Prompt for Claude Code

## 🎯 Main Goal
Build **Inventra** — a web-based Decision Support System for inventory and sales management targeting SMEs. Built with Django + HTML + CSS + Tailwind CSS + SQLite. Build everything from scratch, clean and properly structured.

---

## 🎨 Design & Theme
- Clean, modern, professional dashboard inspired by marketing intelligence dashboards
- Light theme with white/light gray background, dark sidebar navigation
- Sidebar is fixed on the left, content area scrolls independently
- Cards with subtle shadows and rounded corners
- Color coding: green for safe/positive, orange for warning, red for critical
- Charts are clean and minimal using Plotly
- Typography: clean sans-serif, numbers bold and prominent
- Use Tailwind CSS utility classes throughout
- Sidebar navigation switches content WITHOUT full page reload using JavaScript tab switching
- Active sidebar item highlighted

---

## 🗂️ Project Structure
```
inventra/
├── inventra/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── dashboard/
│   ├── management/
│   │   └── commands/
│   ├── templates/
│   │   └── dashboard/
│   │       ├── base.html
│   │       ├── login.html
│   │       ├── signup.html
│   │       ├── index.html
│   │       ├── products.html
│   │       ├── product_detail.html
│   │       ├── admin_panel.html
│   │       ├── settings.html
│   │       └── whatif.html
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── forms.py
├── media/
│   └── user_csvs/
├── static/
└── manage.py
```

---

## 🔐 Authentication

**Login page `/login/`:**
- Email and password fields
- "Remember me" checkbox
- Link to signup
- Clean centered card layout
- Redirect to `/` after login

**Signup page `/signup/`:**
- Username, email, password, confirm password
- Link to login
- After signup → redirect to `/settings/` with message "Welcome! Upload your CSV to get started"

**Rules:**
- All dashboard pages require `@login_required`
- New users with no CSV uploaded → always redirected to `/settings/` upload page
- Logout at `/logout/` → redirect to `/login/`

---

## 🗄️ Models

```python
from django.contrib.auth.models import User

class UserCSV(models.Model):
    user = OneToOneField(User, on_delete=CASCADE)
    original_filename = CharField(max_length=255)
    file_path = CharField(max_length=500)
    uploaded_at = DateTimeField(auto_now=True)
    is_processed = BooleanField(default=False)
    product_count = IntegerField(default=0)

class Product(models.Model):
    user = ForeignKey(User, on_delete=CASCADE)
    name = CharField(max_length=200)
    current_stock = FloatField(default=25)
    unit_price = FloatField(default=10.0)
    lead_time = IntegerField(default=2)
    supplier_name = CharField(max_length=200, blank=True)
    supplier_contact = CharField(max_length=200, blank=True)
    avg_weekly_demand = FloatField(default=0)
    std_weekly_demand = FloatField(default=0)
    daily_velocity = FloatField(default=0)
    has_csv_history = BooleanField(default=False)
    created_at = DateTimeField(auto_now_add=True)

class InventorySettings(models.Model):
    user = ForeignKey(User, on_delete=CASCADE)
    selected_product = ForeignKey(Product, null=True, blank=True, on_delete=SET_NULL)
    service_level_z = FloatField(default=1.65)
    lead_time = IntegerField(default=2)
    order_quantity = IntegerField(default=20)

class PurchaseOrder(models.Model):
    user = ForeignKey(User, on_delete=CASCADE)
    product = ForeignKey(Product, on_delete=CASCADE)
    quantity = FloatField()
    status = CharField(choices=[
        ('Suggested', 'Suggested'),
        ('Ordered', 'Ordered'),
        ('Received', 'Received')
    ], default='Suggested')
    created_at = DateTimeField(auto_now_add=True)
    notes = TextField(blank=True)

class StockTransaction(models.Model):
    user = ForeignKey(User, on_delete=CASCADE)
    product = ForeignKey(Product, on_delete=CASCADE)
    transaction_type = CharField(choices=[('SALE', 'Sale'), ('RESTOCK', 'Restock')])
    quantity = FloatField()
    stock_after = FloatField()
    notes = TextField(blank=True)
    date = DateTimeField(auto_now_add=True)
```

---

## 📊 Core Formulas (Most Important — Implement Exactly)

### CSV Cleaning (apply on upload)
```python
df = pd.read_csv(file_path)
df = df.dropna(subset=['store_name'])               # drop missing store
df = df[df['final_amount'] > 0]                     # remove negative amounts
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df = df.sort_values('transaction_date')

# Per product:
df_product = df[df['product_name'] == product_name]
weekly = df_product.set_index('transaction_date')['quantity'].resample('W').sum()
weekly = weekly.replace(0, None).ffill()            # fill zero weeks
```

### Demand Statistics (per product)
```python
avg_weekly_demand = weekly.mean()
std_weekly_demand = weekly.std()
daily_velocity = avg_weekly_demand / 7
```

### SES Forecast (Single Exponential Smoothing)
```python
# Test alpha values 0.2, 0.5, 0.8 — pick lowest MAE
def ses_forecast(demand, alpha):
    forecasts = [demand[0]]
    for t in range(1, len(demand)):
        f = alpha * demand[t-1] + (1 - alpha) * forecasts[t-1]
        forecasts.append(f)
    return forecasts

# MAE
mae = np.mean(np.abs(np.array(actual) - np.array(forecast)))

# Best alpha = alpha with lowest MAE
best_alpha = min([0.2, 0.5, 0.8], key=lambda a: compute_mae(ses_forecast(demand, a), demand))
```

### Moving Average Forecast (for comparison)
```python
# 4-week moving average
def ma_forecast(demand, window=4):
    forecasts = []
    for t in range(len(demand)):
        if t < window:
            forecasts.append(np.mean(demand[:t+1]))
        else:
            forecasts.append(np.mean(demand[t-window:t]))
    return forecasts
```

### Algorithm Comparison
```python
mae_ses  = np.mean(np.abs(np.array(actual) - np.array(ses_forecasts)))
mae_ma   = np.mean(np.abs(np.array(actual) - np.array(ma_forecasts)))
rmse_ses = np.sqrt(np.mean((np.array(actual) - np.array(ses_forecasts))**2))
rmse_ma  = np.sqrt(np.mean((np.array(actual) - np.array(ma_forecasts))**2))
best_method = 'SES' if mae_ses <= mae_ma else 'MA-4'
```

### Safety Stock
```python
z_map = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}
safety_stock = Z * std_weekly_demand * np.sqrt(lead_time)
```

### Reorder Point (ROP)
```python
rop = avg_weekly_demand * lead_time + safety_stock
```

### Sells Out In
```python
sells_out_in_days = round(current_stock / daily_velocity) if daily_velocity > 0 else 0
```

### Stock Status
```python
if current_stock > rop:
    status = 'Safe'        # green
elif current_stock > safety_stock:
    status = 'Order Soon'  # orange
else:
    status = 'Critical'    # red
```

### Replenishment Quantity
```python
order_qty = max(0, rop - current_stock + safety_stock)
```

### Financial Calculations
```python
inventory_value = current_stock * unit_price

weekly_revenue = df_product.groupby(
    pd.Grouper(key='transaction_date', freq='W'))['final_amount'].sum()
avg_weekly_revenue  = round(weekly_revenue.mean(), 2)
max_revenue         = round(weekly_revenue.max(), 2)
min_revenue         = round(weekly_revenue.min(), 2)
max_revenue_date    = weekly_revenue.idxmax().strftime('%b %d %Y')
min_revenue_date    = weekly_revenue.idxmin().strftime('%b %d %Y')

holding_cost_weekly = round(safety_stock * unit_price * 0.25 / 52, 2)
holding_cost_annual = round(safety_stock * unit_price * 0.25, 2)

stockout_cost_per_occurrence = avg_weekly_revenue
stockout_cost_static         = stockouts_static * avg_weekly_revenue
stockout_cost_dss            = stockouts_forecast * avg_weekly_revenue
dss_savings                  = stockout_cost_static - stockout_cost_dss
```

### Simulation
```python
# Run two policies over all historical weeks
# Policy 1: Static threshold (reorder when stock <= 10)
# Policy 2: Forecast-based ROP (reorder when stock <= calculated ROP)

def simulate(demand, threshold, order_quantity, initial_stock=30):
    stock = initial_stock
    stockouts = 0
    levels = []
    for t in range(len(demand)):
        stock -= demand[t]
        if stock < 0:
            stockouts += 1
            stock = 0
        levels.append(stock)
        if stock <= threshold:
            stock += order_quantity
    service_level = round((len(demand) - stockouts) / len(demand) * 100, 1)
    return stockouts, service_level, levels

stockouts_static,   service_static,   levels_static   = simulate(demand, 10,  order_qty)
stockouts_forecast, service_forecast, levels_forecast = simulate(demand, rop, order_qty)
```

### What-If Tool (JavaScript — no page reload)
```javascript
const Z_MAP = {80:0.84,81:0.88,82:0.92,83:0.95,84:0.99,85:1.04,
               86:1.08,87:1.13,88:1.17,89:1.23,90:1.28,91:1.34,
               92:1.41,93:1.48,94:1.56,95:1.65,96:1.75,97:1.88,
               98:2.05,99:2.33};

function calculate(stock, leadTime, serviceLevel, avgDemand, stdDemand) {
    const z             = Z_MAP[serviceLevel] || 1.65;
    const dailyVelocity = avgDemand / 7;
    const safetStock    = z * stdDemand * Math.sqrt(leadTime);
    const rop           = avgDemand * leadTime + safetyStock;
    const orderQty      = Math.max(0, rop - stock);
    const sellsOutIn    = dailyVelocity > 0 ? Math.round(stock / dailyVelocity) : 0;
    return { safetyStock, rop, orderQty, sellsOutIn };
}
```

---

## 🖥️ Pages & Features

### Base Template (`base.html`)
- Fixed left sidebar (dark background, white text)
- Sidebar items with icons: Overview, Products, What-If, Orders, Financial, Admin, Settings, Logout
- Tab switching uses JavaScript — content divs show/hide, NO full page reload
- Active sidebar item highlighted with accent color
- Top bar: page title left, username + logout right
- Main content area: scrollable, padding, light background
- Tailwind CSS via CDN
- Plotly.js via CDN
- Chart.js via CDN

### Login & Signup (`login.html`, `signup.html`)
- No sidebar — standalone centered card layout
- Clean minimal design matching overall theme
- Form validation with inline error messages
- Redirect to dashboard after login, settings after signup

### Main Dashboard `/` — Tab-Based (`index.html`)
All tabs switch via JavaScript, no page reload:

**Tab 1 — Overview**
- Products summary bar: Total / Critical / Order Soon / Safe
- Product selector dropdown (switches active product, reloads page with ?product_id=)
- KPI cards (6 cards): Current Stock, Avg Weekly Demand, ROP, Safety Stock, Sells Out In, Daily Velocity
- Stock alert banner: green/orange/red with plain English message
- Quick Sale / Quick Restock inline form → POST → updates stock + creates StockTransaction
- PO suggestion card (appears only when stock ≤ ROP): shows product, order qty, supplier info, Place Order button
- Recent activity list: last 5 StockTransactions for selected product

**Tab 2 — Forecast**
- SES vs MA forecast Plotly line chart (actual demand = blue, SES = orange dashed, MA = green dashed)
- Best alpha shown
- Algorithm comparison table: MAE and RMSE for SES and MA-4, best method highlighted
- Forecast Stock chart: projected stock depletion over next 8 weeks with LT zone and DoS zone shaded

**Tab 3 — Simulation**
- Inventory level comparison Plotly chart: static threshold (red) vs DSS ROP (green)
- Horizontal threshold lines
- Results comparison table: Stockouts and Service Level for both policies
- Plain English conclusion card: "DSS eliminated X stockouts, saving RM Y in potential lost sales"

**Tab 4 — My Orders**
- Summary counts: Suggested / Ordered / Received this month
- Orders table: Date, Product, Qty, Status badge, Actions
- Confirm Order button (Suggested → Ordered)
- Mark Received button (Ordered → Received, auto-adds stock, creates RESTOCK transaction)
- Cancel button (Suggested only → deletes PO)

**Tab 5 — Financial**
- 4 KPI cards: Inventory Value, Avg Weekly Revenue, Annual Holding Cost, DSS Savings (green hero card)
- Weekly revenue trend Plotly line chart with average line
- Financial summary table with all metrics and RM currency
- Plain English explanation under each metric for SME users

### Products Overview `/products/`
- Summary bar: Total / Critical / Order Soon / Safe counts
- 3 Plotly charts side by side:
  - Donut: stock health distribution (Safe / Order Soon / Critical)
  - Bar: all products current stock vs ROP, color coded by status
  - Horizontal bar: top 5 products closest to stockout (lowest sells_out_in)
- Full products table: Product, Stock, ROP, Safety Stock, Sells Out In, Status, Actions
- Actions per row: View, Sale (inline qty form), Restock (inline qty form), Place Order

### Individual Product `/products/<id>/`
- Full analysis for that specific product
- All KPI cards
- Stock alert banner
- SES + MA forecast chart
- MAE/RMSE comparison table
- ROP and Safety Stock calculation breakdown
- Quick Sale/Restock form
- PO suggestion card if stock ≤ ROP
- Last 5 transactions for this product only
- Back button to /products/

### What-If Tool `/whatif/`
- Three sliders: Current Stock (1–40), Lead Time (1–6 weeks), Service Level (80–99%)
- All calculations in pure JavaScript — instant update, no server call
- Result cards: Safety Stock, ROP, Order Quantity, Sells Out In
- Color coded alert: green / orange / red
- Chart.js bar chart: Current Stock vs Safety Stock vs ROP
- avgDemand and stdDemand injected from Django context into JS variables

### Admin Panel `/admin-panel/`
Four sections, all on one page:

**Section A — Product Management**
- Table: Name, Stock, Price, Lead Time, Supplier, CSV/Manual badge, Actions
- Add Product form: name, stock, price, lead time, supplier name, supplier contact, avg demand (for manual products)
- Edit inline per row
- Delete with confirmation

**Section B — Record Transaction**
- Form: Product dropdown, SALE or RESTOCK, Quantity, Notes
- Submit → updates Product.current_stock, creates StockTransaction with stock_after
- Success message on submit

**Section C — Purchase Order Management**
- Table: Product, Qty, Status, Date, Actions
- Mark as Ordered, Mark as Received (auto-updates stock + creates RESTOCK transaction), Delete

**Section D — Transaction History**
- Last 20 transactions
- Color coded rows: red for SALE, green for RESTOCK
- Columns: Date, Product, Type, Qty, Notes, Stock After
- Clear History button with JS confirmation

### Settings `/settings/`

**Section A — CSV Upload**
- If no CSV uploaded: prompt with instructions and required columns listed
- If CSV exists: show filename, upload date, product count, Replace CSV option
- On upload:
  1. Validate file is .csv
  2. Validate required columns exist: transaction_date, product_name, quantity, final_amount, unit_price
  3. Validate at least 10 rows
  4. Show warning: "This will delete all your current products and transaction history"
  5. If confirmed: delete all user's Products, Transactions, PurchaseOrders, UserCSV
  6. Clean and process CSV (apply all cleaning steps)
  7. For each unique product: calculate demand stats, create Product tagged to user
  8. Save UserCSV record with file_path = media/user_csvs/<user_id>/data.csv
  9. Redirect to /products/ with success message

**Section B — Inventory Settings**
- Lead time: dropdown 1–6 weeks
- Service level: dropdown 90% / 95% / 97.5% / 99%
- Order quantity: number input
- Save → persists to InventorySettings model for this user

---

## 🔒 Per-User Data Isolation
- Every model has `user = ForeignKey(User, on_delete=CASCADE)`
- Every query in views.py filters: `Model.objects.filter(user=request.user)`
- New user with no UserCSV → redirect to /settings/ before accessing any dashboard
- Each user's CSV stored at `media/user_csvs/<user.id>/data.csv`
- Users cannot access other users' data under any circumstance

---

## 🎬 Complete Demo Flow
1. Register new user → redirected to Settings → upload grocery_chain_data.csv
2. System processes CSV → 18 products loaded automatically
3. /products/ shows all 18 with status badges and 3 charts
4. Main dashboard shows Overview tab with Chicken Breast selected by default
5. Switch to Forecast tab → SES vs MA chart with best alpha and MAE table
6. Switch to Simulation tab → comparison chart and service level results
7. Record sale of 10 units → stock drops → alert changes to orange
8. Record another sale → stock critical → PO suggestion card appears
9. Click Place Order → PO saved to database
10. Go to My Orders tab → see order as Suggested → Confirm → Mark Received → stock restores
11. Financial tab → shows DSS savings highlighted in green
12. What-If tab → move sliders → instant JS recalculation
13. Log out → log back in → all data intact
14. Register second user → upload different CSV → completely separate data

---

## ⚙️ Technical Requirements
- Django 4.x
- Tailwind CSS via CDN (no build step needed)
- Plotly.js via CDN for all server-rendered charts
- Chart.js via CDN for What-If client-side chart
- SQLite database
- `MEDIA_ROOT = BASE_DIR / 'media'`
- `MEDIA_URL = '/media/'`
- `LOGIN_URL = '/login/'`
- `LOGIN_REDIRECT_URL = '/'`
- All views use `@login_required` decorator
- No hardcoded file paths — always use `os.path.join`
- Graceful error handling — no crashes if CSV missing or product not found
- `requirements.txt` with pinned versions

---

## 📦 Deliverables
Provide complete ready-to-run code for every file:
- `inventra/settings.py`
- `inventra/urls.py`
- `dashboard/models.py`
- `dashboard/views.py`
- `dashboard/urls.py`
- `dashboard/forms.py`
- `dashboard/admin.py`
- `templates/dashboard/base.html`
- `templates/dashboard/login.html`
- `templates/dashboard/signup.html`
- `templates/dashboard/index.html`
- `templates/dashboard/products.html`
- `templates/dashboard/product_detail.html`
- `templates/dashboard/admin_panel.html`
- `templates/dashboard/settings.html`
- `templates/dashboard/whatif.html`
- `requirements.txt`
- Step-by-step setup instructions (pip install, migrate, runserver)