from django.contrib.auth.models import User
from django.db import models


class UserCSV(models.Model):
    """Tracks the single CSV a user has uploaded as their data source."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='csv')
    original_filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(auto_now=True)
    is_processed = models.BooleanField(default=False)
    product_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.username} — {self.original_filename}"


class Product(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='products')
    name = models.CharField(max_length=200)
    current_stock = models.FloatField(default=25)
    unit_price = models.FloatField(default=10.0)
    lead_time = models.IntegerField(default=2)
    supplier_name = models.CharField(max_length=200, blank=True)
    supplier_contact = models.CharField(max_length=200, blank=True)
    avg_weekly_demand = models.FloatField(default=0)
    std_weekly_demand = models.FloatField(default=0)
    daily_velocity = models.FloatField(default=0)
    prior_avg_demand = models.FloatField(default=0, help_text="Your best estimate of weekly demand. Used as a starting point that phases out automatically once 4 weeks of real sales are recorded.")
    has_csv_history = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']
        unique_together = ('user', 'name')

    def __str__(self):
        return f"{self.name} ({self.user.username})"


class InventorySettings(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='inventory_settings')
    selected_product = models.ForeignKey(
        Product, null=True, blank=True, on_delete=models.SET_NULL
    )
    service_level_z = models.FloatField(default=1.65)
    lead_time = models.IntegerField(default=2)
    order_quantity = models.IntegerField(default=20)

    def __str__(self):
        return f"Settings for {self.user.username}"


class PurchaseOrder(models.Model):
    STATUS_CHOICES = [
        ('Suggested', 'Suggested'),
        ('Ordered', 'Ordered'),
        ('Received', 'Received'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='purchase_orders')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='purchase_orders')
    quantity = models.FloatField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Suggested')
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"PO #{self.pk} — {self.product.name} x{self.quantity} ({self.status})"


class StockTransaction(models.Model):
    TYPE_CHOICES = [
        ('SALE', 'Sale'),
        ('RESTOCK', 'Restock'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transactions')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='transactions')
    transaction_type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    quantity = models.FloatField()
    stock_after = models.FloatField()
    notes = models.TextField(blank=True)
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.transaction_type} {self.quantity} {self.product.name}"
