from django.contrib import admin

from .models import (
    InventorySettings, Product, PurchaseOrder, StockTransaction, UserCSV,
)


@admin.register(UserCSV)
class UserCSVAdmin(admin.ModelAdmin):
    list_display = ('user', 'original_filename', 'product_count', 'is_processed', 'uploaded_at')
    list_filter = ('is_processed',)
    search_fields = ('user__username', 'original_filename')


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'current_stock', 'unit_price', 'avg_weekly_demand', 'has_csv_history')
    list_filter = ('has_csv_history', 'user')
    search_fields = ('name', 'user__username')


@admin.register(InventorySettings)
class InventorySettingsAdmin(admin.ModelAdmin):
    list_display = ('user', 'lead_time', 'service_level_z', 'order_quantity', 'selected_product')


@admin.register(PurchaseOrder)
class PurchaseOrderAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'product', 'quantity', 'status', 'created_at')
    list_filter = ('status',)
    search_fields = ('product__name', 'user__username')


@admin.register(StockTransaction)
class StockTransactionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'product', 'transaction_type', 'quantity', 'stock_after', 'date')
    list_filter = ('transaction_type',)
    search_fields = ('product__name', 'user__username')
