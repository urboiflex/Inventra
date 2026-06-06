from django.urls import path

from . import views

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),

    # Dashboard
    path('', views.index, name='index'),
    path('products/', views.products_view, name='products'),
    path('products/<int:pk>/', views.product_detail, name='product_detail'),
    path('whatif/', views.whatif_view, name='whatif'),
    path('settings/', views.settings_view, name='settings'),
    path('admin-panel/', views.admin_panel, name='admin_panel'),

    # Transactions & orders
    path('transaction/record/', views.record_transaction, name='record_transaction'),
    path('orders/create/', views.create_order, name='create_order'),
    path('orders/<int:pk>/action/', views.order_action, name='order_action'),

    # Product management
    path('products/add/', views.product_create, name='product_create'),
    path('products/<int:pk>/edit/', views.product_edit, name='product_edit'),
    path('products/<int:pk>/delete/', views.product_delete, name='product_delete'),
    path('history/clear/', views.clear_history, name='clear_history'),
]
