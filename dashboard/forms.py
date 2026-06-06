from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import InventorySettings, Product

# Shared Tailwind input styling applied to widgets.
INPUT_CLASS = (
    "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 "
    "focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 focus:outline-none"
)


def _style(fields):
    """Apply the shared Tailwind class to a set of bound form fields."""
    for field in fields.values():
        existing = field.widget.attrs.get('class', '')
        field.widget.attrs['class'] = (existing + ' ' + INPUT_CLASS).strip()


class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _style(self.fields)

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Data file",
        help_text="Accepted formats: .csv or .xlsx (Inventra native or UCI Online Retail)",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['csv_file'].widget.attrs['class'] = (
            "block w-full text-sm text-gray-700 file:mr-4 file:rounded-lg file:border-0 "
            "file:bg-indigo-600 file:px-4 file:py-2 file:text-white hover:file:bg-indigo-700"
        )
        self.fields['csv_file'].widget.attrs['accept'] = '.csv,.xlsx'

    def clean_csv_file(self):
        f = self.cleaned_data['csv_file']
        if not f.name.lower().endswith(('.csv', '.xlsx')):
            raise forms.ValidationError("Please upload a .csv or .xlsx file.")
        return f


class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            'name', 'current_stock', 'unit_price', 'lead_time',
            'supplier_name', 'supplier_contact', 'avg_weekly_demand', 'std_weekly_demand',
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _style(self.fields)


class InventorySettingsForm(forms.ModelForm):
    SERVICE_LEVEL_CHOICES = [
        (1.28, '90%'),
        (1.65, '95%'),
        (1.96, '97.5%'),
        (2.33, '99%'),
    ]
    LEAD_TIME_CHOICES = [(i, f"{i} week{'s' if i > 1 else ''}") for i in range(1, 7)]

    service_level_z = forms.TypedChoiceField(
        choices=SERVICE_LEVEL_CHOICES, coerce=float, label="Service level"
    )
    lead_time = forms.TypedChoiceField(
        choices=LEAD_TIME_CHOICES, coerce=int, label="Lead time"
    )

    class Meta:
        model = InventorySettings
        fields = ['lead_time', 'service_level_z', 'order_quantity']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _style(self.fields)
