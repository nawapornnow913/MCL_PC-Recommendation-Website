from django import forms

USAGE_CHOICES = [
    ('gaming', '🎮 คอมเล่นเกม'),
    ('creator', '🎨 สายกราฟฟิก / ตัดต่อวิดีโอ'),
    ('work', '💼 ทำงานทั่วไป / ออฟฟิศ'),
]

CPU_BRAND_CHOICES = [
    ('all', ' all'),
    ('intel', '🔵 Intel'),
    ('amd', '🔴 AMD')
]

GPU_BRAND_CHOICES = [
    ('all', 'all'),
    ('nvidia', '🟢 NVIDIA'),
    ('amd', '🔴 AMD ')
]

class SpecRequestForm(forms.Form):
    usage_type = forms.ChoiceField(
        label='จุดประสงค์การใช้งานหลัก',
        choices=USAGE_CHOICES,
        initial='gaming',
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    budget = forms.IntegerField(
        label='งบประมาณ (บาท)', 
        min_value=5000,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'เช่น 35000', 'id': 'budget-input'})
    )
    
    # เพิ่มฟิลด์เลือกค่าย 
    cpu_brand = forms.ChoiceField(
        label='ค่าย CPU',
        choices=CPU_BRAND_CHOICES,
        initial='all',
        widget=forms.RadioSelect(attrs={'class': 'form-check-input brand-radio'})
    )
    gpu_brand = forms.ChoiceField(
        label='ค่าย การ์ดจอ',
        choices=GPU_BRAND_CHOICES,
        initial='all',
        widget=forms.RadioSelect(attrs={'class': 'form-check-input brand-radio'})
    )