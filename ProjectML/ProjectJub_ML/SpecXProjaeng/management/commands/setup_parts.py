from django.core.management.base import BaseCommand
from SpecXProjaeng.models import Component

class Command(BaseCommand):
    help = 'Generate PC Parts'

    def handle(self, *args, **kwargs):
        Component.objects.all().delete()
        
        parts = [
            # RAM
            ('ram', 'Blackberry 8GB DDR4', 790, '2666MHz'),
            ('ram', 'Kingston Fury 16GB (8x2)', 1590, '3200MHz'),
            ('ram', 'Corsair Vengeance 32GB', 3200, 'DDR5 5200MHz'),
            
            # SSD
            ('ssd', 'Hikvision 256GB', 590, 'SATA III'),
            ('ssd', 'WD Blue SN570 500GB', 1490, 'M.2 NVMe'),
            ('ssd', 'Samsung 980 1TB', 2690, 'M.2 NVMe'),
            
            # PSU
            ('psu', 'DTECH 600W', 450, 'Standard'),
            ('psu', 'SilverStone 500W', 1190, '80+ White'),
            ('psu', 'Thermaltake 650W', 1990, '80+ Bronze'),
            
            # Case
            ('case', 'Office Case', 390, 'Black Standard'),
            ('case', 'Tsunami Galaxy', 990, 'RGB Glass'),
            ('case', 'NZXT H5 Flow', 2890, 'Premium White'),

            # MB Intel
            ('mb_intel', 'Biostar H610M', 1890, 'LGA1700'),
            ('mb_intel', 'Asus Prime B760M', 3590, 'LGA1700'),

            # MB AMD
            ('mb_amd', 'Asrock A520M', 1690, 'AM4'),
            ('mb_amd', 'MSI B550M Pro', 3290, 'AM4'),
        ]
        
        for cat, name, price, spec in parts:
            Component.objects.create(category=cat, name=name, price=price, spec_text=spec)
            
        self.stdout.write(self.style.SUCCESS('✅ สร้างอุปกรณ์เสริมเรียบร้อย!'))