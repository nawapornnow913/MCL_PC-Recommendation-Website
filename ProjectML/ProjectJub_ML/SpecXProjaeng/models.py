from django.db import models

class CPU(models.Model):
    name = models.CharField(max_length=255)
    benchmark_score = models.IntegerField()
    price = models.IntegerField()
    socket = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.name

class GPU(models.Model):
    name = models.CharField(max_length=255)
    benchmark_score = models.IntegerField()
    price = models.IntegerField()

    def __str__(self):
        return self.name

class Component(models.Model):
    CATEGORY_CHOICES = [
        ('mb_intel', 'Mainboard Intel'), ('mb_amd', 'Mainboard AMD'),
        ('ram', 'RAM'), ('ssd', 'Storage'), ('psu', 'Power Supply'),
        ('case', 'Case'), ('monitor', 'Monitor'), ('mouse', 'Mouse'),
        ('keyboard', 'Keyboard'), ('headset', 'Headset'), ('mic', 'Microphone'),
    ]
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    name = models.CharField(max_length=255)
    price = models.IntegerField()
    spec_text = models.TextField(null=True, blank=True)
    socket = models.CharField(max_length=50, null=True, blank=True)
    ram_type = models.CharField(max_length=20, null=True, blank=True)
    watt = models.IntegerField(default=0)

    def __str__(self):
        return f"[{self.category}] {self.name}"
    

