import pandas as pd
import requests
import re
from django.core.management.base import BaseCommand
from SpecXProjaeng.models import CPU, GPU

class Command(BaseCommand):
    help = 'Scrape Real Data from PassMark & Convert to THB'

    def handle(self, *args, **kwargs):
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # --- 1. CPU ---
        self.stdout.write("⏳ กำลังดูดข้อมูล CPU...")
        url_cpu = "https://www.cpubenchmark.net/high_end_cpus.html"
        try:
            # ใช้ flavor='bs4' เพื่อแก้ปัญหา lxml error
            response = requests.get(url_cpu, headers=headers)
            tables = pd.read_html(response.text, flavor='bs4') 
            df = tables[0]
            
            CPU.objects.all().delete()
            
            count = 0
            for index, row in df.iterrows():
                try:
                    name = str(row['CPU Name'])
                    score = int(row['CPU Mark'])
                    raw_price = str(row['Price'])
                    
                    price_match = re.search(r'\d+\.?\d*', raw_price.replace(',', ''))
                    if price_match:
                        price_thb = int(float(price_match.group()) * 36 * 1.1)
                    else:
                        continue

                    # เช็คราคาและชื่อยี่ห้อ (บรรทัดนี้ต้องพิมพ์ให้ครบ)
                    if price_thb < 1500 or ("Intel" not in name and "AMD" not in name): 
                        continue

                    CPU.objects.create(name=name, benchmark_score=score, price=price_thb)
                    count += 1
                except: 
                    continue
            self.stdout.write(self.style.SUCCESS(f'✅ CPU: {count} รุ่น'))
        except Exception as e: 
            self.stdout.write(self.style.ERROR(f'❌ CPU Error: {e}'))

        # --- 2. GPU ---
        self.stdout.write("⏳ กำลังดูดข้อมูล GPU...")
        url_gpu = "https://www.videocardbenchmark.net/high_end_gpus.html"
        try:
            response = requests.get(url_gpu, headers=headers)
            tables = pd.read_html(response.text, flavor='bs4')
            df = tables[0]
            
            GPU.objects.all().delete()
            
            count = 0
            for index, row in df.iterrows():
                try:
                    name = str(row['Videocard Name'])
                    score = int(row['G3D Mark'])
                    raw_price = str(row['Price'])
                    
                    price_match = re.search(r'\d+\.?\d*', raw_price.replace(',', ''))
                    if price_match:
                        price_thb = int(float(price_match.group()) * 36 * 1.1)
                    else:
                        continue

                    # ต้องมีเงื่อนไข < 2000 และมีเครื่องหมาย : ต่อท้าย
                    if price_thb < 2000: 
                        continue

                    GPU.objects.create(name=name, benchmark_score=score, price=price_thb)
                    count += 1
                except: 
                    continue
            self.stdout.write(self.style.SUCCESS(f'✅ GPU: {count} รุ่น'))
        except Exception as e: 
            self.stdout.write(self.style.ERROR(f'❌ GPU Error: {e}'))


def scrape_mouse_to_csv():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    print("⏳ กำลังดูดข้อมูล Mouse...")
    
    # URL เป้าหมาย (สมมติว่าเป็นเว็บฐานข้อมูลสเปคเมาส์หรือร้านค้า)
    # หมายเหตุ: คุณจะต้องเปลี่ยน URL และ Class ของ HTML ให้ตรงกับเว็บจริงที่คุณต้องการดึง
    url_mouse = "https://ihavecpu.com/category/mouse" 
    
    try:
        response = requests.get(url_mouse, headers=headers)
        response.raise_for_status() # เช็คว่าเว็บตอบกลับเป็น 200 OK ไหม
        
        soup = BeautifulSoup(response.text, 'html.parser')
        mouse_list = []
        count = 0
        
        # สมมติว่าเมาส์แต่ละตัวอยู่ใน Tag <div class="product-item">
        items = soup.find_all('div', class_='product-item')
        
        for item in items:
            try:
                # 1. ดึงข้อความดิบจาก HTML (ต้องเปลี่ยน class ให้ตรงกับเว็บเป้าหมาย)
                name = item.find('h2', class_='product-title').text.strip()
                brand = name.split()[0] # สมมติว่าคำแรกของชื่อคือแบรนด์
                
                raw_dpi = item.find('span', class_='spec-dpi').text.strip()
                raw_weight = item.find('span', class_='spec-weight').text.strip()
                raw_price = item.find('span', class_='price').text.strip()
                
                # 2. ทำความสะอาดข้อมูล (Data Cleaning) ด้วย Regex
                # ดึงเฉพาะตัวเลขออกจากข้อความ เช่น "25,600 DPI" -> 25600
                dpi_match = re.search(r'\d+', raw_dpi.replace(',', ''))
                dpi = int(dpi_match.group()) if dpi_match else 0
                
                # ดึงเฉพาะตัวเลขน้ำหนัก เช่น "63g" -> 63
                weight_match = re.search(r'\d+', raw_weight)
                weight = int(weight_match.group()) if weight_match else 0
                
                # แปลงราคา
                price_match = re.search(r'\d+\.?\d*', raw_price.replace(',', ''))
                if price_match:
                    # สมมติเว็บเป็น USD แปลงเป็น THB + ภาษี
                    price_thb = int(float(price_match.group()) * 36 * 1.1)
                else:
                    continue

                # 3. กรองข้อมูลขยะออก
                if price_thb < 200 or dpi == 0: 
                    continue

                # 4. จัดเก็บลง Dictionary ให้ตรงกับคอลัมน์ที่คุณต้องการ
                mouse_list.append({
                    'Brand': brand,
                    'Model': name.replace(brand, '').strip(), # ตัดแบรนด์ออกจากชื่อรุ่น
                    'Connectivity': 'Wireless', # ค่าตั้งต้นชั่วคราว (ต้องเขียน Logic ดึงเพิ่ม)
                    'DPI_Max': dpi,
                    'Weight_Grams': weight,
                    'Target_Usage': 'Gaming' if dpi > 10000 else 'Office', # จำแนกกลุ่มเบื้องต้น
                    'Sensor_Type': 'Optical',
                    'Buttons': 6,
                    'Price_THB': price_thb
                })
                count += 1
                
            except AttributeError:
                # ถ้าหา Tag ไม่เจอ (เช่น สินค้าบางตัวข้อมูลไม่ครบ) ให้ข้ามไป
                continue
                
        # --- ส่วนสำคัญ: แปลงข้อมูลเป็น CSV ---
        if mouse_list:
            # นำ List ของ Dictionary เข้า Pandas DataFrame
            df = pd.DataFrame(mouse_list)
            
            # บันทึกเป็นไฟล์ CSV
            csv_filename = 'scraped_mouse_data.csv'
            # ใช้ utf-8-sig เพื่อให้เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig') 
            
            print(f'✅ ดูดข้อมูลสำเร็จ! เมาส์: {count} รุ่น')
            print(f'📁 บันทึกไฟล์เรียบร้อย: {csv_filename}')
        else:
            print("⚠️ ไม่พบข้อมูลที่ตรงตามเงื่อนไข หรือโครงสร้าง HTML ไม่ถูกต้อง")
            
    except Exception as e: 
        print(f'❌ Mouse Scrape Error: {e}')

# สั่งรันโค้ด
if __name__ == '__main__':
    scrape_mouse_to_csv()