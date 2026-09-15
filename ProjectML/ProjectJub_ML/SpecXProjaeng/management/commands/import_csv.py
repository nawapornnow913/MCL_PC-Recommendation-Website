import pandas as pd
import re
import os
from django.core.management.base import BaseCommand
from SpecXProjaeng.models import CPU, GPU, Component

class Command(BaseCommand):
    help = 'Smart Import CSV with Targeted Schema Parsing (ML-Grade ETL)'

    def normalize_socket(self, val):
        if pd.isna(val): return "OTHER"
        s = str(val).upper()
        if '1700' in s: return '1700'
        if '1200' in s: return '1200'
        if '1151' in s: return '1151'
        if 'AM4' in s: return 'AM4'
        if 'AM5' in s: return 'AM5'
        return re.sub(r'[^A-Z0-9]', '', s)

    def parse_price(self, val, is_usd=False):
        if pd.isna(val) or str(val).strip() == '': return 0
        num = re.sub(r'[^\d.]', '', str(val))
        try:
            price = float(num)
            return int(price * 47) if is_usd else int(price)
        except: return 0

    def handle(self, *args, **kwargs):
        data_dir = 'csv_data/' 
        
        self.stdout.write("⏳ กำลังเริ่มระบบ ETL อัจฉริยะ (อ่านโครงสร้างไฟล์จาก Header จริง)...")
        CPU.objects.all().delete()
        GPU.objects.all().delete()
        Component.objects.all().delete()

        # 1. นำเข้า CPU
        try:
            df_cpu = pd.read_csv(os.path.join(data_dir, 'combined_cpu_data_strict_pc_only.csv'))
            for _, row in df_cpu.iterrows():
                p = self.parse_price(row.get('price_cpu', 0))
                if p > 0:
                    CPU.objects.create(
                        name=str(row.get('name_cpu', 'Unknown')),
                        benchmark_score=int(float(row.get('benchmark_cpu', 0))),
                        price=p,
                        socket=self.normalize_socket(row.get('socket_cpu', ''))
                    )
            self.stdout.write(self.style.SUCCESS('✅ CPU สำเร็จ'))
        except Exception as e: self.stdout.write(self.style.ERROR(f'❌ CPU Error: {e}'))

        # 2. นำเข้า GPU
        try:
            df_gpu = pd.read_csv(os.path.join(data_dir, 'pc_gpu_data.csv'))
            for _, row in df_gpu.iterrows():
                p = self.parse_price(row.get('price_gpu', 0))
                score = row.get('passmark_gpu', 0)
                if p > 0:
                    GPU.objects.create(
                        name=str(row.get('name_gpu', 'Unknown')),
                        benchmark_score=int(float(score)),
                        price=p
                    )
            self.stdout.write(self.style.SUCCESS('✅ GPU สำเร็จ'))
        except Exception as e: self.stdout.write(self.style.ERROR(f'❌ GPU Error: {e}'))

        # 3. นำเข้า Component อื่นๆ
        def load_generic(file, cat, is_usd=False):
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                self.stdout.write(self.style.WARNING(f'⚠️ ข้าม {file}: ไม่พบไฟล์นี้'))
                return

            try:
                df = pd.read_csv(file_path)
                imported_count = 0
                
                for _, row in df.iterrows():
                    # ----------------------------------------------------
                    # สกัดชื่อ (รองรับ name__ram และ name_powersupplies)
                    # ----------------------------------------------------
                    name = "Unknown"
                    for col_name in ['name__ram', 'name_powersupplies', 'name_mainboards', 'model_name_case', 'drive_name_hdd', 'model_name_moniter', 'model_name_mouse', 'name_keyboards', 'name_headsets', 'name_microphones', 'product_name']:
                        if col_name in row.index and not pd.isna(row[col_name]):
                            name = str(row[col_name])
                            break
                    
                    if name == "Unknown": # แผนสำรองค้นหาจาก Keyword
                        for col in row.index:
                            if 'name' in str(col).lower() or 'model' in str(col).lower():
                                name = str(row[col])
                                break
                                
                    # ----------------------------------------------------
                    # สกัดราคา (รองรับ price_ram และ price_powersupplies)
                    # ----------------------------------------------------
                    p_val = 0
                    for col_price in ['price_ram', 'price_powersupplies', 'price_mainboards', 'price_case', 'price_hdd', 'price_moniter', 'price_mouse', 'price_keyboards', 'price_headsets', 'price_microphones', 'price_thb']:
                        if col_price in row.index and not pd.isna(row[col_price]):
                            p_val = row[col_price]
                            break
                            
                    if p_val == 0: # แผนสำรอง
                        for col in row.index:
                            if 'price' in str(col).lower():
                                p_val = row[col]
                                break

                    price = self.parse_price(p_val, is_usd)
                    
                    if price > 0:
                        spec = ""
                        for col in row.index:
                            if 'spec' in str(col).lower() or 'desc' in str(col).lower() or 'detail' in str(col).lower():
                                if not pd.isna(row[col]): spec = str(row[col]); break
                        
                        combined_text = (name + " " + spec).upper()
                                
                        sock_col = ""
                        for col in row.index:
                            if 'socket' in str(col).lower() and not pd.isna(row[col]): sock_col = row[col]; break
                        sock = self.normalize_socket(sock_col)
                        
                        # ----------------------------------------------------
                        # สกัด RAM Type (รองรับ ddr_type_ram)
                        # ----------------------------------------------------
                        ram = ""
                        if 'ddr_type_ram' in row.index and not pd.isna(row['ddr_type_ram']):
                            ram = str(row['ddr_type_ram']).upper()
                        
                        if not ram:
                            if "DDR5" in combined_text: ram = "DDR5"
                            elif "DDR4" in combined_text: ram = "DDR4"
                            elif "DDR3" in combined_text: ram = "DDR3"
                            elif cat == 'ram': ram = "DDR4" # ถ้าเป็นไฟล์แรมแต่หาชนิดไม่เจอ ให้เดาว่าเป็น DDR4
                            else: ram = ""
                            
                        # ----------------------------------------------------
                        # สกัด Wattage ของ PSU ให้แม่นยำ 100%
                        # ----------------------------------------------------
                        watt = 0
                        # ลองค้นหาตัวเลขที่นำหน้าตัว W ในข้อความ (เช่น 400W, 1050W)
                        w_match = re.search(r'(\d{3,4})\s*W\b', combined_text)
                        if w_match:
                            watt = int(w_match.group(1))
                        else:
                            # ถ้าไม่มี W ให้ลองหาคำว่า Watt
                            watt_match = re.search(r'(\d{3,4})\s*WATT', combined_text)
                            if watt_match:
                                watt = int(watt_match.group(1))

                        actual_cat = cat
                        if cat == 'mb_check':
                            actual_cat = 'mb_intel' if any(x in sock for x in ['1700','1200','1151']) else 'mb_amd'

                        for col in row.index:
                            if 'form' in str(col).lower() or 'factor' in str(col).lower():
                                if not pd.isna(row[col]):
                                    name = f"[{row[col]}] {name}"
                                    break

                        Component.objects.create(
                            category=actual_cat, name=str(name), price=price,
                            socket=sock, ram_type=ram, watt=watt
                        )
                        imported_count += 1
                
                if imported_count > 0:
                    self.stdout.write(self.style.SUCCESS(f'✅ {file} สำเร็จ (นำเข้า {imported_count} รายการ)'))
                else:
                    self.stdout.write(self.style.WARNING(f'⚠️ {file} อ่านไฟล์ได้ แต่ไม่พบข้อมูลที่ตรงเงื่อนไข (ราคาอาจเป็น 0)'))
            except Exception as e: 
                self.stdout.write(self.style.ERROR(f'❌ {file} Error: {e}'))

        load_generic('mainboards.csv', 'mb_check')
        load_generic('GEAR - CASE.csv', 'case')
        load_generic('hdd_list_all.csv', 'ssd')
        load_generic('GEAR - MONITER.csv', 'monitor')
        load_generic('GEAR - MOUSE.csv', 'mouse')
        load_generic('keyboards.csv', 'keyboard')
        load_generic('headsets.csv', 'headset')
        load_generic('microphones.csv', 'mic')
        
        # บังคับ is_usd=False เพื่อให้แปลงค่าเงินบาทตามความจริง
        load_generic('ram_data.csv', 'ram', False) 
        load_generic('powersupplies.csv', 'psu')
        
        self.stdout.write(self.style.SUCCESS('🎉 นำเข้าสำเร็จทุุกไฟล์ ข้อมูลพร้อมใช้งาน!'))