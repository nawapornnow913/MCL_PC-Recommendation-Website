import pandas as pd
import numpy as np
import random
import joblib
import os
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

print("💼 เริ่มต้นการสร้าง Dataset (Version: Office, Dev & Data Analysis)...")

def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)

def safe_load(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        print(f"❌ ไม่พบไฟล์: {filename}")
        return None

# --- 1. โหลดข้อมูล ---
df_cpu = safe_load('combined_cpu_data_strict_pc_only.csv')
df_gpu = safe_load('pc_gpu_data.csv')
df_mb = safe_load('mainboards.csv')
df_ram = safe_load('ram_data.csv')
df_psu = safe_load('powersupplies.csv')
df_hdd = safe_load('hdd_list_all.csv')

# --- 🎯 1.5 Filter ตัดของแพงเวอร์ออก (สายออฟฟิศไม่จำเป็นต้องใช้การ์ดจอตัวท็อป) ---
print("🔍 กำลังปรับสมดุลชิ้นส่วนให้เหมาะกับงานออฟฟิศและเขียนโค้ด...")
if df_gpu is not None:
    df_gpu['price_temp'] = clean_numeric(df_gpu['price_gpu'])
    # ตัดการ์ดจอที่ราคาเกิน 25,000 บาทออก (พวก 4080, 4090 ไม่จำเป็นสำหรับสายเอกสาร/โค้ด)
    df_gpu = df_gpu[df_gpu['price_temp'] <= 25000].drop(columns=['price_temp'])

# --- 2. ล้างข้อมูลให้สะอาด ---
print("🧹 กำลัง Normalize ข้อมูลและสร้าง Features ใหม่...")

def clean_socket(s):
    if pd.isna(s): return ""
    return re.sub(r'(SOCKET|LGA|INTEL|AMD|\s|-)', '', str(s).upper())

df_cpu['clean_socket'] = df_cpu['socket_cpu'].apply(clean_socket)
df_mb['clean_socket'] = df_mb['socket_mainboards'].apply(clean_socket)
df_ram['clean_ddr'] = df_ram['ddr_type_ram'].astype(str).str.upper().str.extract(r'(DDR[345])')[0].fillna('UNKNOWN')

df_cpu['price'] = clean_numeric(df_cpu['price_cpu'])
df_cpu['score'] = clean_numeric(df_cpu['benchmark_cpu'])
df_gpu['price'] = clean_numeric(df_gpu['price_gpu'])
df_gpu['score'] = clean_numeric(df_gpu['passmark_gpu'])
df_mb['price'] = clean_numeric(df_mb['price_mainboards'])
df_ram['price'] = clean_numeric(df_ram['price_ram'])
df_psu['price'] = clean_numeric(df_psu['price_powersupplies'])
df_hdd['price'] = clean_numeric(df_hdd['price_hdd'])

df_ram['capacity_gb'] = df_ram['name__ram'].astype(str).str.upper().str.extract(r'(\d+)\s*G')[0].astype(float).fillna(8.0)
df_psu['wattage'] = df_psu['name_powersupplies'].astype(str).str.upper().str.extract(r'(\d+)\s*(?:W|WATT)')[0].astype(float).fillna(500)

def get_cpu_tdp(score):
    if score > 35000: return 150
    elif score > 20000: return 105
    elif score > 10000: return 65
    return 45
df_cpu['est_tdp'] = df_cpu['score'].apply(get_cpu_tdp)

def get_gpu_tdp(score):
    if score > 20000: return 200
    elif score > 10000: return 120
    return 75
df_gpu['est_tdp'] = df_gpu['score'].apply(get_gpu_tdp)

def parse_mb_specs(spec_text):
    specs = {'ram_slots': 2, 'm2_slots': 0, 'm2_nvme': False}
    spec_text = str(spec_text).upper()
    ram_match = re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text)
    if ram_match: specs['ram_slots'] = int(ram_match.group(1))
    m2_match = re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text)
    if m2_match: 
        specs['m2_slots'] = int(m2_match.group(1))
        specs['m2_nvme'] = True 
        if 'SATA ONLY' in spec_text or 'NON-NVME' in spec_text:
            specs['m2_nvme'] = False
    return specs

# --- 3. เริ่มประกอบสเปค ---
builds = []
target_builds = 5000
valid_build_count = 0
attempts = 0

print(f"⚙️ กำลังสุ่มประกอบสเปคสายทำงาน เป้าหมาย: {target_builds} ชุด...")

while valid_build_count < target_builds and attempts < 20000:
    attempts += 1
    
    cpu = df_cpu.sample(1).iloc[0]
    mb_options = df_mb[df_mb['clean_socket'] == cpu['clean_socket']]
    if mb_options.empty: continue
    
    mb = mb_options.sample(1).iloc[0]
    mb_specs = parse_mb_specs(mb['specifications_mainboards'])
    
    mb_ddr_match = re.search(r'(DDR[345])', str(mb['specifications_mainboards']).upper() + str(mb['name_mainboards']).upper())
    mb_ddr_type = mb_ddr_match.group(1) if mb_ddr_match else "DDR4"
    
    # สุ่มการ์ดจอ (ที่ราคาไม่แพงเกินไป เพราะเรากรองไว้แล้วด้านบน)
    gpu = df_gpu.sample(1).iloc[0]
    
    temp_budget = cpu['price'] + mb['price'] + gpu['price']
    
    # 🎯 RAM สำหรับสายทำงาน: เน้น 16GB เป็นมาตรฐานขั้นต่ำถ้างบถึง
    ram_options = df_ram[df_ram['clean_ddr'] == mb_ddr_type]
    if ram_options.empty: continue
        
    if temp_budget >= 15000:
        suitable_rams = ram_options[ram_options['capacity_gb'] >= 16] # นักพัฒนา / Data เน้น 16GB-32GB
    else:
        suitable_rams = ram_options[ram_options['capacity_gb'] >= 8] # งานเอกสารทั่วไป 8GB พอไหว
        
    if suitable_rams.empty: suitable_rams = ram_options
    ram = suitable_rams.sample(1).iloc[0]
    
    ram_qty = 1
    if mb_specs['ram_slots'] >= 2 and (temp_budget >= 20000 or ram['capacity_gb'] < 16):
        ram_qty = 2 
    ram_qty = min(ram_qty, mb_specs['ram_slots'])
    
    # 🎯 Storage: งานเอกสาร/โค้ด เน้นความเร็วเปิดโปรแกรม SSD 500GB-1TB กำลังดี
    storage_list = []
    storage_price, storage_tdp = 0, 0
    
    if mb_specs['m2_slots'] > 0 and mb_specs['m2_nvme']:
        nvme_options = df_hdd[df_hdd['type_hdd'].astype(str).str.contains('NVMe|PCIe', case=False, na=False)]
        if not nvme_options.empty:
            st = nvme_options.sample(1).iloc[0]
            storage_list.append(st)
            storage_price += st['price']
            storage_tdp += 8 
    else:
        sata_options = df_hdd[df_hdd['type_hdd'].astype(str).str.contains('SATA', case=False, na=False)]
        if not sata_options.empty:
            st = sata_options.sample(1).iloc[0]
            storage_list.append(st)
            storage_price += st['price']
            storage_tdp += 5 

    if not storage_list: continue
        
    total_tdp = cpu['est_tdp'] + gpu['est_tdp'] + (5 * ram_qty) + storage_tdp + 50 
    required_wattage = total_tdp * 1.3
    
    psu_options = df_psu[df_psu['wattage'] >= required_wattage]
    if psu_options.empty: continue
    psu = psu_options.sort_values('wattage').sample(1).iloc[0]

    total_price = temp_budget + (ram['price'] * ram_qty) + storage_price + psu['price']

    builds.append({
        'cpu': cpu['name_cpu'],
        'mb': mb['name_mainboards'],
        'gpu': gpu['name_gpu'],
        'ram': f"{ram['name__ram']} x{ram_qty}",
        'psu': psu['name_powersupplies'],
        'storage': storage_list[0]['drive_name_hdd'] if 'drive_name_hdd' in storage_list[0] else storage_list[0]['name_hdd'],
        'total_price': total_price,
        'benchmark_cpu': cpu['score'],
        'passmark_gpu': gpu['score']
    })
    
    valid_build_count += 1
    if valid_build_count % 500 == 0: print(f"   ✓ ประกอบสำเร็จ {valid_build_count}/{target_builds} ชุด...")

if valid_build_count > 0:
    df_train = pd.DataFrame(builds)
    scaler_cpu = MinMaxScaler().fit(df_train[['benchmark_cpu']])
    scaler_gpu = MinMaxScaler().fit(df_train[['passmark_gpu']])

    df_train['cpu_scaled'] = scaler_cpu.transform(df_train[['benchmark_cpu']])
    df_train['gpu_scaled'] = scaler_gpu.transform(df_train[['passmark_gpu']])
    
    # 🎯 หัวใจหลัก: ให้คะแนน CPU สำคัญกว่า GPU (70/30) เพราะเป็นสายทำงาน/เขียนโค้ด
    df_train['final_score'] = (df_train['cpu_scaled'] * 0.70) + (df_train['gpu_scaled'] * 0.30)

    # 🎯 เปลี่ยนชื่อ Level ให้ตรงกับสายงาน
    conditions = [
        (df_train['final_score'] > 0.65), # เครื่องแรงรันโมเดล ทำ Data สบาย
        (df_train['final_score'] > 0.40)  # เครื่องออฟฟิศชั้นดี เปิดหลายโปรแกรมลื่น
    ]
    df_train['level'] = np.select(conditions, ['Dev & Data Pro', 'Smart Office'], default='General Admin')

    X = df_train[['cpu_scaled', 'gpu_scaled', 'total_price']]
    y = df_train['level']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # บันทึกไฟล์โมเดลแยกชื่อออกมา เพื่อไม่ให้ปนกับสายกราฟิก
    joblib.dump(model, 'office_spec_classifier.pkl')
    joblib.dump(scaler_cpu, 'office_scaler_cpu.pkl')
    joblib.dump(scaler_gpu, 'office_scaler_gpu.pkl')
    df_train.to_csv('generated_office_builds.csv', index=False)
    
    print("\n✅ เสร็จสิ้น! ได้คอมสายทำงานทั่วไป/เขียนโค้ดทั้งหมด", valid_build_count, "ชุด")
    print("💾 บันทึกไฟล์โมเดลชื่อ: office_spec_classifier.pkl เรียบร้อย!")
else:
    print("❌ สเปคยังคงเป็น 0 กรุณาเช็คข้อมูลดิบอีกครั้ง")