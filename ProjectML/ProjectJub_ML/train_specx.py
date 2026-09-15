import pandas as pd
import numpy as np
import random
import joblib
import os
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

print("🧠 เริ่มต้นการสร้าง Dataset (Version: Pro Smart Constraints)...")

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

# --- 2. ล้างข้อมูลให้สะอาด (Data Cleansing) ---
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

# สร้างฟังก์ชันคำนวณ TDP แบบ Dynamic อิงจากความแรง
def get_cpu_tdp(score):
    if score > 35000: return 150
    elif score > 20000: return 105
    elif score > 10000: return 65
    return 45
df_cpu['est_tdp'] = df_cpu['score'].apply(get_cpu_tdp)

def get_gpu_tdp(score):
    if score > 25000: return 280
    elif score > 15000: return 170
    elif score > 8000: return 120
    return 75
df_gpu['est_tdp'] = df_gpu['score'].apply(get_gpu_tdp)

# --- ฟังก์ชันช่วยวิเคราะห์บอร์ด ---
def parse_mb_specs(spec_text):
    specs = {'ram_slots': 2, 'sata_slots': 4, 'm2_nvme': False, 'm2_slots': 0}
    spec_text = str(spec_text).upper()
    
    ram_match = re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text)
    if ram_match: specs['ram_slots'] = int(ram_match.group(1))
    
    # เช็คว่ามี M.2 และรองรับ NVMe/PCIe หรือไม่
    m2_match = re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text)
    if m2_match: 
        specs['m2_slots'] = int(m2_match.group(1))
        # บอร์ดส่วนใหญ่ที่มี M.2 จะรองรับ NVMe ยกเว้นรุ่นเก่าจัดๆ 
        # ให้เช็คคำว่า NVMe หรือ PCIe ถ้าไม่มีถือว่ารองรับ NVMe เป็นค่าเริ่มต้น 
        # (เพราะ M.2 SATA เริ่มหายากในบอร์ดใหม่)
        specs['m2_nvme'] = True 
        if 'SATA ONLY' in spec_text or 'NON-NVME' in spec_text:
            specs['m2_nvme'] = False
            
    return specs

# --- 3. เริ่มประกอบสเปค ---
builds = []
target_builds = 5000
valid_build_count = 0
attempts = 0

error_tracker = {
    '1_socket_mismatch': 0, '2_ram_ddr_mismatch': 0, 
    '3_storage_mismatch': 0, '4_psu_watt_mismatch': 0
}

print(f"⚙️ กำลังสุ่มประกอบสเปค เป้าหมาย: {target_builds} ชุด...")

while valid_build_count < target_builds and attempts < 20000:
    attempts += 1
    
    # 1. CPU
    cpu = df_cpu.sample(1).iloc[0]
    
    # 2. Mainboard
    mb_options = df_mb[df_mb['clean_socket'] == cpu['clean_socket']]
    if mb_options.empty: 
        error_tracker['1_socket_mismatch'] += 1
        continue
    mb = mb_options.sample(1).iloc[0]
    mb_specs = parse_mb_specs(mb['specifications_mainboards'])
    
    mb_ddr_match = re.search(r'(DDR[345])', str(mb['specifications_mainboards']).upper() + str(mb['name_mainboards']).upper())
    mb_ddr_type = mb_ddr_match.group(1) if mb_ddr_match else "DDR4"
    
    # 3. GPU
    gpu = df_gpu.sample(1).iloc[0]
    
    temp_budget = cpu['price'] + mb['price'] + gpu['price']
    
    # 4. RAM (ปรับลอจิกใหม่)
    ram_options = df_ram[df_ram['clean_ddr'] == mb_ddr_type]
    if ram_options.empty:
        error_tracker['2_ram_ddr_mismatch'] += 1
        continue
        
    # งบน้อยเอา 8GB ลงไป, งบมากเอา 16GB ขึ้นไป
    if temp_budget >= 10000:
        suitable_rams = ram_options[ram_options['capacity_gb'] >= 16]
    else:
        suitable_rams = ram_options[ram_options['capacity_gb'] <= 8]
        
    if suitable_rams.empty: suitable_rams = ram_options # กันเหนียว
    ram = suitable_rams.sample(1).iloc[0]
    
    # พยายามใส่ RAM เป็นคู่ (Dual Channel) ถ้าบอร์ดมี Slot พอ
    ram_qty = 1
    if mb_specs['ram_slots'] >= 2 and (temp_budget >= 10000 or ram['capacity_gb'] < 8):
        ram_qty = 2 # ใส่ 2 แถวถ้างบเยอะ หรือถ้าเจอแรมแถวละ 4GB
    # อย่าลืมจำกัดไม่ให้เกินจำนวนสล็อต
    ram_qty = min(ram_qty, mb_specs['ram_slots'])
    
    # 5. Storage (แก้ปัญหา M.2 ผิดประเภท)
    storage_list = []
    storage_price = 0
    storage_tdp = 0
    
    is_m2_available = mb_specs['m2_slots'] > 0
    
    if is_m2_available and mb_specs['m2_nvme']:
        # บอร์ดรับ NVMe -> เล็งหา NVMe ก่อน
        nvme_options = df_hdd[df_hdd['type_hdd'].astype(str).str.contains('NVMe|PCIe', case=False, na=False)]
        if not nvme_options.empty:
            st = nvme_options.sample(1).iloc[0]
            storage_list.append(st)
            storage_price += st['price']
            storage_tdp += 8 # ตีไฟ M.2 NVMe ลูกละ 8W
    else:
        # บอร์ดไม่มี M.2 หรือไม่รับ NVMe -> หา SSD SATA หรือ HDD
        sata_options = df_hdd[df_hdd['type_hdd'].astype(str).str.contains('SATA', case=False, na=False)]
        if not sata_options.empty:
            st = sata_options.sample(1).iloc[0]
            storage_list.append(st)
            storage_price += st['price']
            storage_tdp += 5 # ตีไฟ SATA ลูกละ 5W

    if not storage_list:
        error_tracker['3_storage_mismatch'] += 1
        continue
        
    # 6. PSU (คำนวณไฟละเอียดยิบ)
    # CPU + GPU + RAM (แถวละ 5W) + Storage (ตามชนิด) + บอร์ด/พัดลม (50W)
    total_tdp = cpu['est_tdp'] + gpu['est_tdp'] + (5 * ram_qty) + storage_tdp + 50 
    required_wattage = total_tdp * 1.3
    
    psu_options = df_psu[df_psu['wattage'] >= required_wattage]
    if psu_options.empty:
        error_tracker['4_psu_watt_mismatch'] += 1
        continue
    psu = psu_options.sort_values('wattage').sample(1).iloc[0]

    total_price = temp_budget + (ram['price'] * ram_qty) + storage_price + psu['price']

    builds.append({
        'cpu': cpu['name_cpu'],
        'mb': mb['name_mainboards'],
        'gpu': gpu['name_gpu'],
        'ram': f"{ram['name__ram']} x{ram_qty}",
        'psu': psu['name_powersupplies'],
        'total_price': total_price,
        'benchmark_cpu': cpu['score'],
        'passmark_gpu': gpu['score']
    })
    
    valid_build_count += 1
    if valid_build_count % 500 == 0: print(f"   ✓ ประกอบสำเร็จ {valid_build_count}/{target_builds} ชุด...")

print("\n📊 --- รายงานปัญหา (Error Report) ---")
print(f"ความพยายามทั้งหมด: {attempts} ครั้ง")
print(f"ประกอบสำเร็จ: {valid_build_count} ชุด")
print(f"ปัญหา Socket: {error_tracker['1_socket_mismatch']}")
print(f"ปัญหา RAM DDR: {error_tracker['2_ram_ddr_mismatch']}")
print(f"ปัญหา Storage: {error_tracker['3_storage_mismatch']}")
print(f"ปัญหา PSU ไฟไม่พอ: {error_tracker['4_psu_watt_mismatch']}")
print("------------------------------------\n")

if valid_build_count > 0:
    # --- 4. เตรียมข้อมูลเทรน AI ---
    df_train = pd.DataFrame(builds)
    scaler_cpu = MinMaxScaler().fit(df_train[['benchmark_cpu']])
    scaler_gpu = MinMaxScaler().fit(df_train[['passmark_gpu']])

    df_train['cpu_scaled'] = scaler_cpu.transform(df_train[['benchmark_cpu']])
    df_train['gpu_scaled'] = scaler_gpu.transform(df_train[['passmark_gpu']])
    df_train['final_score'] = (df_train['cpu_scaled'] * 0.5) + (df_train['gpu_scaled'] * 0.5)

    conditions = [(df_train['final_score'] > 0.75), (df_train['final_score'] > 0.45)]
    df_train['level'] = np.select(conditions, ['Pro', 'Standard'], default='Basic')

    X = df_train[['cpu_scaled', 'gpu_scaled', 'total_price']]
    y = df_train['level']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'pc_spec_classifier9.pkl')
    joblib.dump(scaler_cpu, 'scaler_cpu9.pkl')
    joblib.dump(scaler_gpu, 'scaler_gpu9.pkl')
    df_train.to_csv('generated_pc_builds.csv', index=False)
    print("💾 เทรนและบันทึกโมเดลเรียบร้อย! พร้อมใช้งานกับไฟล์ test_budget.py แล้ว")
else:
    print("❌ สเปคยังคงเป็น 0 กรุณาเช็คข้อมูลดิบอีกครั้ง")