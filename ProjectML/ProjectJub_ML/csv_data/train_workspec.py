import pandas as pd
import numpy as np
import random
import joblib
import os
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

print("🎨 เริ่มต้นการสร้าง Dataset (Version: Workstation & Creator Pro 1TB+)...")

def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)

def safe_load(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        print(f"❌ ไม่พบไฟล์: {filename}")
        return None

# --- 🎯 ฟังก์ชันใหม่: แปลงขนาดความจุ Storage ให้เป็น GB ---
def parse_storage_gb(size_str):
    size_str = str(size_str).upper()
    match = re.search(r'([\d.]+)\s*(TB|GB)', size_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == 'TB':
            return val * 1000  # แปลง TB เป็น GB
        elif unit == 'GB':
            return val
    return 500 # ค่าเริ่มต้น

# --- 1. โหลดข้อมูล ---
df_cpu = safe_load('combined_cpu_data_strict_pc_only.csv')
df_gpu = safe_load('pc_gpu_data.csv')
df_mb = safe_load('mainboards.csv')
df_ram = safe_load('ram_data.csv')
df_psu = safe_load('powersupplies.csv')
df_hdd = safe_load('hdd_list_all.csv')

# --- 1.5 ฟิลเตอร์เฉพาะของแรงสาย Workstation ---
print("🔍 กำลังคัดกรองเฉพาะชิ้นส่วนระดับ Workstation / Creator...")

workstation_gpu_keywords = 'QUADRO|RTX A|TITAN|RTX PRO|BLACKWELL|ADA|RTX 3080|RTX 3090|RTX 4070|RTX 4080|RTX 4090|RTX 5080|RTX 5090|RADEON PRO|RX 7900'
if df_gpu is not None:
    df_gpu = df_gpu[df_gpu['name_gpu'].astype(str).str.contains(workstation_gpu_keywords, case=False, na=False)]

workstation_cpu_keywords = 'I7|I9|XEON|RYZEN 7|RYZEN 9|THREADRIPPER'
if df_cpu is not None:
    df_cpu = df_cpu[df_cpu['name_cpu'].astype(str).str.contains(workstation_cpu_keywords, case=False, na=False)]

if df_gpu.empty or df_cpu.empty:
    print("⚠️ คำเตือน: ข้อมูล GPU หรือ CPU สาย Workstation มีน้อยเกินไป อาจประกอบสเปคได้ยาก")

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

# 🎯 แปลงความจุ HDD/SSD เป็น GB ให้พร้อมสำหรับการคัดกรอง
df_hdd['capacity_gb'] = df_hdd['size_hdd'].apply(parse_storage_gb)

def get_cpu_tdp(score, name):
    name = str(name).upper()
    if 'THREADRIPPER' in name or 'XEON' in name: return 250
    if score > 35000: return 150
    if score > 20000: return 105
    return 65
df_cpu['est_tdp'] = df_cpu.apply(lambda row: get_cpu_tdp(row['score'], row['name_cpu']), axis=1)

def get_gpu_tdp(score, name):
    name = str(name).upper()
    if '4090' in name or '3090' in name: return 450
    if 'QUADRO' in name or 'RTX A' in name or 'PRO' in name or 'BLACKWELL' in name: return 250 
    if score > 25000: return 300
    return 150
df_gpu['est_tdp'] = df_gpu.apply(lambda row: get_gpu_tdp(row['score'], row['name_gpu']), axis=1)

def parse_mb_specs(spec_text):
    specs = {'ram_slots': 4, 'sata_slots': 4, 'm2_nvme': True, 'm2_slots': 1} 
    spec_text = str(spec_text).upper()
    ram_match = re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text)
    if ram_match: specs['ram_slots'] = int(ram_match.group(1))
    m2_match = re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text)
    if m2_match: specs['m2_slots'] = int(m2_match.group(1))
    return specs

# --- 3. เริ่มประกอบสเปค ---
builds = []
target_builds = 5000
valid_build_count = 0
attempts = 0

error_tracker = {'1_socket_mismatch': 0, '2_ram_ddr_mismatch': 0, '3_storage_mismatch': 0, '4_psu_watt_mismatch': 0}

print(f"⚙️ กำลังสุ่มประกอบสเปคสาย Workstation เป้าหมาย: {target_builds} ชุด...")

while valid_build_count < target_builds and attempts < 30000:
    attempts += 1
    
    cpu = df_cpu.sample(1).iloc[0]
    
    mb_options = df_mb[df_mb['clean_socket'] == cpu['clean_socket']]
    if mb_options.empty: 
        error_tracker['1_socket_mismatch'] += 1
        continue
    mb = mb_options.sample(1).iloc[0]
    mb_specs = parse_mb_specs(mb['specifications_mainboards'])
    
    mb_ddr_match = re.search(r'(DDR[345])', str(mb['specifications_mainboards']).upper() + str(mb['name_mainboards']).upper())
    mb_ddr_type = mb_ddr_match.group(1) if mb_ddr_match else "DDR4"
    
    gpu = df_gpu.sample(1).iloc[0]
    
    temp_budget = cpu['price'] + mb['price'] + gpu['price']
    
    # 🎯 งานกราฟิกต้องการ RAM ขั้นต่ำ 32GB 
    ram_options = df_ram[df_ram['clean_ddr'] == mb_ddr_type]
    suitable_rams = ram_options[ram_options['capacity_gb'] >= 16] 
    
    if suitable_rams.empty:
        error_tracker['2_ram_ddr_mismatch'] += 1
        continue
        
    ram = suitable_rams.sample(1).iloc[0]
    ram_qty = 2 
    ram_qty = min(ram_qty, mb_specs['ram_slots'])
    
    # 🎯 สายตัดต่อ เน้น NVMe SSD และความจุขั้นต่ำ 1TB (1000GB)
    storage_list = []
    storage_price = 0
    storage_tdp = 0
    
    # ดึงเฉพาะที่เป็น NVMe/PCIe และความจุ >= 1000GB
    nvme_options = df_hdd[(df_hdd['type_hdd'].astype(str).str.contains('NVMe|PCIe', case=False, na=False)) & 
                          (df_hdd['capacity_gb'] >= 1000)]
                          
    if not nvme_options.empty and mb_specs['m2_slots'] > 0:
        st = nvme_options.sample(1).iloc[0]
        storage_list.append(st)
        storage_price += st['price']
        storage_tdp += 10
    else:
        error_tracker['3_storage_mismatch'] += 1
        continue # ข้ามถ้าไม่มี NVMe 1TB+ 
        
    total_tdp = cpu['est_tdp'] + gpu['est_tdp'] + (5 * ram_qty) + storage_tdp + 60 
    required_wattage = total_tdp * 1.4 
    
    psu_options = df_psu[df_psu['wattage'] >= required_wattage]
    if psu_options.empty:
        error_tracker['4_psu_watt_mismatch'] += 1
        continue
    psu = psu_options.sort_values('wattage', ascending=False).sample(1).iloc[0]

    total_price = temp_budget + (ram['price'] * ram_qty) + storage_price + psu['price']

    builds.append({
        'cpu': cpu['name_cpu'],
        'mb': mb['name_mainboards'],
        'gpu': gpu['name_gpu'],
        'ram': f"{ram['name__ram']} x{ram_qty} (Total: {ram['capacity_gb'] * ram_qty}GB)",
        'psu': psu['name_powersupplies'],
        'storage': f"{storage_list[0]['drive_name_hdd']} ({storage_list[0]['size_hdd']})",
        'total_price': total_price,
        'benchmark_cpu': cpu['score'],
        'passmark_gpu': gpu['score']
    })
    
    valid_build_count += 1
    if valid_build_count % 500 == 0: print(f"  ✓ ประกอบสำเร็จ {valid_build_count}/{target_builds} ชุด...")

print("\n📊 --- รายงานปัญหา (Error Report) ---")
print(f"ความพยายามทั้งหมด: {attempts} ครั้ง")
print(f"ประกอบสำเร็จ: {valid_build_count} ชุด")
print(f"รายละเอียด Error: {error_tracker}")
print("------------------------------------\n")

if valid_build_count > 0:
    df_train = pd.DataFrame(builds)
    scaler_cpu = MinMaxScaler().fit(df_train[['benchmark_cpu']])
    scaler_gpu = MinMaxScaler().fit(df_train[['passmark_gpu']])

    df_train['cpu_scaled'] = scaler_cpu.transform(df_train[['benchmark_cpu']])
    df_train['gpu_scaled'] = scaler_gpu.transform(df_train[['passmark_gpu']])
    
    df_train['final_score'] = (df_train['cpu_scaled'] * 0.45) + (df_train['gpu_scaled'] * 0.55)

    conditions = [
        (df_train['final_score'] > 0.80), 
        (df_train['final_score'] > 0.50)  
    ]
    df_train['level'] = np.select(conditions, ['Ultimate Workstation', 'Pro Creator'], default='Entry Creator')

    X = df_train[['cpu_scaled', 'gpu_scaled', 'total_price']]
    y = df_train['level']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'workstation_spec_classifier1.pkl')
    joblib.dump(scaler_cpu, 'workstation_scaler_cpu1.pkl')
    joblib.dump(scaler_gpu, 'workstation_scaler_gpu1.pkl')
    df_train.to_csv('generated_workstation_builds.csv', index=False)
    print("💾 เทรนและบันทึกโมเดลสาย Workstation เรียบร้อย! (ไฟล์: workstation_spec_classifier.pkl)")
else:
    print("❌ สเปคเป็น 0 สาเหตุน่าจะมาจากใน CSV ของคุณไม่มีข้อมูล GPU/CPU สาย Workstation หรือ SSD NVMe 1TB+ มากพอ")