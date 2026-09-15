import pandas as pd
import joblib
import os
import re
import warnings

# ปิดแจ้งเตือนจุกจิก
warnings.filterwarnings("ignore")

# --- 1. โหลดสมอง AI และตัวปรับสเกล ---
try:
    model = joblib.load('pc_spec_classifier9.pkl')
    scaler_cpu = joblib.load('scaler_cpu9.pkl')
    scaler_gpu = joblib.load('scaler_gpu9.pkl')
    print("🤖 AI และตัวปรับสเกลพร้อมทำงาน...")
except Exception as e:
    print(f"❌ ไม่พบไฟล์โมเดล: {e} (กรุณารันไฟล์ train_specx.py ก่อน)")
    exit()

# --- 2. ฟังก์ชันล้างข้อมูลแบบปลอดภัย ---
def safe_clean(series):
    s = series.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def get_cpu_tdp(score):
    if score > 35000: return 150
    elif score > 20000: return 105
    elif score > 10000: return 65
    return 45

def get_gpu_tdp(score):
    if score > 25000: return 280
    elif score > 15000: return 170
    elif score > 8000: return 120
    return 75

def load_data():
    df_cpu = pd.read_csv('combined_cpu_data_strict_pc_only.csv')
    df_mb = pd.read_csv('mainboards.csv')
    df_ram = pd.read_csv('ram_data.csv')
    df_gpu = pd.read_csv('pc_gpu_data.csv') if os.path.exists('pc_gpu_data.csv') else None
    df_psu = pd.read_csv('powersupplies.csv')
    df_hdd = pd.read_csv('hdd_list_all.csv')

    df_cpu['price'] = safe_clean(df_cpu['price_cpu'])
    df_mb['price'] = safe_clean(df_mb['price_mainboards'])
    df_ram['price'] = safe_clean(df_ram['price_ram'])
    df_gpu['price'] = safe_clean(df_gpu['price_gpu'])
    df_psu['price'] = safe_clean(df_psu['price_powersupplies'])
    df_hdd['price'] = safe_clean(df_hdd['price_hdd'])

    df_cpu['score'] = safe_clean(df_cpu['benchmark_cpu'])
    df_gpu['score'] = safe_clean(df_gpu['passmark_gpu'])

    df_cpu['est_tdp'] = df_cpu['score'].apply(get_cpu_tdp)
    df_gpu['est_tdp'] = df_gpu['score'].apply(get_gpu_tdp)

    df_ram['capacity_gb'] = df_ram['name__ram'].astype(str).str.upper().str.extract(r'(\d+)\s*G')[0].astype(float).fillna(8.0)
    df_psu['wattage'] = df_psu['name_powersupplies'].astype(str).str.upper().str.extract(r'(\d+)\s*(?:W|WATT)')[0].astype(float).fillna(500)
    df_cpu['clean_socket'] = df_cpu['socket_cpu'].astype(str).str.upper().str.replace(r'(SOCKET|LGA|\s|-)', '', regex=True)
    df_mb['clean_socket'] = df_mb['socket_mainboards'].astype(str).str.upper().str.replace(r'(SOCKET|LGA|\s|-)', '', regex=True)
    df_ram['clean_ddr'] = df_ram['ddr_type_ram'].astype(str).str.upper().str.extract(r'(DDR[345])')[0].fillna('UNKNOWN')

    return df_cpu, df_gpu, df_mb, df_ram, df_psu, df_hdd

df_cpu, df_gpu, df_mb, df_ram, df_psu, df_hdd = load_data()

def parse_mb_slots(spec_text):
    spec_text = str(spec_text).upper()
    ram_slots = int(re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text).group(1)) if re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text) else 2
    
    m2_match = re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text)
    m2_slots = int(m2_match.group(1)) if m2_match else 0
    sata_slots = int(re.search(r'(\d+)\s*[X\*]\s*SATA', spec_text).group(1)) if re.search(r'(\d+)\s*[X\*]\s*SATA', spec_text) else 4
    
    m2_nvme = False
    if m2_slots > 0:
        m2_nvme = True
        if 'SATA ONLY' in spec_text or 'NON-NVME' in spec_text:
            m2_nvme = False
            
    return ram_slots, m2_slots, sata_slots, m2_nvme

# --- 3. ฟังก์ชันจัดสเปคแบบใช้ Constraint ---
def get_balanced_spec(budget):
    gpu_b = budget * 0.40
    cpu_b = budget * 0.25
    
    # 1. เลือก GPU
    gpu_pool = df_gpu[df_gpu['price'] <= gpu_b].sort_values('score', ascending=False)
    gpu = gpu_pool.iloc[0] if not gpu_pool.empty else df_gpu.sort_values('price').iloc[0]

    # 2. เลือก CPU และ MB
    cpu_pool = df_cpu[df_cpu['price'] <= cpu_b].sort_values('score', ascending=False)
    matched_cpu, matched_mb = None, None
    
    for _, cpu_candidate in cpu_pool.iterrows():
        mb_pool = df_mb[df_mb['clean_socket'] == cpu_candidate['clean_socket']].sort_values('price')
        if not mb_pool.empty:
            matched_cpu = cpu_candidate
            matched_mb = mb_pool.iloc[0]
            break
            
    if matched_cpu is None:
        for _, cpu_candidate in df_cpu.sort_values('price').iterrows():
            mb_pool = df_mb[df_mb['clean_socket'] == cpu_candidate['clean_socket']].sort_values('price')
            if not mb_pool.empty:
                matched_cpu = cpu_candidate
                matched_mb = mb_pool.iloc[0]
                break
                
    if matched_cpu is None:
        raise ValueError("ไม่มีคู่ CPU และ Mainboard ที่เข้ากันได้ในฐานข้อมูลเลย")
        
    cpu = matched_cpu
    mb = matched_mb

    ram_slots, m2_slots, sata_slots, m2_nvme = parse_mb_slots(mb['specifications_mainboards'])
    mb_ddr_match = re.search(r'(DDR[345])', str(mb['specifications_mainboards']).upper() + str(mb['name_mainboards']).upper())
    mb_ddr_type = mb_ddr_match.group(1) if mb_ddr_match else "DDR4"

    # 3. เลือก RAM
    ram_pool = df_ram[df_ram['clean_ddr'] == mb_ddr_type]
    if ram_pool.empty: raise ValueError(f"ไม่มี RAM ชนิด: {mb_ddr_type}")

    ram_qty = 1
    if budget >= 10000:
        suitable_rams = ram_pool[ram_pool['capacity_gb'] >= 16].sort_values('price')
        if not suitable_rams.empty:
            ram = suitable_rams.iloc[0]
        else:
            suitable_rams_8gb = ram_pool[ram_pool['capacity_gb'] == 8].sort_values('price')
            if not suitable_rams_8gb.empty and ram_slots >= 2:
                ram = suitable_rams_8gb.iloc[0]
                ram_qty = 2
            else:
                ram = ram_pool.sort_values('price', ascending=False).iloc[0]
    else:
        suitable_rams = ram_pool[ram_pool['capacity_gb'] <= 8].sort_values('price')
        if not suitable_rams.empty:
            ram = suitable_rams.iloc[0]
            if ram_slots >= 2 and (ram['price'] * 2) < (budget * 0.15):
                ram_qty = 2
        else:
            ram = ram_pool.sort_values('price').iloc[0]

    ram_qty = min(ram_qty, ram_slots)
    ram_tdp = 5 * ram_qty 

    # 4. เลือก Storage
    storage_list, storage_price, storage_tdp = [], 0, 0
    if m2_slots > 0 and m2_nvme:
        nvme_pool = df_hdd[df_hdd['type_hdd'].astype(str).str.contains('NVMe|PCIe', case=False, na=False)].sort_values('price')
        if not nvme_pool.empty:
            st = nvme_pool.iloc[0]
            storage_list.append(st['name_hdd'] if 'name_hdd' in st else st['type_hdd'])
            storage_price += st['price']
            storage_tdp += 8
    elif sata_slots > 0 or m2_slots > 0:
        sata_pool = df_hdd[df_hdd['type_hdd'].astype(str).str.contains('SATA', case=False, na=False)].sort_values('price')
        if not sata_pool.empty:
            st = sata_pool.iloc[0]
            storage_list.append(st['name_hdd'] if 'name_hdd' in st else st['type_hdd'])
            storage_price += st['price']
            storage_tdp += 5

    # 5. เลือก PSU (เพิ่มลอจิกป้องกัน PSU ระเบิด/ถูกเกินไป)
    total_tdp = cpu['est_tdp'] + gpu['est_tdp'] + ram_tdp + storage_tdp + 50
    req_wattage = total_tdp * 1.4 # เผื่อไฟไว้ 40% เลยให้ชัวร์
    
    psu_pool = df_psu[df_psu['wattage'] >= req_wattage]
    if psu_pool.empty: raise ValueError(f"ไม่มี PSU ที่จ่ายไฟได้ถึง {req_wattage:.0f}W")
    
    # กรองคุณภาพ: ราคาต้องไม่ต่ำกว่า 6% ของงบรวม หรือไม่ต่ำกว่า วัตต์ x 2.0 บาท
    min_psu_price = max(budget * 0.06, req_wattage * 2.0)
    
    safe_psu_pool = psu_pool[psu_pool['price'] >= min_psu_price].sort_values('price')
    
    if not safe_psu_pool.empty:
        # มี PSU ที่คุณภาพ/ราคาผ่านเกณฑ์ -> เลือกตัวที่ถูกที่สุดในกลุ่มที่ได้มาตรฐาน
        psu = safe_psu_pool.iloc[0]
    else:
        # ถ้างบน้อยมากจนหา PSU ตามเกณฑ์ราคาไม่เจอ ให้เอาตัวที่ "แพงที่สุดและดีที่สุด" เท่าที่วัตต์พอมาใช้
        psu = psu_pool.sort_values('price', ascending=False).iloc[0]

    # --- 6. คำนวณราคารวมทั้งหมด ---
    total_price = cpu['price'] + gpu['price'] + mb['price'] + (ram['price'] * ram_qty) + storage_price + psu['price']

    # --- 7. ให้ AI ทำนาย ---
    s_cpu = scaler_cpu.transform([[cpu['score']]])[0][0]
    s_gpu = scaler_gpu.transform([[gpu['score']]])[0][0]
    prediction = model.predict([[s_cpu, s_gpu, total_price]])[0]

    return {
        "CPU": f"{cpu['name_cpu']} ({cpu['price']:,.0f}.-) [~{cpu['est_tdp']}W]",
        "MB": f"{mb['name_mainboards']} ({mb['price']:,.0f}.-)",
        "GPU": f"{gpu['name_gpu']} ({gpu['price']:,.0f}.-) [~{gpu['est_tdp']}W]",
        "RAM": f"{ram['name__ram']} x{ram_qty} ({ram['price']*ram_qty:,.0f}.-) [{ram['capacity_gb']*ram_qty}GB total]",
        "Storage": f"{' + '.join(storage_list)} ({storage_price:,.0f}.-)",
        "PSU": f"{psu['name_powersupplies']} ({psu['price']:,.0f}.-) [ต้องการขั้นต่ำ {req_wattage:.0f}W]",
        "Total_Price": total_price,
        "Rank": prediction,
        "Scale": f"CPU {s_cpu:.2f} | GPU {s_gpu:.2f}"
    }

# --- 4. รันโปรแกรมหลัก ---
print("\n" + "="*50)
print(" 🚀 PC BUILDER AI (V6.0 Anti-Bomb PSU Edition)")
print("="*50)

while True:
    try:
        user_input = input("\n💰 กรุณากรอกงบประมาณ (บาท) [พิมพ์ 0 เพื่อออก]: ")
        val = float(user_input)
        
        if val == 0:
            print("👋 ลาก่อน")
            break
            
        if val < 5000:
            print("⚠️ งบน้อยเกินไป อาจจะจัดสเปคครบชุดไม่ได้ แนะนำงบ 10,000 บาทขึ้นไป")
            continue
            
        res = get_balanced_spec(val)
        
        print(f"\n✅ จัดสเปคสำเร็จ (สเปคเข้ากันได้ 100%)!")
        print(f"🔸 CPU:     {res['CPU']}")
        print(f"🔸 MB:      {res['MB']}")
        print(f"🔸 GPU:     {res['GPU']}")
        print(f"🔸 RAM:     {res['RAM']}")
        print(f"🔸 Storage: {res['Storage']}")
        print(f"🔸 PSU:     {res['PSU']}")
        print(f"-"*40)
        print(f"💵 ราคารวมทั้งเครื่อง: {res['Total_Price']:,.0f} บาท (จากงบ {val:,.0f} บาท)")
        print(f"⭐ ระดับความแรง (AI): {res['Rank']}")
        print(f"📊 คะแนนสเกล (0-1):  {res['Scale']}")
        
    except ValueError as ve:
        if "could not convert" in str(ve):
            print("❌ กรุณากรอกตัวเลขงบประมาณให้ถูกต้อง")
        else:
            print(f"❌ จัดสเปคไม่สำเร็จ: {ve}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")