import pandas as pd
import joblib
import os
import re
import warnings

# ปิดแจ้งเตือนจุกจิก
warnings.filterwarnings("ignore")

# --- 1. โหลดสมอง AI และตัวปรับสเกล (สาย Workstation) ---
try:
    model = joblib.load('workstation_spec_classifier.pkl')
    scaler_cpu = joblib.load('workstation_scaler_cpu.pkl')
    scaler_gpu = joblib.load('workstation_scaler_gpu.pkl')
    print("🎨 AI Workstation และตัวปรับสเกลพร้อมทำงาน...")
except Exception as e:
    print(f"❌ ไม่พบไฟล์โมเดล: {e} (กรุณารันไฟล์สร้าง Dataset ของ Workstation ก่อน)")
    exit()

# --- 2. ฟังก์ชันล้างข้อมูลแบบปลอดภัย ---
def safe_clean(series):
    s = series.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def get_cpu_tdp(score, name):
    name = str(name).upper()
    if 'THREADRIPPER' in name or 'XEON' in name: return 250
    if score > 35000: return 150
    if score > 20000: return 105
    return 65

def get_gpu_tdp(score, name):
    name = str(name).upper()
    if '4090' in name or '3090' in name: return 450
    if 'QUADRO' in name or 'RTX A' in name or 'PRO' in name or 'BLACKWELL' in name: return 250
    if score > 25000: return 300
    return 150

def load_data():
    df_cpu = pd.read_csv('combined_cpu_data_strict_pc_only.csv')
    df_mb = pd.read_csv('mainboards.csv')
    df_ram = pd.read_csv('ram_data.csv')
    df_gpu = pd.read_csv('pc_gpu_data.csv') if os.path.exists('pc_gpu_data.csv') else None
    df_psu = pd.read_csv('powersupplies.csv')
    df_hdd = pd.read_csv('hdd_list_all.csv')

    # 🎯 กรองเฉพาะอุปกรณ์สาย Workstation / High-end
    gpu_kw = 'QUADRO|RTX A|TITAN|RTX PRO|BLACKWELL|ADA|RTX 3080|RTX 3090|RTX 4070|RTX 4080|RTX 4090|RTX 5080|RTX 5090|RADEON PRO|RX 7900'
    cpu_kw = 'I7|I9|XEON|RYZEN 7|RYZEN 9|THREADRIPPER'
    
    df_gpu = df_gpu[df_gpu['name_gpu'].astype(str).str.contains(gpu_kw, case=False, na=False)]
    df_cpu = df_cpu[df_cpu['name_cpu'].astype(str).str.contains(cpu_kw, case=False, na=False)]

    df_cpu['price'] = safe_clean(df_cpu['price_cpu'])
    df_mb['price'] = safe_clean(df_mb['price_mainboards'])
    df_ram['price'] = safe_clean(df_ram['price_ram'])
    df_gpu['price'] = safe_clean(df_gpu['price_gpu'])
    df_psu['price'] = safe_clean(df_psu['price_powersupplies'])
    df_hdd['price'] = safe_clean(df_hdd['price_hdd'])

    df_cpu['score'] = safe_clean(df_cpu['benchmark_cpu'])
    df_gpu['score'] = safe_clean(df_gpu['passmark_gpu'])

    df_cpu['est_tdp'] = df_cpu.apply(lambda row: get_cpu_tdp(row['score'], row['name_cpu']), axis=1)
    df_gpu['est_tdp'] = df_gpu.apply(lambda row: get_gpu_tdp(row['score'], row['name_gpu']), axis=1)

    df_ram['capacity_gb'] = df_ram['name__ram'].astype(str).str.upper().str.extract(r'(\d+)\s*G')[0].astype(float).fillna(8.0)
    df_psu['wattage'] = df_psu['name_powersupplies'].astype(str).str.upper().str.extract(r'(\d+)\s*(?:W|WATT)')[0].astype(float).fillna(500)
    df_cpu['clean_socket'] = df_cpu['socket_cpu'].astype(str).str.upper().str.replace(r'(SOCKET|LGA|\s|-)', '', regex=True)
    df_mb['clean_socket'] = df_mb['socket_mainboards'].astype(str).str.upper().str.replace(r'(SOCKET|LGA|\s|-)', '', regex=True)
    df_ram['clean_ddr'] = df_ram['ddr_type_ram'].astype(str).str.upper().str.extract(r'(DDR[345])')[0].fillna('UNKNOWN')

    return df_cpu, df_gpu, df_mb, df_ram, df_psu, df_hdd

df_cpu, df_gpu, df_mb, df_ram, df_psu, df_hdd = load_data()

def parse_mb_slots(spec_text):
    spec_text = str(spec_text).upper()
    ram_slots = int(re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text).group(1)) if re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text) else 4
    m2_slots = int(re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text).group(1)) if re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text) else 1
    return ram_slots, m2_slots

# --- 3. ฟังก์ชันจัดสเปคแบบ Workstation Constraints ---
def get_workstation_spec(budget):
    # งาน 3D/เรนเดอร์ ให้งบ GPU มากหน่อย (45%) และ CPU (25%)
    gpu_b = budget * 0.45
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
            # บอร์ดทำงานเอาราคากลางๆ ค่อนไปทางสูงเพื่อภาคจ่ายไฟที่ดี
            matched_mb = mb_pool.iloc[int(len(mb_pool)*0.3)] if len(mb_pool) > 3 else mb_pool.iloc[-1]
            break
            
    if matched_cpu is None:
        for _, cpu_candidate in df_cpu.sort_values('price').iterrows():
            mb_pool = df_mb[df_mb['clean_socket'] == cpu_candidate['clean_socket']].sort_values('price')
            if not mb_pool.empty:
                matched_cpu = cpu_candidate
                matched_mb = mb_pool.iloc[-1]
                break
                
    if matched_cpu is None:
        raise ValueError("ไม่มีคู่ CPU และ Mainboard สาย Workstation ที่เข้ากันได้")
        
    cpu = matched_cpu
    mb = matched_mb

    ram_slots, m2_slots = parse_mb_slots(mb['specifications_mainboards'])
    mb_ddr_match = re.search(r'(DDR[345])', str(mb['specifications_mainboards']).upper() + str(mb['name_mainboards']).upper())
    mb_ddr_type = mb_ddr_match.group(1) if mb_ddr_match else "DDR4"

    # 3. เลือก RAM (บังคับ 16GB ต่อแถว และพยายามใส่ 2 แถวให้ได้ 32GB)
    ram_pool = df_ram[(df_ram['clean_ddr'] == mb_ddr_type) & (df_ram['capacity_gb'] >= 16)].sort_values('price')
    if ram_pool.empty:
        # กันเหนียวถ้าไม่มี 16GB
        ram_pool = df_ram[df_ram['clean_ddr'] == mb_ddr_type].sort_values('price')
        if ram_pool.empty: raise ValueError(f"ไม่มี RAM ชนิด: {mb_ddr_type}")

    ram = ram_pool.iloc[0]
    ram_qty = 2 if ram_slots >= 2 else 1 # บังคับ Dual Channel
    ram_tdp = 5 * ram_qty 

    # 4. เลือก Storage (บังคับ NVMe งานครีเอเตอร์ต้องเร็ว)
    storage_list, storage_price, storage_tdp = [], 0, 0
    nvme_pool = df_hdd[df_hdd['type_hdd'].astype(str).str.contains('NVMe|PCIe', case=False, na=False)].sort_values('price', ascending=False)
    
    if not nvme_pool.empty:
        # พยายามหาความจุสูงๆ ถ้างบเหลือ
        st = nvme_pool[nvme_pool['price'] <= (budget * 0.1)].iloc[0] if not nvme_pool[nvme_pool['price'] <= (budget * 0.1)].empty else nvme_pool.iloc[-1]
        storage_name = st['drive_name_hdd'] if 'drive_name_hdd' in st else st['type_hdd']
        storage_list.append(storage_name)
        storage_price += st['price']
        storage_tdp += 10
    else:
        raise ValueError("ไม่พบ SSD NVMe ในฐานข้อมูล (งาน Workstation บังคับใช้ NVMe)")

    # 5. เลือก PSU (เผื่อ 1.4x สำคัญมากเวลารันเรนเดอร์ข้ามคืน)
    total_tdp = cpu['est_tdp'] + gpu['est_tdp'] + ram_tdp + storage_tdp + 60
    req_wattage = total_tdp * 1.4 
    
    psu_pool = df_psu[df_psu['wattage'] >= req_wattage]
    if psu_pool.empty: raise ValueError(f"ไม่มี PSU ที่จ่ายไฟได้ถึง {req_wattage:.0f}W")
    
    # PSU สายทำงานควรจะคุณภาพสูง
    min_psu_price = max(budget * 0.08, req_wattage * 2.5)
    safe_psu_pool = psu_pool[psu_pool['price'] >= min_psu_price].sort_values('price')
    
    if not safe_psu_pool.empty:
        psu = safe_psu_pool.iloc[0]
    else:
        psu = psu_pool.sort_values('price', ascending=False).iloc[0]

    # --- 6. คำนวณราคารวมทั้งหมด ---
    total_price = cpu['price'] + gpu['price'] + mb['price'] + (ram['price'] * ram_qty) + storage_price + psu['price']

    # --- 7. ให้ AI ทำนาย ---
    s_cpu = scaler_cpu.transform([[cpu['score']]])[0][0]
    s_gpu = scaler_gpu.transform([[gpu['score']]])[0][0]
    
    input_df = pd.DataFrame({'cpu_scaled': [s_cpu], 'gpu_scaled': [s_gpu], 'total_price': [total_price]})
    prediction = model.predict(input_df)[0]

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
print("\n" + "="*55)
print(" 🖥️ WORKSTATION & CREATOR BUILDER AI (V1.0)")
print(" เน้น 3D/Render/Video Editing (บังคับ RAM 32GB+ & NVMe)")
print("="*55)

while True:
    try:
        user_input = input("\n💰 กรุณากรอกงบประมาณ (บาท) [พิมพ์ 0 เพื่อออก]: ")
        val = float(user_input)
        
        if val == 0:
            print("👋 ขอให้สนุกกับการทำงาน")
            break
            
        if val < 30000:
            print("⚠️ งบน้อยกว่า 30,000 บาท อาจหาชิ้นส่วนกลุ่ม Workstation มือหนึ่งยาก ระบบจะพยายามหาของที่ถูกที่สุดให้...")
            
        res = get_workstation_spec(val)
        
        print(f"\n✅ ประกอบสเปคสายทำงานสำเร็จ!")
        print(f"🔸 CPU:     {res['CPU']}")
        print(f"🔸 MB:      {res['MB']}")
        print(f"🔸 GPU:     {res['GPU']}")
        print(f"🔸 RAM:     {res['RAM']}")
        print(f"🔸 Storage: {res['Storage']}")
        print(f"🔸 PSU:     {res['PSU']}")
        print(f"-"*45)
        print(f"💵 ราคารวมทั้งเครื่อง: {res['Total_Price']:,.0f} บาท (จากงบ {val:,.0f} บาท)")
        print(f"⭐ ระดับ Workstation: ** {res['Rank']} **")
        print(f"📊 คะแนนสเกล (0-1):  {res['Scale']}")
        
    except ValueError as ve:
        if "could not convert" in str(ve):
            print("❌ กรุณากรอกตัวเลขงบประมาณให้ถูกต้อง")
        else:
            print(f"❌ จัดสเปคไม่สำเร็จ: {ve}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")