import pandas as pd
import joblib
import os
import re
import warnings

warnings.filterwarnings("ignore")

print("\n" + "="*60)
print(" 💼 OFFICE, DEV & DATA PC BUILDER AI (V1.0)")
print(" เน้น CPU, RAM สบายตา ทำงานลื่นไหล ไม่เน้นการ์ดจอแพง")
print("="*60)

# --- 1. โหลดสมอง AI และตัวปรับสเกล (สาย Office) ---
try:
    model = joblib.load('office_spec_classifier.pkl')
    scaler_cpu = joblib.load('office_scaler_cpu.pkl')
    scaler_gpu = joblib.load('office_scaler_gpu.pkl')
except Exception as e:
    print(f"❌ ไม่พบไฟล์โมเดลสายทำงาน: {e}")
    print("กรุณารันไฟล์ train_office_worker.py ก่อน")
    exit()

# --- 2. ฟังก์ชันช่วยจัดการข้อมูล ---
def safe_clean(series):
    s = series.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def get_cpu_tdp(score):
    if score > 35000: return 150
    if score > 20000: return 105
    return 65

def get_gpu_tdp(score):
    if score > 20000: return 200
    if score > 10000: return 120
    return 75

def load_data():
    df_cpu = pd.read_csv('combined_cpu_data_strict_pc_only.csv')
    df_mb = pd.read_csv('mainboards.csv')
    df_ram = pd.read_csv('ram_data.csv')
    df_gpu = pd.read_csv('pc_gpu_data.csv') if os.path.exists('pc_gpu_data.csv') else None
    df_psu = pd.read_csv('powersupplies.csv')
    df_hdd = pd.read_csv('hdd_list_all.csv')

    # กรองเอาการ์ดจอแพงๆ ออกไปเลย (ไม่เกิน 25,000 บาท)
    if df_gpu is not None:
        df_gpu['price'] = safe_clean(df_gpu['price_gpu'])
        df_gpu = df_gpu[df_gpu['price'] <= 25000]

    df_cpu['price'] = safe_clean(df_cpu['price_cpu'])
    df_mb['price'] = safe_clean(df_mb['price_mainboards'])
    df_ram['price'] = safe_clean(df_ram['price_ram'])
    df_psu['price'] = safe_clean(df_psu['price_powersupplies'])
    df_hdd['price'] = safe_clean(df_hdd['price_hdd'])

    df_cpu['score'] = safe_clean(df_cpu['benchmark_cpu'])
    if df_gpu is not None: df_gpu['score'] = safe_clean(df_gpu['passmark_gpu'])

    df_cpu['est_tdp'] = df_cpu['score'].apply(get_cpu_tdp)
    if df_gpu is not None: df_gpu['est_tdp'] = df_gpu['score'].apply(get_gpu_tdp)

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
    m2_slots = int(re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text).group(1)) if re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text) else 0
    return ram_slots, m2_slots

# --- 3. ฟังก์ชันจัดสเปคแบบสายทำงาน ---
def get_office_spec(budget):
    # 🎯 ลอจิกสายออฟฟิศ: ถ้างบน้อย ไม่ต้องใส่การ์ดจอ ให้ใช้ CPU ล้วนๆ
    use_dedicated_gpu = True
    if budget < 18000:
        use_dedicated_gpu = False
        gpu_b = 0
        cpu_b = budget * 0.40 # ทุ่มงบให้ CPU แรงๆ แทน
    else:
        gpu_b = budget * 0.25 # สาย Dev/Data เอาการ์ดจอระดับกลางไปรันโมเดล
        cpu_b = budget * 0.30

    # 1. เลือก CPU และ MB
    cpu_pool = df_cpu[df_cpu['price'] <= cpu_b].sort_values('score', ascending=False)
    matched_cpu, matched_mb = None, None
    
    # พยายามหลีกเลี่ยง CPU ที่ลงท้ายด้วย 'F' (ไม่มีการ์ดจอในตัว) ถ้างบไม่พอซื้อการ์ดจอแยก
    if not use_dedicated_gpu:
        cpu_pool = cpu_pool[~cpu_pool['name_cpu'].astype(str).str.upper().str.endswith('F')]

    for _, cpu_candidate in cpu_pool.iterrows():
        mb_pool = df_mb[df_mb['clean_socket'] == cpu_candidate['clean_socket']].sort_values('price')
        if not mb_pool.empty:
            matched_cpu = cpu_candidate
            # บอร์ดออฟฟิศไม่ต้องแพงมาก เอาตัวเริ่มต้น-กลาง
            matched_mb = mb_pool.iloc[0] if budget < 25000 else mb_pool.iloc[int(len(mb_pool)*0.2)]
            break
            
    if matched_cpu is None:
        raise ValueError("ไม่มีคู่ CPU และ Mainboard ที่เข้ากับงบประมาณนี้")
        
    cpu = matched_cpu
    mb = matched_mb

    ram_slots, m2_slots = parse_mb_slots(mb['specifications_mainboards'])
    mb_ddr_match = re.search(r'(DDR[345])', str(mb['specifications_mainboards']).upper() + str(mb['name_mainboards']).upper())
    mb_ddr_type = mb_ddr_match.group(1) if mb_ddr_match else "DDR4"

    # 2. เลือก GPU (ถ้ามี)
    gpu = None
    if use_dedicated_gpu:
        gpu_pool = df_gpu[df_gpu['price'] <= gpu_b].sort_values('score', ascending=False)
        if not gpu_pool.empty:
            gpu = gpu_pool.iloc[0]

    # 3. เลือก RAM (มาตรฐานสายทำงานคือ 16GB, ถ้างบเยอะพยายามดันเป็น 32GB)
    target_ram_gb = 32 if budget >= 25000 else 16
    if budget < 12000: target_ram_gb = 8 # งบน้อยจริงๆ ถึงจะยอมให้ 8GB
    
    ram_pool = df_ram[(df_ram['clean_ddr'] == mb_ddr_type) & (df_ram['capacity_gb'] >= (target_ram_gb / 2))].sort_values('price')
    if ram_pool.empty:
        ram_pool = df_ram[df_ram['clean_ddr'] == mb_ddr_type].sort_values('price')
        
    ram = ram_pool.iloc[0]
    ram_qty = 2 if (ram_slots >= 2 and ram['capacity_gb'] * 2 <= target_ram_gb) else 1
    ram_tdp = 5 * ram_qty 

    # 4. เลือก Storage (สายออฟฟิศ 500GB ก็พอ, ถ้างบเยอะค่อย 1TB)
    storage_list, storage_price, storage_tdp = [], 0, 0
    target_storage = 1000 if budget > 20000 else 500
    
    # แปลงความจุชั่วคราวเพื่อเทียบ
    df_hdd['cap_gb'] = df_hdd['size_hdd'].astype(str).str.upper().str.extract(r'([\d.]+)\s*(TB|GB)')[0].astype(float).fillna(500)
    df_hdd.loc[df_hdd['size_hdd'].astype(str).str.upper().str.contains('TB'), 'cap_gb'] *= 1000

    if m2_slots > 0:
        nvme_pool = df_hdd[(df_hdd['type_hdd'].astype(str).str.contains('NVMe|PCIe', case=False, na=False)) & 
                           (df_hdd['cap_gb'] >= target_storage)].sort_values('price')
        if not nvme_pool.empty:
            st = nvme_pool[nvme_pool['price'] <= (budget * 0.15)].iloc[-1] if not nvme_pool[nvme_pool['price'] <= (budget * 0.15)].empty else nvme_pool.iloc[0]
            storage_list.append(st)
            storage_price += st['price']
            storage_tdp += 8
            
    # ถ้าไม่มี NVMe ให้เอา SATA
    if not storage_list:
        sata_pool = df_hdd[(df_hdd['type_hdd'].astype(str).str.contains('SATA', case=False, na=False))].sort_values('price')
        if not sata_pool.empty:
            st = sata_pool.iloc[0]
            storage_list.append(st)
            storage_price += st['price']
            storage_tdp += 5

    # 5. เลือก PSU
    gpu_tdp = gpu['est_tdp'] if gpu is not None else 0
    total_tdp = cpu['est_tdp'] + gpu_tdp + ram_tdp + storage_tdp + 40
    req_wattage = total_tdp * 1.3 # เผื่อไฟไว้นิดหน่อย สายออฟฟิศไม่ต้องเผื่อเยอะเท่าสายเกม
    
    psu_pool = df_psu[df_psu['wattage'] >= req_wattage].sort_values('price')
    psu = psu_pool.iloc[0] if not psu_pool.empty else df_psu.sort_values('wattage', ascending=False).iloc[0]

    # --- 6. คำนวณราคารวมทั้งหมด ---
    gpu_price = gpu['price'] if gpu is not None else 0
    total_price = cpu['price'] + gpu_price + mb['price'] + (ram['price'] * ram_qty) + storage_price + psu['price']

    # --- 7. ให้ AI ทำนาย ---
    s_cpu = scaler_cpu.transform([[cpu['score']]])[0][0]
    # ถ้าไม่มีการ์ดจอ ให้คะแนน GPU เป็น 0 
    gpu_score = gpu['score'] if gpu is not None else 0
    s_gpu = scaler_gpu.transform([[gpu_score]])[0][0]
    
    input_df = pd.DataFrame({'cpu_scaled': [s_cpu], 'gpu_scaled': [s_gpu], 'total_price': [total_price]})
    prediction = model.predict(input_df)[0]
    
    # ปรับข้อความแสดงผล GPU
    gpu_display = f"{gpu['name_gpu']} ({gpu['price']:,.0f}.-)" if gpu is not None else "❌ ไม่ใช้การ์ดจอแยก (ใช้ iGPU ประหยัดงบสำหรับงานเอกสาร)"

    return {
        "CPU": f"{cpu['name_cpu']} ({cpu['price']:,.0f}.-)",
        "MB": f"{mb['name_mainboards']} ({mb['price']:,.0f}.-)",
        "GPU": gpu_display,
        "RAM": f"{ram['name__ram']} x{ram_qty} ({ram['price']*ram_qty:,.0f}.-) [{ram['capacity_gb']*ram_qty}GB total]",
        "Storage": f"{storage_list[0]['drive_name_hdd'] if 'drive_name_hdd' in storage_list[0] else storage_list[0]['name_hdd']} ({storage_price:,.0f}.-)",
        "PSU": f"{psu['name_powersupplies']} ({psu['price']:,.0f}.-) [{psu['wattage']}W]",
        "Total_Price": total_price,
        "Rank": prediction,
        "Scale": f"CPU {s_cpu:.2f} | GPU {s_gpu:.2f}"
    }

# --- 4. รันโปรแกรมหลัก ---
while True:
    try:
        user_input = input("\n💰 กรอกงบประมาณคอมทำงาน (บาท) [พิมพ์ 0 เพื่อออก]: ")
        val = float(user_input)
        
        if val == 0:
            print("👋 ขอให้สนุกกับการทำงาน")
            break
            
        res = get_office_spec(val)
        
        print(f"\n✅ จัดสเปคสายทำงาน/Dev/Data สำเร็จ!")
        print(f"🔸 CPU:     {res['CPU']}")
        print(f"🔸 MB:      {res['MB']}")
        print(f"🔸 GPU:     {res['GPU']}")
        print(f"🔸 RAM:     {res['RAM']} 🎯")
        print(f"🔸 Storage: {res['Storage']}")
        print(f"🔸 PSU:     {res['PSU']}")
        print(f"-"*45)
        print(f"💵 ราคารวมทั้งเครื่อง: {res['Total_Price']:,.0f} บาท (จากงบ {val:,.0f} บาท)")
        print(f"⭐ เหมาะสำหรับสายอาชีพ: ** {res['Rank']} **")
        print(f"📊 พลังประมวลผล (0-1): {res['Scale']}")
        
    except ValueError as ve:
        if "could not convert" in str(ve):
            print("❌ กรุณากรอกตัวเลขงบประมาณให้ถูกต้อง")
        else:
            print(f"❌ จัดสเปคไม่สำเร็จ: {ve}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")