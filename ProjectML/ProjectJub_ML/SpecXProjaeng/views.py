from django.shortcuts import render
from .forms import SpecRequestForm
from django.http import JsonResponse 
import pandas as pd
import joblib
import os
import re
import random
import math
from django.conf import settings
import warnings
import json

warnings.filterwarnings("ignore")

MODEL_PATH = settings.BASE_DIR / 'pc_spec_classifier9.pkl'
SCALER_CPU_PATH = settings.BASE_DIR / 'scaler_cpu9.pkl'
SCALER_GPU_PATH = settings.BASE_DIR / 'scaler_gpu9.pkl'

WS_MODEL_PATH = settings.BASE_DIR / 'workstation_spec_classifier1.pkl'
WS_SCALER_CPU_PATH = settings.BASE_DIR / 'workstation_scaler_cpu1.pkl'
WS_SCALER_GPU_PATH = settings.BASE_DIR / 'workstation_scaler_gpu1.pkl'

OFFICE_MODEL_PATH = settings.BASE_DIR / 'office_spec_classifier.pkl'
OFFICE_SCALER_CPU_PATH = settings.BASE_DIR / 'office_scaler_cpu.pkl'
OFFICE_SCALER_GPU_PATH = settings.BASE_DIR / 'office_scaler_gpu.pkl'

try:
    model = joblib.load(MODEL_PATH)
    default_scaler_cpu = joblib.load(SCALER_CPU_PATH)
    default_scaler_gpu = joblib.load(SCALER_GPU_PATH)
except: model = default_scaler_cpu = default_scaler_gpu = None

try:
    ws_model = joblib.load(WS_MODEL_PATH)
    ws_scaler_cpu = joblib.load(WS_SCALER_CPU_PATH)
    ws_scaler_gpu = joblib.load(WS_SCALER_GPU_PATH)
except: ws_model = ws_scaler_cpu = ws_scaler_gpu = None

try:
    office_model = joblib.load(OFFICE_MODEL_PATH)
    office_scaler_cpu = joblib.load(OFFICE_SCALER_CPU_PATH)
    office_scaler_gpu = joblib.load(OFFICE_SCALER_GPU_PATH)
except: office_model = office_scaler_cpu = office_scaler_gpu = None

EVAL_PATH = settings.BASE_DIR / 'evaluation_results.pkl'
try:
    eval_results = joblib.load(EVAL_PATH)
except:
    eval_results = {}

def safe_clean(series):
    s = series.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def get_cpu_tdp(score):
    if score > 45000: return 250
    elif score > 35000: return 200
    elif score > 20000: return 125
    elif score > 10000: return 65
    return 45

def get_gpu_tdp(score):
    if score > 30000: return 350
    elif score > 25000: return 300
    elif score > 15000: return 200
    elif score > 8000: return 130
    return 75

def get_clean_socket(s):
    s = str(s).upper()
    if '1151' in s:
        if 'V2' in s or 'GEN 8' in s or 'GEN 9' in s: return '1151V2'
        return '1151'
    if '2066' in s: return '2066'
    if 'AM4' in s: return 'AM4'
    if 'AM5' in s: return 'AM5'
    if '1700' in s: return '1700'
    if '1200' in s: return '1200'
    return re.sub(r'(SOCKET|LGA|\s|-)', '', s)

def is_tier_compatible(cpu_name, mb_name):
    cpu_name = str(cpu_name).upper()
    mb_name = str(mb_name).upper()
    
    is_high_end_cpu = re.search(r'(I7|I9|RYZEN 7|RYZEN 9|\d+K\b|\d+KF\b|\d+X\b)', cpu_name)
    is_entry_mb = re.search(r'(H\d+10|A\d+20|M-K|M-V|M-D|HDV)', mb_name)
    
    if is_high_end_cpu and is_entry_mb: 
        return False 

    if re.search(r'(I9|RYZEN 9)', cpu_name) and re.search(r'(B\d+60|B\d+50)', mb_name):
        return False
        
    return True

def get_form_factor_score(spec_text):
    text = str(spec_text).upper()
    if 'E-ATX' in text or 'EATX' in text: return 4
    if 'ATX' in text and 'MICRO' not in text and 'M-ATX' not in text: return 3
    if 'MICRO' in text or 'M-ATX' in text or 'MATX' in text: return 2
    if 'MINI' in text or 'ITX' in text: return 1
    return 3

def is_currently_market_available(name, item_type='cpu'):
    name = str(name).upper()
    if item_type == 'cpu':
        if re.search(r'\b(1151|1151V2|1150|1155|2066|1366|775|AM3|AM3\+|FM2|FM2\+|TR4)\b', name): 
            return False
        is_intel = 'INTEL' in name or re.search(r'\b(I[3579]|CORE|PENTIUM)\b', name)
        if is_intel:
            if re.search(r'\b([1-9]\d{2}|[2-9]\d{3})[A-Z]{0,2}\b', name): return False
        is_amd = 'AMD' in name or 'RYZEN' in name or 'ATHLON' in name
        if is_amd:
            if re.search(r'\b([1234]\d{3})[A-Z]*\b', name): return False
        if re.search(r'\b(CORE 2|PENTIUM|CELERON|XEON|FX-\d{4}|A\d{1,2}-\d{4}|ATHLON)\b', name): 
            return False
    elif item_type == 'gpu':
        if re.search(r'\b(GTX|GT\s*\d{3,4}|GTS|MX\d{3,4}|RTX\s*20\d{0,2}|ARC)\b', name): return False
        if re.search(r'QUADRO\s*([KMP]\d{3,4}|\d{4})\b', name) and not re.search(r'(RTX|A\d+)', name): return False
        if re.search(r'\b(RX\s*[45]\d{2,3}\b|R[579]\s*\d{3,4}|RADEON HD|VEGA|FIREPRO|WX\d{4})\b', name): return False
        if re.search(r'\b(TESLA|GRID|TITAN|P106|CMP|MINING)\b', name): return False
    return True

def load_data_to_dicts(filter_old=True):
    CSV_DIR = settings.BASE_DIR / 'csv_data'
    df_cpu = pd.read_csv(CSV_DIR / 'combined_cpu_data_strict_pc_only.csv')
    df_gpu = pd.read_csv(CSV_DIR / 'pc_gpu_data.csv')
    df_mb = pd.read_csv(CSV_DIR / 'mainboards.csv')
    df_ram = pd.read_csv(CSV_DIR / 'ram_data.csv')
    df_psu = pd.read_csv(CSV_DIR / 'powersupplies.csv')
    df_hdd = pd.read_csv(CSV_DIR / 'hdd_list_all.csv')
    df_case = pd.read_csv(CSV_DIR / 'GEAR - CASE.csv') 
    df_monitor = pd.read_csv(CSV_DIR / 'GEAR - MONITER.csv') 
    df_mouse = pd.read_csv(CSV_DIR / 'GEAR - MOUSE.csv')
    df_kb = pd.read_csv(CSV_DIR / 'keyboards.csv')
    df_headset = pd.read_csv(CSV_DIR / 'headsets.csv')
    df_mic = pd.read_csv(CSV_DIR / 'microphones.csv')
    df_cooler = pd.read_csv(CSV_DIR / 'cpucooler.csv')

    df_cpu['price'] = safe_clean(df_cpu['price_cpu'])
    df_gpu['price'] = safe_clean(df_gpu['price_gpu'])
    df_mb['price'] = safe_clean(df_mb['price_mainboards'])
    df_ram['price'] = safe_clean(df_ram['price_ram'])
    df_psu['price'] = safe_clean(df_psu['price_powersupplies'])
    df_hdd['price'] = safe_clean(df_hdd['price_hdd'])
    df_case['price'] = safe_clean(df_case['price_case'])
    df_monitor['price'] = safe_clean(df_monitor['price_moniter'])
    df_mouse['price'] = safe_clean(df_mouse['price_mouse'])
    df_kb['price'] = safe_clean(df_kb['price_keyboards'])
    df_headset['price'] = safe_clean(df_headset['price_headsets'])
    df_mic['price'] = safe_clean(df_mic['price_microphones'])
    df_cooler['price'] = safe_clean(df_cooler['store_price_cpucooler'])

    #df_cpu = df_cpu[df_cpu['price'] > 500]
    #df_gpu = df_gpu[df_gpu['price'] > 500]
    
    #df_cpu = df_cpu[df_cpu['name_cpu'].apply(lambda x: is_currently_market_available(x, 'cpu'))]
    #df_gpu = df_gpu[df_gpu['name_gpu'].apply(lambda x: is_currently_market_available(x, 'gpu'))]

    df_cpu['score'] = safe_clean(df_cpu['benchmark_cpu'])
    df_gpu['score'] = safe_clean(df_gpu['passmark_gpu'])
    df_cpu['est_tdp'] = df_cpu['score'].apply(get_cpu_tdp)
    df_gpu['est_tdp'] = df_gpu['score'].apply(get_gpu_tdp)

    df_ram['capacity_gb'] = df_ram['name__ram'].astype(str).str.upper().str.extract(r'(\d+)\s*G')[0].astype(float).fillna(8.0)
    df_psu['wattage'] = df_psu['name_powersupplies'].astype(str).str.upper().str.extract(r'(\d+)\s*(?:W|WATT)')[0].astype(float).fillna(500)
    df_cpu['clean_socket'] = df_cpu['socket_cpu'].apply(get_clean_socket)
    df_mb['clean_socket'] = df_mb['socket_mainboards'].apply(get_clean_socket)
    df_ram['clean_ddr'] = df_ram['ddr_type_ram'].astype(str).str.upper().str.extract(r'(DDR[345])')[0].fillna('UNKNOWN')

    # 🌟 จุดสำคัญ: กรองข้อมูลตามเงื่อนไข
    if filter_old:
        # ใช้สำหรับหน้า AI Recommend (กรองเฉพาะของที่ยังมีขายและราคาปกติ)
        df_cpu = df_cpu[df_cpu['price'] > 500]
        df_gpu = df_gpu[df_gpu['price'] > 500]
        df_cpu = df_cpu[df_cpu['name_cpu'].apply(lambda x: is_currently_market_available(x, 'cpu'))]
        df_gpu = df_gpu[df_gpu['name_gpu'].apply(lambda x: is_currently_market_available(x, 'gpu'))]
    else:
        # ใช้สำหรับหน้า Custom Builder (ไม่กรอง เพื่อให้เช็คสเปคคอมเก่าได้)
        # อาจจะกรองแค่ตัวที่ราคาเป็น 0 หรือข้อมูลเสียจริงๆ ออกพอ
        df_cpu = df_cpu[df_cpu['price'] >= 0]
        df_gpu = df_gpu[df_gpu['price'] >= 0]

    if 'drive_name_hdd' in df_hdd.columns: df_hdd['storage_name'] = df_hdd['drive_name_hdd']
    elif 'name_hdd' in df_hdd.columns: df_hdd['storage_name'] = df_hdd['name_hdd']
    else: df_hdd['storage_name'] = df_hdd['type_hdd']

    # 🌟 NEW: แปลงคอลัมน์ใหม่ให้เป็นตัวเลข เพื่อให้ Machine Learning นำไปประมวลผลต่อได้
    if 'core_count' in df_cpu.columns: df_cpu['core_count'] = pd.to_numeric(df_cpu['core_count'], errors='coerce').fillna(4)
    if 'thread_count' in df_cpu.columns: df_cpu['thread_count'] = pd.to_numeric(df_cpu['thread_count'], errors='coerce').fillna(4)
    if 'vram_gb' in df_gpu.columns: df_gpu['vram_gb'] = pd.to_numeric(df_gpu['vram_gb'], errors='coerce').fillna(4)
    if 'creator_weight' in df_gpu.columns: df_gpu['creator_weight'] = pd.to_numeric(df_gpu['creator_weight'], errors='coerce').fillna(1)
    if 'capacity_gb' in df_hdd.columns: df_hdd['capacity_gb_num'] = pd.to_numeric(df_hdd['capacity_gb'], errors='coerce').fillna(500)

    return (
        df_cpu.to_dict('records'), df_gpu.to_dict('records'), df_mb.to_dict('records'),
        df_ram.to_dict('records'), df_psu.to_dict('records'), df_hdd.to_dict('records'),
        df_case.to_dict('records'),
        df_monitor.to_dict('records'), df_mouse.to_dict('records'), df_kb.to_dict('records'),
        df_headset.to_dict('records'), df_mic.to_dict('records'),
        df_cooler.to_dict('records')
    )

CPU_LIST, GPU_LIST, MB_LIST, RAM_LIST, PSU_LIST, HDD_LIST, CASE_LIST, MONITOR_LIST, MOUSE_LIST, KB_LIST, HEADSET_LIST, MIC_LIST, COOLER_LIST = load_data_to_dicts()

def custom_builder_view(request):
    CPU, GPU, MB, RAM, PSU, HDD, CASE, MON, MOUSE, KB, HS, MIC, COOLER = load_data_to_dicts(filter_old=False)
    context = {
        'cpu_json': json.dumps(CPU), 'gpu_json': json.dumps(GPU), 'mb_json': json.dumps(MB),
        'ram_json': json.dumps(RAM), 'psu_json': json.dumps(PSU), 'hdd_json': json.dumps(HDD),
        'case_json': json.dumps(CASE),
    }
    return render(request, 'custom_builder.html', context)

def parse_mb_slots(spec_text):
    spec_text = str(spec_text).upper()
    ram_slots = int(re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text).group(1)) if re.search(r'(\d+)\s*[X\*]\s*DDR', spec_text) else 2
    m2_match = re.search(r'(\d+)\s*[X\*]\s*M\.2', spec_text)
    m2_slots = int(m2_match.group(1)) if m2_match else 0
    sata_slots = int(re.search(r'(\d+)\s*[X\*]\s*SATA', spec_text).group(1)) if re.search(r'(\d+)\s*[X\*]\s*SATA', spec_text) else 4
    m2_nvme = False if ('SATA ONLY' in spec_text or 'NON-NVME' in spec_text) else (m2_slots > 0)
    return ram_slots, m2_slots, sata_slots, m2_nvme

def get_safe_variety_list(items, top_n=5):
    if not items: return None
    sorted_items = sorted(items, key=lambda x: x['price'])
    return random.choice(sorted_items[:top_n])

# 🌟 NEW: อัปเกรดระบบอธิบายเหตุผลให้ดึง VRAM และ Core Count มาใช้อธิบายให้ผู้ใช้เข้าใจ
def generate_ai_reasoning(b, usage_type, rank):
    cpu_score = b['cpu']['score']
    gpu_score = b['gpu']['score']
    total_price = b['total_price']
    ram_gb = b['ram']['capacity_gb'] * b['ram_qty']
    c_cores = b['cpu'].get('core_count', 4)
    g_vram = b['gpu'].get('vram_gb', 4)
    
    reasons = []
    gpu_ratio = gpu_score / (cpu_score + 1)
    
    if usage_type == 'gaming':
        if g_vram >= 12: reasons.append(f"✅ GPU มี VRAM สูงถึง {int(g_vram)}GB รองรับการปรับภาพระดับ Ultra และกราฟิกยุคใหม่ได้ดีเยี่ยม")
        elif g_vram >= 8: reasons.append(f"✅ VRAM การ์ดจอ {int(g_vram)}GB อยู่ในเกณฑ์มาตรฐานที่เล่นเกมปัจจุบันได้ลื่นไหล")
        
        if gpu_ratio > 3.0: reasons.append("⚠️ GPU ทรงพลังมาก แต่อาจมีคอขวดที่ CPU เล็กน้อย (CPU รีดเฟรมเรทได้ไม่สุดในบางเกม)")
        elif gpu_ratio < 1.2: reasons.append("⚠️ CPU แรงเหลือเฟือ แต่ GPU อาจเป็นคอขวดสำหรับการปรับภาพสุดในเกม AAA")
        else: reasons.append("✅ ความสมดุลระหว่าง CPU และ GPU ยอดเยี่ยม (Bottleneck ต่ำมาก ประสิทธิภาพเกมมิ่งเสถียร)")
            
    elif usage_type == 'creator':
        if c_cores >= 10: reasons.append(f"✅ CPU มีจำนวนคอร์มหาศาล ({int(c_cores)} Cores) ช่วยเรนเดอร์งานหนักและ Export วิดีโอได้ไวมาก")
        if g_vram >= 12: reasons.append(f"✅ VRAM การ์ดจอ {int(g_vram)}GB เพียงพอสำหรับการพรีวิวโปรเจกต์ 3D หรือวิดีโอ 4K ขนาดใหญ่โดยไม่กระตุก")
        if ram_gb >= 32: reasons.append(f"✅ RAM ขนาดใหญ่ ({int(ram_gb)}GB) รองรับโปรแกรมตัดต่อ/กราฟิกได้ลื่นไหล ไม่ค้าง")
        
    else: # work
        storage_cap = b['storage'].get('capacity_gb_num', 500)
        if storage_cap >= 1000: reasons.append(f"✅ พื้นที่เก็บข้อมูลขนาดใหญ่ ({int(storage_cap/1000)}TB) เพียงพอสำหรับเซฟเอกสารและโปรเจกต์งานมหาศาล")
        if c_cores >= 6: reasons.append(f"✅ CPU {int(c_cores)} Cores จัดการการเปิดหลายโปรแกรมพร้อมกัน (Multitasking) ได้เป็นอย่างดี")
        if cpu_score > 20000: reasons.append("✅ ประสิทธิภาพ CPU สูงมาก รองรับการรันโปรแกรมเฉพาะทาง หรือเปิดไฟล์ Excel ข้อมูลเยอะๆ ได้รวดเร็ว")
    
    value_index = (cpu_score + gpu_score) / total_price if total_price > 0 else 0
    if value_index > 1.6: reasons.append("🔥 ความคุ้มค่า (Performance/Price) สูงเป็นพิเศษ ได้สเปคแรงในราคาที่ถูกกว่าค่าเฉลี่ยตลาด")
    elif value_index < 0.9: reasons.append("💡 เน้นใช้อุปกรณ์พรีเมียมหรือเทคโนโลยีใหม่ (อาจจ่ายแพงขึ้นเล็กน้อยแลกกับฟีเจอร์ที่ทันสมัย)")
        
    if "Pro" in rank or "Great" in rank: reasons.insert(0, "✨ AI แนะนำสเปคนี้เพราะจัดสรรทรัพยากรตรงกับความต้องการของสายงานที่สุด")
        
    return reasons

def recommend_pc(request):
    recommended_builds = []
    error = None
    
    if request.method == 'POST':
        form = SpecRequestForm(request.POST)
        if form.is_valid():
            total_budget = float(form.cleaned_data['budget'])
            usage_type = form.cleaned_data.get('usage_type', 'gaming')
            include_monitor = form.cleaned_data.get('include_monitor', False)
            include_gear = form.cleaned_data.get('include_gear', False)
            
            random.seed(int(total_budget))
            
            gear_allowance = 0
            if include_monitor: gear_allowance += max(3000, total_budget * 0.15) 
            if include_gear: gear_allowance += max(1500, total_budget * 0.08)

            pc_budget = total_budget - gear_allowance

            if pc_budget < 4000:
                return render(request, 'showpages.html', {'form': form, 'error': "งบสำหรับเครื่อง PC น้อยเกินไป (หลังจากหักงบอุปกรณ์เสริม) แนะนำให้เพิ่มงบ"})

            if usage_type == 'gaming':
                gpu_ratio, cpu_ratio = 0.48, 0.20 
                if pc_budget < 20000: gpu_ratio, cpu_ratio = 0.40, 0.25
            elif usage_type == 'creator':
                if pc_budget >= 60000: gpu_ratio, cpu_ratio = 0.55, 0.25 
                else: gpu_ratio, cpu_ratio = 0.40, 0.30 
                if pc_budget < 20000: gpu_ratio, cpu_ratio = 0.30, 0.40
            else: 
                gpu_ratio, cpu_ratio = 0.05, 0.50 

            gpu_b = pc_budget * gpu_ratio
            cpu_b = pc_budget * cpu_ratio

            valid_mb_sockets = set([m['clean_socket'] for m in MB_LIST])
            filtered_cpu_list = [c for c in CPU_LIST if c['clean_socket'] in valid_mb_sockets]
            filtered_gpu_list = GPU_LIST

            if total_budget >= 18000:
                cpu_brand_pref = form.cleaned_data.get('cpu_brand', 'all')
                gpu_brand_pref = form.cleaned_data.get('gpu_brand', 'all')

                if cpu_brand_pref == 'intel': filtered_cpu_list = [c for c in filtered_cpu_list if 'INTEL' in str(c['name_cpu']).upper()]
                elif cpu_brand_pref == 'amd': filtered_cpu_list = [c for c in filtered_cpu_list if 'RYZEN' in str(c['name_cpu']).upper() or 'AMD' in str(c['name_cpu']).upper() or 'THREADRIPPER' in str(c['name_cpu']).upper()]

                if gpu_brand_pref == 'nvidia': filtered_gpu_list = [g for g in filtered_gpu_list if re.search(r'(RTX|GTX|QUADRO|TESLA|ADA|TITAN|A\d+)', str(g['name_gpu']).upper())]
                elif gpu_brand_pref == 'amd': filtered_gpu_list = [g for g in filtered_gpu_list if re.search(r'(RX\s|RADEON|PRO\b)', str(g['name_gpu']).upper())]
            
            base_filtered_gpu = filtered_gpu_list 

            if usage_type == 'creator':
                creator_gpu_kw = r'(RTX\s*A\d+|ADA|RADEON\s*PRO\s*W\d+|RTX\s*[345]0\d{2}|RX\s*[67]\d{3}|A\d{3,4}|QUADRO)'
                filtered_gpu_list = [g for g in base_filtered_gpu if re.search(creator_gpu_kw, str(g['name_gpu']), re.I) and 'ARC' not in str(g['name_gpu']).upper()]
                if not filtered_gpu_list: filtered_gpu_list = base_filtered_gpu 
                temp_gpu = sorted([g for g in filtered_gpu_list if g['price'] <= gpu_b], key=lambda x: x['score'] * x.get('creator_weight', 1), reverse=True)
            elif usage_type == 'work':
                banned_work_kw = r'(TESLA|QUADRO|RTX\s*A\d|RTX\s*PRO|ADA|RADEON\s*PRO|FIREPRO|GRID|TITAN)'
                filtered_gpu_list = [g for g in base_filtered_gpu if 3000 <= g['price'] <= 25000 and not re.search(banned_work_kw, str(g['name_gpu']), re.I)]
                temp_gpu = sorted([g for g in filtered_gpu_list if g['price'] <= gpu_b], key=lambda x: x['score'], reverse=True)
            else: 
                banned_gaming_kw = r'(TESLA|QUADRO|RTX\s*A\d|PRO\b|ADA|FIREPRO|GRID|TITAN|ARC)'
                filtered_gpu_list = [g for g in base_filtered_gpu if not re.search(banned_gaming_kw, str(g['name_gpu']), re.I)]
                temp_gpu = sorted([g for g in filtered_gpu_list if g['price'] <= gpu_b], key=lambda x: x['score'], reverse=True)

            gpu_pool = []
            seen_gpu = set()
            for g in temp_gpu:
                base_name = re.sub(r'\(.*?\)', '', str(g['name_gpu'])).strip().upper()
                if base_name not in seen_gpu:
                    seen_gpu.add(base_name)
                    gpu_pool.append(g)
                if len(gpu_pool) >= 20: break 

            temp_cpu = sorted([c for c in filtered_cpu_list if c['price'] <= cpu_b], key=lambda x: x['score'], reverse=True)
            cpu_pool = []
            seen_cpu = set()
            for c in temp_cpu:
                base_name = re.sub(r'\(.*?\)', '', str(c['name_cpu'])).strip().upper()
                if base_name not in seen_cpu:
                    seen_cpu.add(base_name)
                    cpu_pool.append(c)
                if len(cpu_pool) >= 20: break

            if not gpu_pool: gpu_pool = sorted(filtered_gpu_list, key=lambda x: x['price'])[:8]
            if not cpu_pool: cpu_pool = sorted(filtered_cpu_list, key=lambda x: x['price'])[:8]

            onboard_candidates = []
            for c in filtered_cpu_list:
                c_name = str(c['name_cpu']).upper()
                if 'INTEL' in c_name and re.search(r'F\b', c_name): continue
                if 'RYZEN' in c_name and 'AM4' in str(c.get('socket_cpu', '')).upper() and not re.search(r'G\b', c_name): continue
                if 'RYZEN' in c_name and 'AM5' in str(c.get('socket_cpu', '')).upper() and re.search(r'F\b', c_name): continue
                onboard_candidates.append(c)
            
            dummy_gpu = {'name_gpu': 'Onboard Graphics', 'price': 0, 'score': 2500, 'est_tdp': 0, 'vram_gb': 2} # เพิ่ม vram_gb ให้ Onboard กันบั๊ก

            if usage_type == 'work' or pc_budget < 25000:
                max_cpu_allowance = cpu_b * 1.2 if pc_budget >= 20000 else cpu_b
                valid_onboards = [c for c in onboard_candidates if c['price'] <= max(cpu_b * 1.5, 4000)]

                if not valid_onboards:
                    valid_onboards = sorted(onboard_candidates, key=lambda x: x['price'])[:8]
                else:
                    top_performance = sorted(valid_onboards, key=lambda x: x['score'], reverse=True)[:8]
                    top_cheap = sorted(valid_onboards, key=lambda x: x['price'])[:8]
                    combined_onboards = []
                    seen_onboard = set()
                    for c in top_performance + top_cheap:
                        if c['name_cpu'] not in seen_onboard:
                            seen_onboard.add(c['name_cpu'])
                            combined_onboards.append(c)
                    valid_onboards = combined_onboards
                cpu_pool = valid_onboards + cpu_pool
                
            force_onboard = False
            if usage_type == 'work' and pc_budget < 18000: force_onboard = True
            elif usage_type == 'creator' and pc_budget < 15000: force_onboard = True
            elif usage_type == 'gaming' and pc_budget < 12000: force_onboard = True

            if force_onboard: gpu_pool = [dummy_gpu]
            else: gpu_pool.append(dummy_gpu)

            if usage_type == 'work':
                if pc_budget >= 80000: target_cap = r'(4\s*TB|4000\s*GB|8\s*TB|8000\s*GB)'
                elif pc_budget >= 40000: target_cap = r'(2\s*TB|2000\s*GB)'
                elif pc_budget >= 20000: target_cap = r'(1\s*TB|1000\s*GB)'
                else: target_cap = r'(500\s*GB|512\s*GB)' 
            elif usage_type == 'creator':   
                if pc_budget >= 80000: target_cap = r'(4\s*TB|4000\s*GB)'
                elif pc_budget >= 50000: target_cap = r'(2\s*TB|2000\s*GB)'
                elif pc_budget >= 25000: target_cap = r'(1\s*TB|1000\s*GB)'
                else: target_cap = r'(500\s*GB|512\s*GB)'
            else: 
                if pc_budget >= 50000: target_cap = r'(2\s*TB|2000\s*GB)'
                elif pc_budget >= 34000: target_cap = r'(1\s*TB|1000\s*GB)'
                elif pc_budget >= 20000: target_cap = r'(500\s*GB|512\s*GB)'
                else: target_cap = r'(240\s*GB|250\s*GB|256\s*GB|500\s*GB|512\s*GB)'

            ready_nvme = [s for s in HDD_LIST if re.search(r'(NVMe|PCIe)', str(s['type_hdd']), re.I)]
            ready_sata = [s for s in HDD_LIST if re.search(r'SATA', str(s['type_hdd']), re.I)]
            base_case = [c for c in CASE_LIST if not re.search(r'(CABLE|FAN|HUB|BRACKET|STRIP)', str(c.get('model_name_case', '')), re.I)]
            if pc_budget >= 50000: pool_case = [c for c in base_case if c['price'] >= 2000]
            elif pc_budget >= 30000: pool_case = [c for c in base_case if c['price'] >= 1200]
            else: pool_case = base_case

            raw_builds = []
            for cpu in cpu_pool:
                for gpu in gpu_pool:
                    c_name = str(cpu['name_cpu']).upper()
                    is_no_igpu = False
                    if 'INTEL' in c_name and re.search(r'F\b', c_name): is_no_igpu = True
                    if 'RYZEN' in c_name and 'AM4' in str(cpu.get('socket_cpu', '')).upper() and not re.search(r'G\b', c_name): is_no_igpu = True
                    if 'RYZEN' in c_name and 'AM5' in str(cpu.get('socket_cpu', '')).upper() and re.search(r'F\b', c_name): is_no_igpu = True
                    
                    if gpu['name_gpu'] == 'Onboard Graphics' and is_no_igpu: continue 

                    max_core_ratio = 0.85 if pc_budget < 20000 else 0.70
                    if pc_budget >= 15000 and (cpu['price'] + gpu['price']) > (pc_budget * max_core_ratio): continue

                    mb_pool = [m for m in MB_LIST if m['clean_socket'] == cpu['clean_socket']]
                    mb_pool = [m for m in mb_pool if is_tier_compatible(cpu['name_cpu'], m['name_mainboards'])]
                    if not mb_pool: continue
                    mb = get_safe_variety_list(mb_pool, top_n=4)

                    mb_spec = str(mb.get('specifications_mainboards', '')).upper()
                    mb_name = str(mb.get('name_mainboards', '')).upper()
                    ram_slots, m2_slots, sata_slots, m2_nvme = parse_mb_slots(mb_spec)
                    mb_ddr_match = re.search(r'(DDR[345])', mb_spec + " " + mb_name)
                    mb_ddr_type = mb_ddr_match.group(1) if mb_ddr_match else "DDR4"
                    ram_matched = [r for r in RAM_LIST if r['clean_ddr'] == mb_ddr_type]
                    
                    if pc_budget >= 35000:
                        if mb_ddr_type == "DDR5":
                            prem_ram = [r for r in ram_matched if re.search(r'(5200|5600|6000|6400|7200|8000)', str(r.get('name__ram', '')))]
                            if prem_ram: ram_matched = prem_ram
                        else:
                            prem_ram = [r for r in ram_matched if re.search(r'(3200|3600|4000)', str(r.get('name__ram', '')))]
                            if prem_ram: ram_matched = prem_ram

                    if not ram_matched: continue

                    ram_qty = 1
                    if pc_budget >= 50000:
                        ram_32 = [r for r in ram_matched if r['capacity_gb'] >= 32]
                        ram_16 = [r for r in ram_matched if r['capacity_gb'] >= 16]
                        if ram_32 and ram_slots >= 2:
                            ram = get_safe_variety_list(ram_32, 5)
                            ram_qty = ram_slots if pc_budget >= 80000 else min(ram_slots, 2)
                        elif ram_16 and ram_slots >= 4:
                            ram = get_safe_variety_list(ram_16, 5)
                            ram_qty = min(ram_slots, 4)
                        elif ram_32:
                            ram = get_safe_variety_list(ram_32, 5)
                            ram_qty = 1
                        elif ram_16 and ram_slots >= 2:
                            ram = get_safe_variety_list(ram_16, 5)
                            ram_qty = min(ram_slots, 2)
                        else:
                            ram = get_safe_variety_list(ram_matched, 5)
                            ram_qty = 1
                    elif pc_budget >= 30000:
                        ram_16 = [r for r in ram_matched if r['capacity_gb'] >= 16]
                        if ram_16 and ram_slots >= 2:
                            ram = get_safe_variety_list(ram_16, 5)
                            ram_qty = 2
                        elif ram_16: ram = get_safe_variety_list(ram_16, 5); ram_qty = 1
                        else: ram = get_safe_variety_list(ram_matched, 5); ram_qty = 1
                    elif pc_budget >= 15000:
                        ram_8 = [r for r in ram_matched if r['capacity_gb'] == 8]
                        if ram_8 and ram_slots >= 2:
                            ram = get_safe_variety_list(ram_8, 5)
                            ram_qty = 2
                        else:
                            ram_16 = [r for r in ram_matched if r['capacity_gb'] >= 16]
                            if ram_16: ram = get_safe_variety_list(ram_16, 5)
                            else: ram = get_safe_variety_list(ram_matched, 5)
                    else:
                        ram_8 = [r for r in ram_matched if r['capacity_gb'] <= 8]
                        if ram_8: ram = get_safe_variety_list(ram_8, 5)
                        else: ram = get_safe_variety_list(ram_matched, 5)

                    is_kit = bool(re.search(r'(X2|\*2|KIT)', str(ram.get('name__ram', '')).upper()))
                    sticks_per_item = 2 if is_kit else 1
                    ram_qty = min(ram_qty, ram_slots // sticks_per_item)
                    ram_qty = max(1, ram_qty)
                    ram_tdp = 5 * (ram_qty * sticks_per_item)

                    storage_list, storage_price, storage_tdp = [], 0, 0
                    if m2_slots > 0 and m2_nvme and ready_nvme:
                        cap_matched = [s for s in ready_nvme if re.search(target_cap, str(s['storage_name']), re.I)]
                        if cap_matched: st = get_safe_variety_list(cap_matched, 5)
                        else: st = sorted(ready_nvme, key=lambda x: x['price'], reverse=True)[0] if pc_budget >= 50000 else sorted(ready_nvme, key=lambda x: x['price'])[0]
                        storage_tdp = 8
                    elif (sata_slots > 0 or m2_slots > 0) and ready_sata:
                        cap_matched = [s for s in ready_sata if re.search(target_cap, str(s['storage_name']), re.I)]
                        if cap_matched: st = get_safe_variety_list(cap_matched, 5)
                        else: st = sorted(ready_sata, key=lambda x: x['price'], reverse=True)[0] if pc_budget >= 50000 else sorted(ready_sata, key=lambda x: x['price'])[0]
                        storage_tdp = 5
                    else:
                        st = get_safe_variety_list(HDD_LIST, 5)
                        storage_tdp = 5
                        
                    if pc_budget >= 100000 and ((storage_tdp == 8 and m2_slots >= 4) or (storage_tdp == 5 and sata_slots >= 4)):
                        storage_list.append(f"{st['storage_name']} <span style='color: #0d6efd; font-weight: bold;'>x4</span>")
                        storage_price = st['price'] * 4; storage_tdp *= 4
                    elif pc_budget >= 75000 and ((storage_tdp == 8 and m2_slots >= 3) or (storage_tdp == 5 and sata_slots >= 3)):
                        storage_list.append(f"{st['storage_name']} <span style='color: #0d6efd; font-weight: bold;'>x3</span>")
                        storage_price = st['price'] * 3; storage_tdp *= 3
                    elif pc_budget >= 50000 and ((storage_tdp == 8 and m2_slots >= 2) or (storage_tdp == 5 and sata_slots >= 2)):
                        storage_list.append(f"{st['storage_name']} <span style='color: #0d6efd; font-weight: bold;'>x2</span>")
                        storage_price = st['price'] * 2; storage_tdp *= 2
                    else:
                        storage_list.append(str(st['storage_name']))
                        storage_price = st['price']

                    actual_draw = cpu['est_tdp'] + gpu['est_tdp'] + ram_tdp + storage_tdp + 50
                    if gpu['est_tdp'] >= 300: safety_margin = 350  
                    elif gpu['est_tdp'] >= 200: safety_margin = 200  
                    elif gpu['est_tdp'] >= 130: safety_margin = 150  
                    else: safety_margin = 100  

                    req_wattage = actual_draw + safety_margin
                    req_wattage = req_wattage * 1.15
                    req_wattage = math.ceil(req_wattage / 50.0) * 50

                    if pc_budget >= 70000 and req_wattage < 1000: req_wattage = 1000
                    elif pc_budget >= 50000 and req_wattage < 850: req_wattage = 850
                    elif pc_budget >= 35000 and req_wattage < 750: req_wattage = 750
                    elif pc_budget >= 25000 and req_wattage < 650: req_wattage = 650
                        
                    psu_pool = [p for p in PSU_LIST if p['wattage'] >= req_wattage and 'SFX' not in str(p.get('name_powersupplies', '')).upper()]
                    if not psu_pool: 
                        if PSU_LIST: max_avail_w = max([p['wattage'] for p in PSU_LIST]); psu_pool = [p for p in PSU_LIST if p['wattage'] == max_avail_w]
                        else: psu_pool = PSU_LIST
                    
                    if pc_budget >= 60000: safe_psu_pool = [p for p in psu_pool if p['price'] >= 3500]
                    elif pc_budget >= 20000: safe_psu_pool = [p for p in psu_pool if p['price'] >= max(pc_budget * 0.05, req_wattage * 2.0)]
                    else: safe_psu_pool = psu_pool
                    
                    if safe_psu_pool: psu = get_safe_variety_list(safe_psu_pool, 5)
                    else: psu = sorted(psu_pool, key=lambda x: x['price'], reverse=True)[0]

                    temp_cases = pool_case
                    if form.cleaned_data.get('case_atx') or form.cleaned_data.get('case_matx') or form.cleaned_data.get('case_itx'):
                        filtered_cases = []
                        for c in temp_cases:
                            ff = str(c.get('form_factor_case', '')).upper()
                            if form.cleaned_data.get('case_atx') and ('ATX' in ff or 'E-ATX' in ff) and 'MICRO' not in ff and 'M-ATX' not in ff: filtered_cases.append(c)
                            elif form.cleaned_data.get('case_matx') and ('MICRO' in ff or 'M-ATX' in ff or 'MATX' in ff): filtered_cases.append(c)
                            elif form.cleaned_data.get('case_itx') and ('MINI' in ff or 'ITX' in ff): filtered_cases.append(c)
                        if filtered_cases: temp_cases = filtered_cases

                    mb_size_score = get_form_factor_score(mb_spec + " " + mb_name)
                    valid_cases = [c for c in temp_cases if get_form_factor_score(str(c.get('specifications_case', '')) + " " + str(c.get('model_name_case', ''))) >= mb_size_score]
                    if not valid_cases: valid_cases = temp_cases
                    case = random.choice(valid_cases)

                    cooler_pool_f = [c for c in COOLER_LIST if c['price'] > 0]
                    if form.cleaned_data.get('cool_liquid'): cooler_pool_f = [c for c in cooler_pool_f if re.search(r'(LIQUID|WATER|AIO)', str(c.get('product_name_cpucooler','')) + str(c.get('details_cpucooler','')), re.I)]
                    elif form.cleaned_data.get('cool_air'): cooler_pool_f = [c for c in cooler_pool_f if not re.search(r'(LIQUID|WATER|AIO)', str(c.get('product_name_cpucooler','')) + str(c.get('details_cpucooler','')), re.I)]
                    sel_cooler = random.choice(cooler_pool_f) if cooler_pool_f else None

                    pc_total_price = cpu['price'] + gpu['price'] + mb['price'] + (ram['price'] * ram_qty) + storage_price + psu['price'] + case['price']
                    if sel_cooler: pc_total_price += sel_cooler['price']
                    
                    build_gears = {'cooler': sel_cooler}
                    actual_gear_price = 0

                    if include_monitor:
                        mon_pool = [m for m in MONITOR_LIST if m['price'] > 0]
                        if form.cleaned_data.get('mon_144'): mon_pool = [m for m in mon_pool if int(re.search(r'\d+', str(m.get('refresh_rate_moniter','0'))).group()) >= 144]
                        if form.cleaned_data.get('mon_ips'): mon_pool = [m for m in mon_pool if re.search(r'(IPS|OLED)', str(m.get('panel_type_moniter','')), re.I)]
                        if form.cleaned_data.get('mon_4k'): mon_pool = [m for m in mon_pool if re.search(r'(2560|3840)', str(m.get('resolution_moniter','')))]
                        mon_target = [m for m in mon_pool if m['price'] <= (gear_allowance * 0.7 if include_gear else gear_allowance)]
                        sel_monitor = random.choice(mon_target[-10:]) if mon_target else (mon_pool[-1] if mon_pool else None)
                        if sel_monitor: build_gears['monitor'] = sel_monitor; actual_gear_price += sel_monitor['price']

                    if include_gear:
                        m_pool = [m for m in MOUSE_LIST if m['price'] > 0]
                        k_pool = [k for k in KB_LIST if k['price'] > 0]
                        h_pool = [h for h in HEADSET_LIST if h['price'] > 0]
                        mic_pool = [mc for mc in MIC_LIST if mc['price'] > 0]

                        if form.cleaned_data.get('mouse_gaming'): m_pool = [m for m in m_pool if re.search(r'GAMING', str(m.get('type_mouse','')) + str(m.get('model_name_mouse','')), re.I)]
                        if form.cleaned_data.get('mouse_wireless'): m_pool = [m for m in m_pool if re.search(r'WIRELESS|BLUETOOTH', str(m.get('type_mouse','')) + str(m.get('connection_type_mouse','')), re.I)]
                        if form.cleaned_data.get('kb_mech'): k_pool = [k for k in k_pool if re.search(r'MECHANICAL', str(k.get('name_keyboards','')) + str(k.get('specifications_keyboards','')), re.I)]
                        if form.cleaned_data.get('kb_wireless'): k_pool = [k for k in k_pool if re.search(r'WIRELESS|BLUETOOTH', str(k.get('name_keyboards','')) + str(k.get('specifications_keyboards','')), re.I)]
                        if form.cleaned_data.get('headset_71'): h_pool = [h for h in h_pool if '7.1' in str(h.get('name_headsets','')) or '7.1' in str(h.get('specifications_headsets',''))]
                        if form.cleaned_data.get('headset_wire'): h_pool = [h for h in h_pool if re.search(r'WIRELESS|BLUETOOTH', str(h.get('name_headsets','')) + str(h.get('specifications_headsets','')), re.I)]
                        if form.cleaned_data.get('mic_usb'): mic_pool = [mc for mc in mic_pool if re.search(r'USB|TYPE-C', str(mc.get('name_microphones','')) + str(mc.get('specifications_microphones','')), re.I)]
                        if form.cleaned_data.get('mic_studio'): mic_pool = [mc for mc in mic_pool if re.search(r'CONDENSER|STUDIO', str(mc.get('name_microphones','')) + str(mc.get('specifications_microphones','')), re.I)]

                        sel_mouse = get_safe_variety_list(m_pool, 10) if m_pool else None
                        sel_kb = get_safe_variety_list(k_pool, 10) if k_pool else None
                        sel_hs = get_safe_variety_list(h_pool, 10) if h_pool else None
                        sel_mic = get_safe_variety_list(mic_pool, 10) if mic_pool else None

                        build_gears['mouse'] = sel_mouse; build_gears['keyboard'] = sel_kb; build_gears['headset'] = sel_hs; build_gears['mic'] = sel_mic
                        actual_gear_price += sum([x['price'] for x in [sel_mouse, sel_kb, sel_hs, sel_mic] if x])

                    final_total_price = pc_total_price + actual_gear_price

                    max_allowed = total_budget * 1 if total_budget < 25000 else total_budget
                    min_allowed = total_budget * 0.40 if total_budget < 20000 else total_budget * 0.55
                    if total_budget > 150000: min_allowed = 100000; max_allowed = total_budget
                    if usage_type == 'work' or gpu['name_gpu'] == 'Onboard Graphics': min_allowed = 0 
                    
                    if min_allowed <= final_total_price <= max_allowed:
                        raw_builds.append({
                            'cpu': cpu, 'gpu': gpu, 'mb': mb, 'ram': ram, 'ram_qty': ram_qty, 'storage': st,
                            'ssd_name': " + ".join(storage_list), 'psu': psu, 'req_w': req_wattage, 'case': case,
                            'gears': build_gears, 'total_price': final_total_price, 'score': cpu['score'] + gpu['score']
                        })

            def get_ml_usage_score(b):
                c_score = b['cpu']['score']
                g_score = b['gpu']['score']
                total_p = b['total_price']
                
                curr_model, scale_c, scale_g = model, default_scaler_cpu, default_scaler_gpu
                if usage_type == 'creator' and ws_model:
                    curr_model, scale_c, scale_g = ws_model, ws_scaler_cpu, ws_scaler_gpu
                elif usage_type == 'work' and office_model:
                    curr_model, scale_c, scale_g = office_model, office_scaler_cpu, office_scaler_gpu

                ml_score = 0
                if curr_model and scale_c and scale_g:
                    try:
                        s_cpu = scale_c.transform([[c_score]])[0][0]
                        s_gpu = scale_g.transform([[g_score]])[0][0]
                        input_df = pd.DataFrame({'cpu_scaled': [s_cpu], 'gpu_scaled': [s_gpu], 'total_price': [total_p]})
                        
                        classes = list(curr_model.classes_)
                        probs = curr_model.predict_proba(input_df)[0]
                        
                        for idx, cls_name in enumerate(classes):
                            cls_name_str = str(cls_name).upper()
                            if 'PRO' in cls_name_str: ml_score += probs[idx] * 100
                            elif 'GREAT' in cls_name_str: ml_score += probs[idx] * 80
                            elif 'GOOD' in cls_name_str: ml_score += probs[idx] * 60
                            elif 'FAIR' in cls_name_str: ml_score += probs[idx] * 40
                            else: ml_score += probs[idx] * 20
                    except:
                        ratio = min(c_score, g_score) / max(c_score, g_score) if max(c_score, g_score) > 0 else 0
                        ml_score = ratio * 100
                
                budget_proximity_score = max(0, 100 - (abs(total_budget - total_p) / total_budget * 100))
                
                # ดึง Feature ใหม่ของอุปกรณ์
                c_cores = b['cpu'].get('core_count', 4)
                c_threads = b['cpu'].get('thread_count', 4)
                g_vram = b['gpu'].get('vram_gb', 4)
                g_creator = b['gpu'].get('creator_weight', 1)
                storage_cap = b['storage'].get('capacity_gb_num', 500)
                
                if usage_type == 'creator':
                    # สายครีเอเตอร์ โบนัสสำหรับ Thread เยอะ, VRAM เยอะ และการ์ดสายตรง
                    core_bonus = min(15, (c_threads / 16) * 15) 
                    vram_bonus = min(15, (g_vram / 16) * 15)
                    # เพิ่มพลังโบนัสให้การ์ดทำงานแบบก้าวกระโดด (จาก * 5 เป็น * 30)
                    weight_bonus = (g_creator - 1) * 30 
                    
                    b['priority_score'] = (ml_score * 0.35) + (budget_proximity_score * 0.15) + core_bonus + vram_bonus + weight_bonus
                    
                elif usage_type == 'work':
                    # สายออฟฟิศ โบนัสสำหรับความจุเยอะ และ CPU ที่เปิดได้หลายโปรแกรมพร้อมกัน
                    storage_bonus = min(20, (storage_cap / 1000) * 15)
                    multitask_bonus = min(15, (c_cores / 8) * 10)
                    b['priority_score'] = (ml_score * 0.55) + (budget_proximity_score * 0.15) + storage_bonus + multitask_bonus
                    
                else: 
                    # สายเกมมิ่ง โบนัส VRAM ถ้างบสูง ป้องกัน Texture คอขวด
                    vram_bonus = 0
                    if total_budget >= 40000 and g_vram >= 12: vram_bonus = 15
                    elif total_budget >= 25000 and g_vram >= 8: vram_bonus = 10
                    core_penalty = -10 if c_cores < 4 else 0
                    b['priority_score'] = (ml_score * 0.60) + (budget_proximity_score * 0.15) + vram_bonus + core_penalty
                    
                return b['priority_score']
            
            raw_builds.sort(key=get_ml_usage_score, reverse=True)
            
            final_builds = []
            seen_combos = set()
            cpu_usage_count = {}

            for b in raw_builds:
                c_name = b['cpu']['name_cpu']; g_name = b['gpu']['name_gpu']
                if g_name == 'Onboard Graphics': combo = f"{c_name}_{g_name}_{b['mb']['name_mainboards']}"
                else: combo = f"{c_name}_{g_name}"
                
                if combo not in seen_combos and cpu_usage_count.get(c_name, 0) < 4:
                    final_builds.append(b)
                    seen_combos.add(combo)
                    cpu_usage_count[c_name] = cpu_usage_count.get(c_name, 0) + 1
                if len(final_builds) >= 12: break
                
            for i, b in enumerate(final_builds):
                ai_global_rank = "N/A"
                current_model, current_scaler_cpu, current_scaler_gpu = model, default_scaler_cpu, default_scaler_gpu

                if usage_type == 'creator' and ws_model: current_model, current_scaler_cpu, current_scaler_gpu = ws_model, ws_scaler_cpu, ws_scaler_gpu
                elif usage_type == 'work' and office_model: current_model, current_scaler_cpu, current_scaler_gpu = office_model, office_scaler_cpu, office_scaler_gpu

                if current_model and current_scaler_cpu and current_scaler_gpu:
                    try:
                        s_cpu = current_scaler_cpu.transform([[b['cpu']['score']]])[0][0]
                        s_gpu = current_scaler_gpu.transform([[b['gpu']['score']]])[0][0]
                        input_df = pd.DataFrame({'cpu_scaled': [s_cpu], 'gpu_scaled': [s_gpu], 'total_price': [b['total_price']]})
                        ai_global_rank = current_model.predict(input_df)[0]
                    except: pass
                
                if i == 0: relative_rank = "Pro (เทพสุดในงบ)"
                elif i <= 2: relative_rank = "Great (ดีมาก / คุ้มค่า)"
                elif i <= 5: relative_rank = "Good (ดี / มาตรฐาน)"
                elif i <= 8: relative_rank = "Fair (พอใช้ / ถูๆไถๆ)"
                else: relative_rank = "Basic (ทางเลือกเสริม)"

                b['rank'] = relative_rank
                b['scale'] = f"ระบบประเมินสเปคนี้อยู่ระดับ : {ai_global_rank}"
                c_score = b['cpu']['score']; g_score = b['gpu']['score']
                total_p = b['total_price']; ram_cap = b['ram']['capacity_gb'] * b['ram_qty']

                gaming_score = min(100, (g_score / 35000) * 100) 
                work_score = min(100, (c_score / 45000) * 100)   
                ratio = min(c_score, g_score) / max(c_score, g_score) if max(c_score, g_score) > 0 else 0
                
                if usage_type == 'work': balance_score = min(100, (c_score / 15000) * 100)
                elif total_p > 60000: balance_score = min(100, (ratio / 0.70) * 100) 
                else: balance_score = min(100, (ratio / 0.85) * 100)
                
                raw_value = (c_score + g_score) / total_p if total_p > 0 else 0
                value_score = min(100, (raw_value / 2.2) * 100)

                b['stats'] = {'gaming': int(gaming_score), 'work': int(work_score), 'balance': int(balance_score), 'value': int(value_score)}
                b['ai_reasoning'] = generate_ai_reasoning(b, usage_type, relative_rank)

            recommended_builds = final_builds
            if not recommended_builds: error = "ไม่สามารถจัดสเปครวมอุปกรณ์ในงบที่ระบุได้ (กรุณาลองเพิ่มงบ หรือปรับงบใหม่อีกครั้ง)"

    else:
        form = SpecRequestForm()
        
    context = {
        'form': form, 'builds': recommended_builds, 'error': error,
        'all_cases': [c for c in CASE_LIST if c['price'] > 0], 
        'all_coolers': [c for c in COOLER_LIST if c['price'] > 0],
        'all_monitors': [m for m in MONITOR_LIST if m['price'] > 0],
        'all_mice': [m for m in MOUSE_LIST if m['price'] > 0],
        'all_keyboards': [k for k in KB_LIST if k['price'] > 0],
        'all_headsets': [h for h in HEADSET_LIST if h['price'] > 0],
        'all_mics': [m for m in MIC_LIST if m['price'] > 0],
    }
    return render(request, 'showpages.html', context)

# =========================================================
# 🌟 ระบบ API สำหรับให้หน้า Custom Builder เรียกใช้ ML Model
# =========================================================
def api_analyze_build(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            cpu_score = float(data.get('cpu_score', 0))
            gpu_score = float(data.get('gpu_score', 0))
            total_price = float(data.get('total_price', 0))
            usage_type = data.get('usage_type', 'gaming') 

            model_name = "pc_spec_classifier9.pkl"
            current_model, current_scaler_cpu, current_scaler_gpu = model, default_scaler_cpu, default_scaler_gpu
            
            if usage_type == 'creator' and ws_model:
                current_model, current_scaler_cpu, current_scaler_gpu = ws_model, ws_scaler_cpu, ws_scaler_gpu
                model_name = "workstation_spec_classifier1.pkl"
            elif usage_type == 'work' and office_model:
                current_model, current_scaler_cpu, current_scaler_gpu = office_model, office_scaler_cpu, office_scaler_gpu
                model_name = "office_spec_classifier.pkl"

            ai_rank = "N/A"
            s_cpu, s_gpu = 0, 0

            if current_model and current_scaler_cpu and current_scaler_gpu:
                s_cpu = float(current_scaler_cpu.transform([[cpu_score]])[0][0])
                s_gpu = float(current_scaler_gpu.transform([[gpu_score]])[0][0])
                
                input_df = pd.DataFrame({'cpu_scaled': [s_cpu], 'gpu_scaled': [s_gpu], 'total_price': [total_price]})
                ai_rank = current_model.predict(input_df)[0]

            # ส่งกลับหน้าเว็บโดยไม่ต้องมี metrics แล้ว
            return JsonResponse({
                'status': 'success', 
                'ai_rank': ai_rank,
                'model_details': {
                    'model_name': model_name,
                    'scaled_cpu': round(s_cpu, 4),
                    'scaled_gpu': round(s_gpu, 4)
                }
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return JsonResponse({'status': 'invalid request'})