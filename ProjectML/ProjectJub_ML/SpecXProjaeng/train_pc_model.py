import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

print("🚀 เริ่มต้นกระบวนการฝึกสอนและวิเคราะห์ประสิทธิภาพ AI...")

# ==========================================
# 1. สร้างข้อมูลจำลอง (Synthetic Dataset)
# ==========================================
print("[1/4] กำลังสร้างข้อมูลจำลอง 10,000 สเปค...")
np.random.seed(42)
num_samples = 10000 

data = {
    'cpu_score': np.random.randint(2000, 60000, num_samples),
    'gpu_score': np.random.randint(1000, 40000, num_samples),
    'ram_gb': np.random.choice([8, 16, 32, 64], num_samples),
    'ssd_gb': np.random.choice([256, 500, 1000, 2000], num_samples),
    'total_price': np.random.randint(10000, 150000, num_samples),
}
df = pd.DataFrame(data)

# ==========================================
# 2. การกำหนดเลเบล (Labeling Logic)
# ==========================================
def assign_tier(row):
    cpu, gpu, ram, ssd, price = row['cpu_score'], row['gpu_score'], row['ram_gb'], row['ssd_gb'], row['total_price']
    perf = (cpu * 0.4) + (gpu * 0.6)
    
    if cpu > (gpu * 3) or gpu > (cpu * 3): return '⚠️ สเปคคอขวด (Bottleneck)'
    if perf < 12000 or ram < 16 or ssd < 500: return '🥉 ระดับเริ่มต้น (Entry)'
    if perf >= 35000 and ram >= 32 and price >= 40000 and ssd >= 1000:
        if cpu > (gpu * 1.3): return '💼 สายทำงาน (Pro Workstation)'
        if gpu > (cpu * 1.3): return '🎮 สายเกมมิ่ง (Pro Gaming)'
        return '🌟 ไฮเอนด์สมดุล (Ultra)'
    if perf >= 20000 and price <= 35000: return '🔥 ระดับคุ้มค่า (Sweet Spot)'
    return '⚖️ ระดับกลาง (Mid-Range)'

df['tier_label'] = df.apply(assign_tier, axis=1)

# ==========================================
# 3. ฝึกสอนโมเดล (Training)
# ==========================================
print("[2/4] กำลังฝึกสอนโมเดล (Random Forest)...")
X = df[['cpu_score', 'gpu_score', 'ram_gb', 'ssd_gb', 'total_price']]
y = df['tier_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. วิเคราะห์ประสิทธิภาพ (Evaluation)
# ==========================================
print("[3/4] กำลังวิเคราะห์ประสิทธิภาพ (Performance Analysis)...")
y_pred = model.predict(X_test)

# คำนวณค่าต่างๆ
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1  = f1_score(y_test, y_pred, average='weighted')

print("-" * 40)
print(f"✅ Accuracy  : {acc:.4f} (ความแม่นยำรวม)")
print(f"🎯 Precision : {pre:.4f} (ความถูกต้องในการทาย)")
print(f"📢 Recall    : {rec:.4f} (ความสามารถในการตรวจจับ)")
print(f"🧪 F1-Score  : {f1:.4f} (ค่าเฉลี่ยสมดุล)")
print("-" * 40)
print("📝 รายงานแยกตามกลุ่ม (Classification Report):")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. บันทึกโมเดล (Export)
# ==========================================
joblib.dump(model, 'pc_tier_model.joblib')
print("[4/4] 🎉 บันทึกไฟล์ 'pc_tier_model.joblib' เรียบร้อยแล้ว!")