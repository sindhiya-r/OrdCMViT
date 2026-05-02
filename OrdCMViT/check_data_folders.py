import os
from pathlib import Path

# Set BCMID root directory (use "." if already inside BCMID folder)
root = Path("./BCMID/")

total = 0
both_modalities = 0
missing_us = 0
missing_mm = 0

details = []

for item in root.iterdir():
    if item.is_dir():
        total += 1
        
        us_dir = item / "Ultrasound"
        mm_dir = item / "Mammogram"
        
        has_us = us_dir.exists()
        has_mm = mm_dir.exists()
        
        if has_us and has_mm:
            both_modalities += 1
        if not has_us:
            missing_us += 1
        if not has_mm:
            missing_mm += 1
        
        details.append((item.name, has_us, has_mm))

print("\n===== BCMID Folder Summary =====")
print(f"Total patient folders: {total}")
print(f"Patients with BOTH modalities: {both_modalities}")
print(f"Patients missing Ultrasound: {missing_us}")
print(f"Patients missing Mammogram: {missing_mm}")

print("\n===== Sample Breakdown (first 10 patients) =====")
for name, us, mm in details[:10]:
    print(f"{name:12} | US: {us} | Mammo: {mm}")