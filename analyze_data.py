import os
from collections import Counter

root_dir = r"d:\projects\tick_classification_model\Tick-20260131T070319Z-3-001\Tick"

def analyze_data(root):
    stats = []
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames: # Leaf directory
            images = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if images:
                stats.append({
                    "path": dirpath,
                    "count": len(images)
                })
    return stats

if __name__ == "__main__":
    results = analyze_data(root_dir)
    for res in results:
        print(f"{res['path']}: {res['count']}")
