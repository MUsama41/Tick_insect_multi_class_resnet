import os
import pandas as pd

def get_label_info(dirpath):
    """
    Map folder structure to clean labels.
    """
    dirpath = dirpath.lower()
    if 'hyaloma anatolicum _ female' in dirpath:
        return 'hyalomma_female', 0
    elif 'hyalomma anatolicum _ male' in dirpath:
        return 'hyalomma_male', 1
    elif 'rhipicephalus microplus _ female' in dirpath:
        return 'rhipicephalus_female', 2
    elif 'rhipicephalus microplus _ male' in dirpath:
        return 'rhipicephalus_male', 3
    return None, None

def generate_metadata(root_dir, output_csv="metadata.csv"):
    data = []
    for dirpath, _, filenames in os.walk(root_dir):
        label, label_idx = get_label_info(dirpath)
        if label is not None:
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(dirpath, f)
                    data.append({
                        "file_path": file_path,
                        "label": label,
                        "label_idx": label_idx
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated {output_csv} with {len(df)} entries.")
    return df

if __name__ == "__main__":
    ROOT_DIR = r"d:\projects\tick_classification_model\Tick-20260131T070319Z-3-001\Tick"
    generate_metadata(ROOT_DIR)
