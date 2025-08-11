import os
from pathlib import Path
import shutil
import yaml

# === CONFIGURATION ===
dataset_dir = Path("batch_1_1_lyd_remapped")  # <-- your dataset directory
splits = ['train', 'val']
backup_old_labels = False

# Class remapping
# Format: old_index -> new_index (or None to remove)
label_map = {
    0: 0,    # Shepherd's Crook → 0
    1: 1,    # Stem End Canker → 1 (Canker)
    2: 1,    # Elliptical Canker → 1
    3: 1,    # Girdling Canker → 1
    4: None, # Blossom Blight → REMOVE
    5: None, # Rootstock Blight → REMOVE
    6: 1,    # Stem End Canker 2 → 1
}

new_class_names = {
    0: "Shepherd's Crook",
    1: "Canker"
}

def remap_labels_in_file(file_path, label_map):
    new_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_class = int(parts[0])
            if old_class not in label_map:
                continue
            new_class = label_map[old_class]
            if new_class is None:
                continue  # skip this object entirely
            parts[0] = str(new_class)
            new_lines.append(" ".join(parts))
    with open(file_path, 'w') as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

def process_split(split):
    label_dir = dataset_dir / "labels" / split
    for file in label_dir.glob("*.txt"):
        if backup_old_labels:
            shutil.copy(file, file.with_suffix(".txt.bak"))
        remap_labels_in_file(file, label_map)

def update_yaml():
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    data['names'] = new_class_names
    data['nc'] = len(new_class_names)
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print("✅ Updated data.yaml")

if __name__ == "__main__":
    for split in splits:
        process_split(split)
    update_yaml()
    print("✅ Finished remapping labels and updating data.yaml")
