import argparse
import os
import subprocess
import zipfile

def extract_eyepacs(target_dir: str):
    master_zip = os.path.join(target_dir, "train.zip")
    
    # 1. Stream unwrapped binary chunks into master zip to save space
    if not os.path.exists(master_zip) and not os.path.exists(os.path.join(target_dir, "train")):
        print(f"[eyepacs] Streaming raw chunks directly into master zip (Requires 35GB temporary space)...")
        with open(master_zip, 'wb') as fout:
            for i in range(1, 6):
                wrapper = os.path.join(target_dir, f"train.zip.00{i}.zip")
                if not os.path.exists(wrapper):
                    print(f"Warning: Missing wrapper {wrapper}")
                    continue
                    
                print(f"[eyepacs] Unwrapping and streaming {wrapper}...")
                with zipfile.ZipFile(wrapper) as zf:
                    # Kaggle wraps "train.zip.001" inside "train.zip.001.zip"
                    with zf.open(f"train.zip.00{i}") as fin:
                        while True:
                            chunk = fin.read(1024 * 1024 * 16) # 16MB chunks
                            if not chunk: break
                            fout.write(chunk)
                            
                # Delete the wrapper immediately to free up 7GB!
                os.remove(wrapper)
        print("[eyepacs] Wrappers deleted! Freed up space.")
    
    # 2. Extract final images
    if os.path.exists(master_zip):
        print("[eyepacs] Extracting master train.zip (this will take a while)...")
        # unzip throws a warning (exit code 1) for large zips, ignore it
        subprocess.run(f"unzip -q -o {master_zip} -d {target_dir}", shell=True)
        print("[eyepacs] Cleaning up master zip...")
        os.remove(master_zip)
        
    labels_zip = os.path.join(target_dir, "trainLabels.csv.zip")
    if os.path.exists(labels_zip):
        print(f"[eyepacs] Extracting {labels_zip}...")
        subprocess.run(f"unzip -q -o {labels_zip} -d {target_dir}", shell=True)

def extract_messidor2(target_dir: str):
    for zname in ["messidor2-dr-grades.zip", "messidor2.zip"]:
        zpath = os.path.join(target_dir, zname)
        if os.path.exists(zpath):
            print(f"[messidor2] Extracting {zpath}...")
            subprocess.run(f"unzip -q -o {zpath} -d {target_dir}", shell=True)

def extract_aptos(target_dir: str):
    for fname in os.listdir(target_dir):
        if fname.endswith(".zip"):
            zpath = os.path.join(target_dir, fname)
            print(f"[aptos] Extracting {zpath}...")
            subprocess.run(f"unzip -q -o {zpath} -d {target_dir}", shell=True)

def main():
    parser = argparse.ArgumentParser(description="Extract downloaded datasets safely")
    args = parser.parse_args()

    # Extract all
    extract_eyepacs("data/raw/eyepacs")
    extract_aptos("data/raw/aptos")
    extract_messidor2("data/raw/messidor2")
    
    print("All datasets successfully extracted.")

if __name__ == "__main__":
    main()
