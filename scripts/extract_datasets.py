import argparse
import os
import subprocess
import zipfile

def extract_eyepacs(target_dir: str):
    # ── Train set ──
    master_zip = os.path.join(target_dir, "train.zip")
    
    # 1. Stream unwrapped binary chunks into master zip to save space
    if not os.path.exists(master_zip) and not os.path.exists(os.path.join(target_dir, "train")):
        wrapper_chunks = sorted([f for f in os.listdir(target_dir) if f.startswith("train.zip.") and f.endswith(".zip")])
        if wrapper_chunks:
            print(f"[eyepacs] Streaming {len(wrapper_chunks)} train chunks into master zip...")
            with open(master_zip, 'wb') as fout:
                for wrapper_name in wrapper_chunks:
                    wrapper = os.path.join(target_dir, wrapper_name)
                    inner_name = wrapper_name.removesuffix(".zip")  # e.g. "train.zip.001"
                    print(f"[eyepacs] Unwrapping and streaming {wrapper_name}...")
                    with zipfile.ZipFile(wrapper) as zf:
                        with zf.open(inner_name) as fin:
                            while True:
                                chunk = fin.read(1024 * 1024 * 16)  # 16MB chunks
                                if not chunk: break
                                fout.write(chunk)
                    os.remove(wrapper)
            print("[eyepacs] Train wrappers deleted.")
    
    # 2. Extract final train images
    if os.path.exists(master_zip):
        print("[eyepacs] Extracting master train.zip...")
        subprocess.run(f"unzip -q -o {master_zip} -d {target_dir}", shell=True)
        os.remove(master_zip)
        
    labels_zip = os.path.join(target_dir, "trainLabels.csv.zip")
    if os.path.exists(labels_zip):
        subprocess.run(f"unzip -q -o {labels_zip} -d {target_dir}", shell=True)

    # ── Test set ──
    test_master_zip = os.path.join(target_dir, "test.zip")
    
    if not os.path.exists(test_master_zip) and not os.path.exists(os.path.join(target_dir, "test")):
        test_chunks = sorted([f for f in os.listdir(target_dir) if f.startswith("test.zip.") and f.endswith(".zip")])
        if test_chunks:
            print(f"[eyepacs] Streaming {len(test_chunks)} test chunks into master zip...")
            with open(test_master_zip, 'wb') as fout:
                for wrapper_name in test_chunks:
                    wrapper = os.path.join(target_dir, wrapper_name)
                    inner_name = wrapper_name.removesuffix(".zip")
                    print(f"[eyepacs] Unwrapping and streaming {wrapper_name}...")
                    with zipfile.ZipFile(wrapper) as zf:
                        with zf.open(inner_name) as fin:
                            while True:
                                chunk = fin.read(1024 * 1024 * 16)
                                if not chunk: break
                                fout.write(chunk)
                    os.remove(wrapper)
            print("[eyepacs] Test wrappers deleted.")
    
    if os.path.exists(test_master_zip):
        print("[eyepacs] Extracting master test.zip...")
        subprocess.run(f"unzip -q -o {test_master_zip} -d {target_dir}", shell=True)
        os.remove(test_master_zip)

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
    parser.add_argument("--dataset", choices=["eyepacs", "aptos", "messidor2", "all"], default="all", help="Which dataset to extract")
    args = parser.parse_args()

    targets = ["eyepacs", "aptos", "messidor2"] if args.dataset == "all" else [args.dataset]

    if "eyepacs" in targets:
        extract_eyepacs("data/raw/eyepacs")
    if "aptos" in targets:
        extract_aptos("data/raw/aptos")
    if "messidor2" in targets:
        extract_messidor2("data/raw/messidor2")
    
    print("Requested datasets successfully extracted.")

if __name__ == "__main__":
    main()
