import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.stratified_split import train_val_split, stratified_split, eval_only_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["eyepacs", "aptos", "messidor2", "all"], default="all")
    args = parser.parse_args()

    manifests = {
        "eyepacs": "data/processed/eyepacs_manifest.csv",
        "aptos": "data/processed/aptos_manifest.csv",
        "messidor2": "data/processed/messidor2_manifest.csv"
    }

    targets = list(manifests.keys()) if args.dataset == "all" else [args.dataset]

    if "eyepacs" in targets and os.path.exists(manifests["eyepacs"]):
        print("Building splits for EyePACS (90% train, 10% val)...")
        train_val_split(manifests["eyepacs"], "eyepacs", "data/splits", train_frac=0.90)

    if "aptos" in targets and os.path.exists(manifests["aptos"]):
        print("Building splits for APTOS (80% train, 10% val, 10% test)...")
        stratified_split(manifests["aptos"], "aptos", "data/splits", train_frac=0.80, val_frac=0.10, test_frac=0.10)

    if "messidor2" in targets and os.path.exists(manifests["messidor2"]):
        print("Building splits for Messidor-2 (100% test)...")
        eval_only_split(manifests["messidor2"], "messidor2", "data/splits")

if __name__ == "__main__":
    main()
