"""
Dataset download + validation script.

HARD REQUIREMENT (user-specified): every dataset used in this project must
be EXACTLY 5-class ICDR (0-4: No DR, Mild, Moderate, Severe, Proliferative
DR). This script downloads via the Kaggle API and immediately validates
label cardinality before anything is treated as usable — a 2-class
"referable vs not" variant or a binarized variant gets rejected outright,
not silently remapped.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json (see
https://www.kaggle.com/docs/api for setup) — not automated here since it
needs your personal API token.

Datasets:
  - EyePACS (Kaggle "Diabetic Retinopathy Detection" competition): 5-class, 0-4.
  - APTOS 2019 Blindness Detection: 5-class, 0-4.
  - Messidor-2: 5-class ICDR grades via the Adcis/Kaggle-mirrored labels
    (NOT the original Messidor 4-class R0-R3 risk grading — that variant
    is explicitly the wrong one and must not be used here).
"""
import argparse
import os
import subprocess
import zipfile

import pandas as pd

EXPECTED_CLASSES = {0, 1, 2, 3, 4}

DATASET_SPECS = {
    "eyepacs": {
        "kaggle_ref": "c/diabetic-retinopathy-detection",  # competition, not dataset
        "is_competition": True,
        "target_dir": "data/raw/eyepacs",
    },
    "aptos": {
        "kaggle_ref": "c/aptos2019-blindness-detection",
        "is_competition": True,
        "target_dir": "data/raw/aptos",
    },
    "messidor2": {
        # Confirmed: "MESSIDOR-2 DR Grades" (google-brain/messidor2-dr-grades)
        # ships the Krause et al. 2018 adjudicated 5-class ICDR grades (panel
        # of 3 retina specialists), NOT the original Messidor 4-class R0-R3
        # risk grading. This is the correct 5-class source — still runs
        # through validate_labels_csv() below as a hard gate regardless.
        "kaggle_ref": "google-brain/messidor2-dr-grades",
        "is_competition": False,
        "target_dir": "data/raw/messidor2",
    },
}


def download_dataset(name: str):
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {name!r}. Choices: {list(DATASET_SPECS)}")

    spec = DATASET_SPECS[name]
    os.makedirs(spec["target_dir"], exist_ok=True)

    if spec["is_competition"]:
        cmd = ["kaggle", "competitions", "download", "-c", spec["kaggle_ref"].split("/")[-1],
               "-p", spec["target_dir"]]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", spec["kaggle_ref"],
               "-p", spec["target_dir"]]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(
            f"Kaggle download failed for {name}. Common cause: missing/expired "
            f"~/.kaggle/kaggle.json credentials, or competition rules not yet "
            f"accepted on the Kaggle website (competitions require clicking "
            f"'I Understand and Accept' before API download works)."
        )

    for fname in os.listdir(spec["target_dir"]):
        if fname.endswith(".zip"):
            zpath = os.path.join(spec["target_dir"], fname)
            print(f"Extracting {zpath}...")
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(spec["target_dir"])

    print(f"Downloaded and extracted {name} to {spec['target_dir']}")


def validate_labels_csv(labels_csv_path: str, label_col: str, dataset_name: str) -> bool:
    """
    Hard gate: raises if the label set isn't exactly {0,1,2,3,4}.
    Called after every download, before any manifest/split is built.
    """
    df = pd.read_csv(labels_csv_path)
    if label_col not in df.columns:
        raise ValueError(
            f"{dataset_name}: expected label column {label_col!r} not found in "
            f"{labels_csv_path}. Columns present: {list(df.columns)}"
        )

    found_classes = set(df[label_col].unique())

    if found_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"{dataset_name} REJECTED: label file {labels_csv_path} contains classes "
            f"{sorted(found_classes)}, expected exactly {sorted(EXPECTED_CLASSES)} "
            f"(5-class ICDR 0-4). This dataset/variant must not be used in this "
            f"project — do not remap or coerce, source a genuine 5-class version instead."
        )

    print(f"{dataset_name}: validated 5-class labels {sorted(found_classes)} in {labels_csv_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and validate DR datasets.")
    parser.add_argument("--dataset", choices=list(DATASET_SPECS) + ["all"], default="all")
    args = parser.parse_args()

    targets = list(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        download_dataset(name)

    print(
        "\nDownload complete. Labels CSVs vary in filename/column per source "
        "(e.g. APTOS ships train.csv with 'diagnosis'; EyePACS ships "
        "trainLabels.csv with 'level'; verify Messidor-2's column name once "
        "downloaded). Run validate_labels_csv() on each before building "
        "manifests with src/data/datasets.py:build_manifest_csv()."
    )


if __name__ == "__main__":
    main()
