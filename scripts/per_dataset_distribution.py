"""Compare raw train+test label concat vs processed manifest."""
import pandas as pd


def show(src, raw_counts, raw_pct, proc_n, proc_pct, raw_n):
    print(f"\n=== {src} ===")
    print(f"  raw concat total: {raw_n}")
    print(f"  raw counts: {raw_counts}")
    print(f"  raw pct:    {raw_pct}")
    print(f"  processed:  n={proc_n}, pct={proc_pct}")


# EyePACS
e_train = pd.read_csv("data/raw/eyepacs/trainLabels.csv")
e_test = pd.read_csv("data/raw/eyepacs/testLabels.csv.csv")
e_raw = pd.concat([
    e_train.rename(columns={"image": "image_id", "level": "label"})[["image_id", "label"]],
    e_test.rename(columns={"image": "image_id", "level": "label"})[["image_id", "label"]],
], ignore_index=True)
e_counts = e_raw.label.value_counts().sort_index()
e_pct = (e_counts / e_counts.sum() * 100).round(4)
proc = pd.read_csv("data/processed/eyepacs_manifest.csv")
p_counts = proc.label.value_counts().sort_index()
p_pct = (p_counts / p_counts.sum() * 100).round(4)
show("eyepacs", e_counts.to_dict(), e_pct.to_dict(), len(proc), p_pct.to_dict(), len(e_raw))

# APTOS: trainLabels only (raw/test has no diagnosis column — unlabeled Kaggle test set)
a_train = pd.read_csv("data/raw/aptos/train.csv")
a_raw = a_train.rename(columns={"id_code": "image_id", "diagnosis": "label"})[["image_id", "label"]]
a_counts = a_raw.label.value_counts().sort_index()
a_pct = (a_counts / a_counts.sum() * 100).round(4)
proc = pd.read_csv("data/processed/aptos_manifest.csv")
p_counts = proc.label.value_counts().sort_index()
p_pct = (p_counts / p_counts.sum() * 100).round(4)
show("aptos", a_counts.to_dict(), a_pct.to_dict(), len(proc), p_pct.to_dict(), len(a_raw))

# Messidor-2
m = pd.read_csv("data/raw/messidor2/messidor-2/messidor_data.csv")
m_use = m[["image_id", "adjudicated_dr_grade"]].rename(columns={"adjudicated_dr_grade": "label"})
m_counts = m_use.label.value_counts().sort_index()
m_pct = (m_counts / m_counts.sum() * 100).round(4)
proc = pd.read_csv("data/processed/messidor2_manifest.csv")
p_counts = proc.label.value_counts().sort_index()
p_pct = (p_counts / p_counts.sum() * 100).round(4)
show("messidor2", m_counts.to_dict(), m_pct.to_dict(), len(proc), p_pct.to_dict(), len(m))
