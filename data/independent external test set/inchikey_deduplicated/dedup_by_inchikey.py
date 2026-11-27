# -*- coding: utf-8 -*-
"""Rebuild InChIKey-deduplicated independent test subsets.

Pipeline (aligned with reviewer_audit/14_restore_c_o_dedup):
  1. Restore hydrogen-suppressed brackets: [#6] -> C, [#8] -> O
  2. Compute RDKit InChIKey
  3. Drop rows whose InChIKey is in train OR validation
  4. Drop within-test InChIKey duplicates

Writes class/pathway/superclass CSVs next to this script (original column schema).

    python "data/independent external test set/inchikey_deduplicated/dedup_by_inchikey.py"
"""
import os
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
DATA = os.path.dirname(EXT)

SUBSETS = [
    ("class_subset.csv", "smiles", "class"),
    ("pathway_subset.csv", "smiles", "pathway"),
    ("superclass_subset.csv", "smiles", "superclass"),
]


def restore_co(smiles):
    if not isinstance(smiles, str):
        return ""
    return smiles.replace("[#6]", "C").replace("[#8]", "O")


def inchikey(smiles):
    smi = restore_co(smiles)
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key or None


def main():
    train = pd.read_csv(os.path.join(DATA, "train_data.csv"))
    val = pd.read_csv(os.path.join(DATA, "validation_data.csv"))
    ref = {
        k
        for k in (inchikey(s) for s in pd.concat([train["index"], val["index"]], ignore_index=True))
        if k
    }
    print("ref InChIKeys (train∪val): %d" % len(ref))

    frames = []
    for name, smi_col, label_col in SUBSETS:
        df = pd.read_csv(os.path.join(EXT, name))
        part = df[[smi_col, label_col]].copy()
        part = part.rename(columns={smi_col: "smiles", label_col: "label"})
        part["source_file"] = name
        part["label_column"] = label_col
        part["row_in_subset"] = range(len(part))
        part["inchikey"] = [inchikey(s) for s in part["smiles"]]
        frames.append(part)
        n_hash = int(part["smiles"].astype(str).str.contains(r"\[#[68]\]", regex=True).sum())
        print("%s: loaded %d (with [#6]/[#8]: %d)" % (name, len(part), n_hash))

    all_df = pd.concat(frames, ignore_index=True)
    n0 = len(all_df)

    parse_ok = all_df["inchikey"].notna()
    hit_ref = all_df["inchikey"].isin(ref)
    # Keep first occurrence of each InChIKey across the combined independent set.
    within_dup = parse_ok & all_df["inchikey"].duplicated(keep="first")
    keep = parse_ok & ~hit_ref & ~within_dup
    all_df["keep"] = keep

    print(
        "parse_fail=%d  hit_train|val=%d  within_dup_drop=%d  kept=%d / %d"
        % (
            int((~parse_ok).sum()),
            int((parse_ok & hit_ref).sum()),
            int((parse_ok & ~hit_ref & within_dup).sum()),
            int(keep.sum()),
            n0,
        )
    )

    total = 0
    for name, smi_col, label_col in SUBSETS:
        sub = all_df[(all_df["source_file"] == name) & all_df["keep"]].copy()
        out = pd.DataFrame(
            {
                "smiles": sub["smiles"].values,
                label_col: sub["label"].values,
            }
        )
        out_path = os.path.join(HERE, name)
        out.to_csv(out_path, index=False)
        n_in = int((all_df["source_file"] == name).sum())
        print("%s: %d -> %d" % (name, n_in, len(out)))
        total += len(out)

    print("total: %d" % total)


if __name__ == "__main__":
    main()
