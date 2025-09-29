#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------------- Regex & mapping rules ----------------

MITO_PATTERNS = [
    r"(?i)^chr?m(t)?$", r"(?i)^m(t)?$", r"(?i)mitochond", r"(?i)\bmito\b", r"(?i)^chrmt$"
]

NON_PRIMARY_HINTS = [
    r"(?i)random", r"(?i)alt", r"(?i)fix", r"(?i)hap", r"(?i)\bun", r"(?i)unlocalized",
    r"(?i)scaff", r"(?i)patch", r"(?i)decoy", r"(?i)contig",
    r"(?i)^GL\d", r"(?i)^KI\d", r"(?i)^NW_\d", r"(?i)^NT_\d"
]

def is_mito(name: str) -> bool:
    return any(re.search(p, name) for p in MITO_PATTERNS)

def looks_non_primary(name: str) -> bool:
    if "_" in name and not re.match(r"(?i)^chr?\d+$", name) and not re.match(r"(?i)^chr?[xy]$", name):
        return True
    return any(re.search(p, name) for p in NON_PRIMARY_HINTS)

def extract_numeric_token(name: str) -> Optional[int]:
    m = re.fullmatch(r"(?i)^chr0*([1-9]\d*)$", name)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"0*([1-9]\d*)$", name)
    if m:
        return int(m.group(1))
    return None

def build_canonical_mapping(all_records: List[Tuple[str, int]]) -> Dict[str, str]:
    if not all_records:
        return {}
    numeric: List[Tuple[int, str]] = []
    others: List[str] = []
    for name, _L in all_records:
        n = extract_numeric_token(name)
        if n is not None:
            numeric.append((n, name))
        else:
            others.append(name)
    numeric_sorted = [nm for _, nm in sorted(numeric, key=lambda x: x[0])]
    others_sorted  = sorted(others, key=lambda s: s.lower())
    ordered = numeric_sorted + others_sorted
    return {nm: f"chr{i}" for i, nm in enumerate(ordered, start=1)}

# ---------------- I/O helpers ----------------

def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ---------------- GTF helpers ----------------

def parse_gtf_attributes(attr_field: str) -> Dict[str, str]:
    raw = attr_field.strip().strip(";")
    out: Dict[str, str] = {}
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    for p in parts:
        if " " in p and "=" not in p:
            key, val = p.split(" ", 1)
            val = val.strip().strip('"')
        else:
            if "=" in p:
                key, val = p.split("=", 1)
                val = val.strip().strip('"')
            else:
                key, val = p, ""
        out[key.strip().lower()] = val.strip().lower()
    return out

def is_protein_coding(attrs: Dict[str, str]) -> Optional[bool]:
    keys = ("transcript_type", "transcript_biotype", "gene_type", "gene_biotype", "biotype")
    seen_any = False
    for k in keys:
        if k in attrs:
            seen_any = True
            if "protein_coding" in attrs[k]:
                return True
    if seen_any:
        return False
    return None

def derive_sizes_from_gtf(gtf_path: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            chrom, _, _, _, end = fields[:5]
            try:
                e = int(end)
            except ValueError:
                continue
            if e > sizes.get(chrom, 0):
                sizes[chrom] = e
    return sizes

# ---------------- Core per singolo assembly ----------------

def build_cds_flags_from_gtf(gtf_path: str, out_npz: str, out_map_tsv: str, only_pc: bool) -> bool:
    """Ritorna True se completato, False altrimenti (errori già loggati)."""
    if not os.path.exists(gtf_path):
        print(f"[ERROR] GTF non trovato: {gtf_path}", file=sys.stderr)
        return False

    sizes_by_orig = derive_sizes_from_gtf(gtf_path)
    if not sizes_by_orig:
        print(f"[ERROR] Nessuna dimensione derivata dal GTF: {gtf_path}", file=sys.stderr)
        return False

    # Filtra contig primari
    records: List[Tuple[str, int]] = []
    for nm, L in sizes_by_orig.items():
        if L <= 0:
            continue
        if is_mito(nm):
            continue
        if looks_non_primary(nm):
            continue
        records.append((nm, L))
    if not records:
        print(f"[ERROR] Nessun contig primario dopo filtro in {gtf_path}", file=sys.stderr)
        return False

    mapping = build_canonical_mapping(records)  # {orig -> chr#}
    if not mapping:
        print(f"[ERROR] Mapping canonico vuoto in {gtf_path}", file=sys.stderr)
        return False

    # Log mapping ordinato
    ordered = sorted(mapping.items(), key=lambda kv: int(kv[1].replace("chr","")))
    print(f"[INFO] {gtf_path}: {len(mapping)} contig primari mappati → chr1..chr{len(mapping)}")
    for orig, cn in ordered:
        print(f"  {orig} -> {cn} (len={sizes_by_orig[orig]:,})")

    # Parse CDS
    cds_by_canonical: Dict[str, List[Tuple[int, int]]] = {cn: [] for cn in mapping.values()}
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _source, feature, start, end, _score, _strand, _frame, attr_field = fields
            if feature != "CDS":
                continue
            if chrom not in mapping:
                continue

            if only_pc:
                attrs = parse_gtf_attributes(attr_field)
                pc = is_protein_coding(attrs)
                if pc is False:
                    continue

            try:
                s = int(start) - 1
                e = int(end)
            except ValueError:
                continue

            L = sizes_by_orig[chrom]
            if s < 0: s = 0
            if e > L: e = L
            if e > s:
                cds_by_canonical[mapping[chrom]].append((s, e))

    # Costruisci boolean flags
    payload: Dict[str, np.ndarray] = {}
    total = 0
    for orig, cn in ordered:
        n = sizes_by_orig[orig]
        arr = np.zeros(n, dtype=bool)
        ivs = cds_by_canonical.get(cn, [])
        if ivs:
            ivs.sort()
            cur_s, cur_e = ivs[0]
            merged: List[Tuple[int, int]] = []
            for s, e in ivs[1:]:
                if s <= cur_e:
                    if e > cur_e: cur_e = e
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            for s, e in merged:
                arr[s:e] = True
        payload[cn] = arr
        csum = int(arr.sum())
        total += csum
        print(f"{cn}: {csum:,} basi codificanti / {n:,}")

    ensure_dir(os.path.dirname(out_npz))
    np.savez_compressed(out_npz, **payload)
    # mapping TSV di supporto
    with open(out_map_tsv, "w") as mfh:
        mfh.write("original\tcanonical\tlength\n")
        for orig, cn in ordered:
            mfh.write(f"{orig}\t{cn}\t{sizes_by_orig[orig]}\n")

    print(f"[DONE] Totale basi codificanti: {total:,}")
    print(f"[DONE] Salvato NPZ: {out_npz}")
    print(f"[DONE] Salvato mapping: {out_map_tsv}")
    return True

# ---------------- Batch runner su assemblies.json ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Batch CDS flags (derive-only) per ciascun assembly elencato in assemblies.json."
    )
    ap.add_argument("--assemblies-json", required=True,
                    help="File JSON con lista di oggetti assembly (usa solo i campi 'id').")
    ap.add_argument("--gtf-template", required=True,
                    help="Template path del GTF con placeholder {assembly} (es: data/raw/{assembly}/annotation.gtf.gz)")
    ap.add_argument("--only-protein-coding", action="store_true",
                    help="Filtra ai soli CDS protein_coding quando l'informazione è disponibile negli attributi.")
    ap.add_argument("--outfile-name", default="cds_flags.primary.bool.npz",
                    help="Nome del file NPZ finale salvato in data/processed/{assembly}/ (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="Mostra cosa farebbe senza eseguire.")
    args = ap.parse_args()

    # Carica assemblies.json e prendi solo gli id
    with open(args.assemblies_json, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        print("[ERROR] assemblies.json deve essere una lista di oggetti con campo 'id'.", file=sys.stderr)
        sys.exit(2)

    ids: List[str] = []
    for obj in data:
        if isinstance(obj, dict) and "id" in obj and obj["id"]:
            ids.append(str(obj["id"]))
    if not ids:
        print("[ERROR] Nessun 'id' trovato in assemblies.json.", file=sys.stderr)
        sys.exit(3)

    print(f"[INFO] Assemblies da processare: {', '.join(ids)}")

    # Esegui per ogni assembly
    all_ok = True
    for asm in ids:
        gtf_path = args.gtf_template.format(assembly=asm)
        out_dir = os.path.join("data", "processed", asm)
        out_npz = os.path.join(out_dir, args.outfile_name)
        out_map = os.path.join(out_dir, "chrom_mapping.tsv")

        print(f"\n[RUN] {asm}")
        print(f"  GTF: {gtf_path}")
        print(f"  OUT: {out_npz}")

        if args.dry_run:
            # Skip esecuzione
            if not os.path.exists(gtf_path):
                print(f"  [WARN] (dry-run) GTF non esiste: {gtf_path}")
            else:
                print(f"  [OK] (dry-run) pronto a eseguire")
            continue

        ok = build_cds_flags_from_gtf(
            gtf_path=gtf_path,
            out_npz=out_npz,
            out_map_tsv=out_map,
            only_pc=args.only_protein_coding
        )
        all_ok = all_ok and ok

    if args.dry_run:
        print("\n[DRY-RUN DONE]")
    elif not all_ok:
        print("\n[WARN] Completato con alcuni errori (vedi log sopra).", file=sys.stderr)
    else:
        print("\n[ALL DONE] Tutti gli assemblies completati.")

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.exit(1)
