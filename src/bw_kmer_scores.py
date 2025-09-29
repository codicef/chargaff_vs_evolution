#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, numpy as np
import pyBigWig

# ---------- quantization ----------
def quantize01_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Map [0,1] -> uint8 0..100 (step 0.01). NaN -> 255 sentinel."""
    out = np.full(arr.shape, 255, dtype=np.uint8)
    m = np.isfinite(arr)
    if not np.any(m):
        return out
    x = np.clip(arr[m], 0.0, 1.0)
    q = np.rint(x * 100.0).astype(np.int16)
    q = np.clip(q, 0, 100).astype(np.uint8)
    out[m] = q
    return out

def rescale(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear rescale [lo,hi] -> [0,1]; preserves NaN."""
    v = vals.astype(float, copy=False)
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        # se lo/hi non sono validi, torna tutto NaN
        return np.full_like(v, np.nan)
    with np.errstate(invalid="ignore"):
        r = (v - lo) / (hi - lo)
    return r

# ---------- rolling mean (intra-finestra) ----------
def rolling_mean(vals: np.ndarray, k: int) -> np.ndarray:
    v = vals.astype(float, copy=False)
    n = v.size
    if n < k:
        return np.empty(0, dtype=float)

    finite = np.isfinite(v).astype(np.int64)
    v_fill = np.nan_to_num(v, nan=0.0)

    cs  = np.concatenate(([0.0], np.cumsum(v_fill)))
    csn = np.concatenate(([0],   np.cumsum(finite)))

    win_sum = cs[k:]  - cs[:-k]
    win_cnt = csn[k:] - csn[:-k]

    with np.errstate(invalid="ignore", divide="ignore"):
        m = win_sum / win_cnt
    m[win_cnt < k] = np.nan
    return m

def robust_min_max_from_values(x: np.ndarray, mode: str):
    """Ritorna (lo, hi) globali secondo la modalità scelta."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    if mode == "minmax":
        return float(np.min(x)), float(np.max(x))
    elif mode == "p01p99":
        return float(np.percentile(x, 1.0)), float(np.percentile(x, 99.0))
    elif mode == "p05p95":
        return float(np.percentile(x, 5.0)), float(np.percentile(x, 95.0))
    else:
        raise ValueError(f"unknown autoscale mode: {mode}")
def rescale_centered_on_zero(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Rescale in modo che:
      lo -> 0
       0 -> 0.5
      hi -> 1
    NaN preservati.
    """
    v = vals.astype(float, copy=False)
    out = np.full_like(v, np.nan)

    # parte negativa (lo..0 -> 0..0.5)
    mask_neg = v < 0
    if lo < 0 and np.any(mask_neg):
        with np.errstate(invalid="ignore"):
            out[mask_neg] = 0.5 * (v[mask_neg] - lo) / (0 - lo)
        out[mask_neg] = np.clip(out[mask_neg], 0.0, 0.5)

    # parte positiva (0..hi -> 0.5..1)
    mask_pos = v >= 0
    if hi > 0 and np.any(mask_pos):
        with np.errstate(invalid="ignore"):
            out[mask_pos] = 0.5 + 0.5 * (v[mask_pos] - 0) / (hi - 0)
        out[mask_pos] = np.clip(out[mask_pos], 0.5, 1.0)

    return out

import re, sys
from typing import List, Tuple, Dict, Optional

# ----------------------------
# Canonical chromosome selection & renaming
# ----------------------------

MITO_PATTERNS = [
    r"(?i)^chr?m(t)?$", r"(?i)^m(t)?$", r"(?i)mitochond", r"(?i)\bmito\b", r"(?i)^chrmt$"
]

# Tag tipici di contig non primari/decoy/patch/unlocalized
NON_PRIMARY_HINTS = [
    r"(?i)random", r"(?i)alt", r"(?i)fix", r"(?i)hap", r"(?i)\bun", r"(?i)unlocalized",
    r"(?i)scaff", r"(?i)patch", r"(?i)decoy", r"(?i)contig",
    r"(?i)^GL\d", r"(?i)^KI\d", r"(?i)^NW_\d", r"(?i)^NT_\d"
]

def is_mito(name: str) -> bool:
    return any(re.search(p, name) for p in MITO_PATTERNS)



def looks_non_primary(name: str) -> bool:
    if "_" in name and not re.match(r"(?i)^chr?\d+$", name):
        # Molti contig non primari hanno underscore; conserviamo eccezione per nomi semplici numerici
        return True
    # if not chr at start skip male
    if not re.match(r"(?i)^chr?", name):
        return True
    return any(re.search(p, name) for p in NON_PRIMARY_HINTS)

def extract_numeric_token(name: str) -> Optional[int]:
    """
    Estrae un numero 'di cromosoma' se presente (chr1, 1, chr01 -> 1).
    (Niente underscore ammessi qui: coerente con la regola degli underscore)
    """
    m = re.fullmatch(r"(?i)^chr0*([1-9]\d*)$", name)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"0*([1-9]\d*)$", name)
    if m:
        return int(m.group(1))
    return None

def build_canonical_mapping(all_records: List[Tuple[str, int]]) -> Dict[str, str]:
    """
    all_records: (orig_name, length) già filtrata da mito e non-primari.
    Ordine finale:
      1) numerici (chr1/1/chr01 -> n) per n crescente
      2) others (non numerici) in ordine alfabetico del nome originale
    Assegna etichette: chr1..chrN seguendo tale ordine.
    """
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
    others_sorted  = sorted(others)  # alfabetico puro

    ordered = numeric_sorted + others_sorted
    return {nm: f"chr{i}" for i, nm in enumerate(ordered, start=1)}


# ---------- helpers (mini) ----------
import json, os, sys

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def bw_path_for(assembly_id: str, method: str) -> str:
    return os.path.join("data", "raw", assembly_id, f"{assembly_id}.{method}.bw")

def outdir_for(assembly_id: str, method: str) -> str:
    return os.path.join("data", "processed", assembly_id, f"{method}_scores", "kmers_scores")
def run_one(assembly_id: str, method: str, kmin: int, kmax: int, autoscale: str, lo: float, hi: float):
    bw_path = bw_path_for(assembly_id, method)
    if not os.path.exists(bw_path):
        print(f"[ERROR] BigWig non trovato: {bw_path}", file=sys.stderr)
        return
    outdir = outdir_for(assembly_id, method)
    ensure_dir(outdir)

    miss = 0
    for k in range(kmin, kmax + 1, 2):
        base = os.path.join(outdir, f"k{k}")

        if os.path.exists(os.path.join(base, f"{method}_mean_k{k}.npz")) and \
             os.path.exists(os.path.join(base, f"{method}_first_k{k}.npz")) and \
                os.path.exists(os.path.join(base, f"{method}_center_k{k}.npz")):
            print(f"[INFO] Skipping esistente: {base}")
        else:
            print(f"[INFO] Missing: {base}")
            miss += 1
    if miss == 0:
        print(f"[INFO] Tutti i k già elaborati in {outdir}, skipping.")
        return


    bw = pyBigWig.open(bw_path)
    bw_chroms = bw.chroms()  # dict: {orig_name: length}

    # --- Filtra records canonici secondo le REGOLE ---
    records: List[Tuple[str, int]] = []
    for nm, L in bw_chroms.items():
        if is_mito(nm):
            # esclusione mitocondrio
            continue
        if looks_non_primary(nm):
            # esclusione non primari/decoy/etc.
            continue
        if L <= 0:
            continue
        records.append((nm, L))

    if not records:
        print(f"[WARN] Nessun contig canonico in {bw_path}.", file=sys.stderr)
        bw.close()
        return

    # --- Costruisci mapping canonico → chr1..chrN ---
    mapping = build_canonical_mapping(records)  # {orig -> chr#}
    if not mapping:
        print(f"[WARN] Mappatura canonica vuota in {bw_path}.", file=sys.stderr)
        bw.close()
        return

    print(f"Mapping canonico: {len(mapping)} contig (su {len(bw_chroms)})")
    for orig, cn in mapping.items():
        print(f"  {orig} -> {cn} (len={bw_chroms[orig]})")

    # ordine finale (per le chiavi di output)
    canon_order = sorted(mapping.values(), key=lambda s: int(s[3:]))  # chr1..chrN
    N = len(canon_order)

    print(f"[INFO] {assembly_id} ({method})  BW={bw_path}")
    print(f"[INFO] OUT={outdir}  Chroms=chr1..chr{N}  k={kmin}..{kmax}")

    # salva mappatura per tracciabilità
    with open(os.path.join(outdir, "canonical_chroms.json"), "w") as fh:
        json.dump({
            "assembly": assembly_id,
            "method": method,
            "source_bigwig": bw_path,
            "mapping": mapping
        }, fh, indent=2)

    # --------- Pass 0: autoscale (se richiesto) ---------
    lo_used, hi_used = lo, hi
    if autoscale != "none":
        print(f"[INFO] autoscale: {autoscale}")
        mins, maxs, samples = [], [], []
        rng = np.random.default_rng(42)
        target_samples = 5_000_000

        canon_orig = [orig for orig, cn in mapping.items() if cn in canon_order]
        for orig in canon_orig:
            L = bw_chroms[orig]
            vals = bw.values(orig, 0, L, numpy=True).astype(float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            v = vals[finite]
            if autoscale == "minmax":
                mins.append(np.min(v)); maxs.append(np.max(v))
            else:
                take = min(v.size, max(1000, int(target_samples / max(1, len(canon_orig)))))
                if v.size > take:
                    idx = rng.choice(v.size, size=take, replace=False)
                    samples.append(v[idx])
                else:
                    samples.append(v)

        if autoscale == "minmax":
            lo_used = float(np.min(mins)) if mins else float("nan")
            hi_used = float(np.max(maxs)) if maxs else float("nan")
        else:
            all_s = np.concatenate(samples) if samples else np.array([], dtype=float)
            mode = "p01p99" if autoscale == "p01p99" else "p05p95"
            lo_used, hi_used = robust_min_max_from_values(all_s, mode)

        print(f"[INFO] autoscale -> lo={lo_used:.6g}, hi={hi_used:.6g}")

    # --------- Pass 1: elaborazione ---------
    # per risalire da chr# all'originale
    inv_map: Dict[str, str] = {cn: orig for orig, cn in mapping.items()}

    for k in range(kmin, kmax + 1, 2):
        assert k >= 3 and k % 2 == 1, "k deve essere dispari e >=3"
        print(f"[INFO] === k={k} ===")
        center = k // 2
        mean_map, first_map, cent_map = {}, {}, {}

        for cn in canon_order:
            orig = inv_map[cn]
            L = bw_chroms[orig]
            vals = bw.values(orig, 0, L, numpy=True).astype(float)

            # normalizzazione centrata su 0 con [lo_used, hi_used]
            vals = rescale_centered_on_zero(vals, lo_used, hi_used)

            mean_win = rolling_mean(vals, k)
            if L >= k:
                first_win = vals[0 : L - k + 1]
                cent_win  = vals[center : L - (k - 1 - center)]
            else:
                first_win = np.empty(0, dtype=float)
                cent_win  = np.empty(0, dtype=float)

            mean_q  = quantize01_to_uint8(mean_win)
            first_q = quantize01_to_uint8(first_win)
            cent_q  = quantize01_to_uint8(cent_win)

            # salva con chiave CANONICA rinominata
            mean_map[cn]  = mean_q
            first_map[cn] = first_q
            cent_map[cn]  = cent_q
        base = os.path.join(outdir, f"k{k}")

        # if os.path.exists(base):
        #     print(f"[WARN] Skipping existing dir: {base}", file=sys.stderr)
        #     continue

        ensure_dir(base)
        np.savez_compressed(os.path.join(base, f"{method}_mean_k{k}.npz"),  **mean_map)
        np.savez_compressed(os.path.join(base, f"{method}_first_k{k}.npz"), **first_map)
        np.savez_compressed(os.path.join(base, f"{method}_center_k{k}.npz"), **cent_map)

        with open(os.path.join(base, f"README_k{k}.txt"), "w") as fh:
            fh.write(
f"""Valori per finestra di lunghezza k sui contig canonici rinominati (chr1..chr{N}):
- mean  : media dei k valori interni alla finestra (NaN -> 255 dopo quantizzazione)
- first : primo valore della finestra
- center: valore al centro (i + {center}; per k pari sarebbe il centrale destro)
Normalizzazione: [lo, hi] = [{lo_used}, {hi_used}] (globali), con NaN preservati
Quantizzazione: uint8 0..100 (passo 0.01); 255 = NaN/sentinella
File:
  {method}_mean_k{k}.npz   (keys=chr1..chr{N}, values=array uint8 len=L-k+1 per ciascun chr)
  {method}_first_k{k}.npz
  {method}_center_k{k}.npz
BW sorgente: {bw_path}
OUT dir    : {base}
Mappatura canonica in: {os.path.join(outdir, "canonical_chroms.json")}
"""
            )

    bw.close()
    print(f"[DONE] {assembly_id} ({method}) → {outdir}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Salva NPZ per con: mean intra-finestra, primo, centrale (quantizzazione 0.01 -> uint8)."
    )
    # modalità: o batch da JSON (solo id) o single da --assembly
    ap.add_argument("--assemblies-json", help="JSON con lista/oggetti; usa solo 'id'.")
    ap.add_argument("--assembly", help="Assembly singolo (es. hg38).")
    ap.add_argument("--method", default="phylop", help="Metodo (default: phylop). Legge data/raw/{assembly}/{assembly}.{method}.bw")

    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--lo", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--autoscale", choices=["none","minmax","p01p99","p05p95"], default="minmax")

    args = ap.parse_args()

    if args.assemblies_json:
        # batch: leggo solo gli id
        with open(args.assemblies_json, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            print("[ERROR] Il JSON deve essere una lista.", file=sys.stderr)
            sys.exit(1)

        ids = []
        for item in data:
            if isinstance(item, dict) and "id" in item:
                ids.append(item["id"])
            elif isinstance(item, str):
                ids.append(item)
            else:
                print(f"[WARN] Elemento ignorato (manca 'id'): {item}", file=sys.stderr)

        print(f"[INFO] Batch: {len(ids)} id trovati.")
        for asm_id in ids:
            run_one(
                assembly_id=asm_id,
                method=args.method,
                kmin=args.kmin,
                kmax=args.kmax,
                autoscale=args.autoscale,
                lo=args.lo,
                hi=args.hi,
            )
        return

    # single
    if not args.assembly:
        print("[ERROR] Specifica --assemblies-json oppure --assembly.", file=sys.stderr)
        sys.exit(1)

    run_one(
        assembly_id=args.assembly,
        method=args.method,
        kmin=args.kmin,
        kmax=args.kmax,
        autoscale=args.autoscale,
        lo=args.lo,
        hi=args.hi,
    )

if __name__ == "__main__":
    main()
