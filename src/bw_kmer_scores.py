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
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Per k=3..7 salva NPZ per chr1..22 con: mean intra-finestra, primo valore, valore centrale. Quantizzazione 0.01 -> uint8."
    )
    ap.add_argument("--bw", required=True, help="BigWig (es. phastCons100way.bw)")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lo", type=float, default=0.0, help="Lower bound per la normalizzazione (usato se --autoscale=none)")
    ap.add_argument("--hi", type=float, default=1.0, help="Upper bound per la normalizzazione (usato se --autoscale=none)")
    ap.add_argument("--autoscale", choices=["none","minmax","p01p99","p05p95"], default="minmax",
                    help="Stima automatica di lo/hi dal BW (globale su chr1..22).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bw = pyBigWig.open(args.bw)

    bw_chroms = bw.chroms()

    print(bw_chroms)
    # usa solo chr1..22
    chroms = [f"{i}" for i in range(1, 23) if f"chr{i}" in bw_chroms]
    prefix = "chr"
    if not chroms:
        print(f"Fall back checking for chr1..22 without 'chr' prefix.")
        chroms = [f"{i}" for i in range(1, 23) if f"{i}" in bw_chroms]
        prefix = ""
    if not chroms:
        raise ValueError("Nessun cromosoma chr1..22 trovato nel BigWig.")

    print(f"[INFO] Chroms: {len(chroms)} (chr1..22); k={args.kmin}..{args.kmax}")

    # --------- Pass 0: autoscale (se richiesto) ---------
    lo_used, hi_used = args.lo, args.hi
    if args.autoscale != "none":
        print(f"[info] autoscale: {args.autoscale} (scan iniziale per stimare lo/hi)")
        mins, maxs = [], []
        # per percentili serve raccogliere i valori; per non esplodere la ram
        # facciamo un reservoir sampling semplice su ogni cromosoma.
        samples = []
        rng = np.random.default_rng(42)
        target_samples = 5_000_000  # ~5m punti totali
        for chrom in chroms:
            l = bw_chroms[prefix + chrom]
            print(f"  {prefix + chrom}: L={l:,} ...")
            vals = bw.values(prefix + chrom, 0, l, numpy=True)  # np.ndarray con nan
            vals = np.asarray(vals, dtype=float)
            print(f"nan %: {np.mean(np.isnan(vals))*100:.2f}%")
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            v = vals[finite]
            if args.autoscale == "minmax":
                mins.append(np.min(v))
                maxs.append(np.max(v))
            else:
                # sampling proporzionale per percentili robusti
                take = min(v.size, max(1000, int(target_samples / max(1, len(chroms)))))
                if v.size > take:
                    idx = rng.choice(v.size, size=take, replace=false)
                    samples.append(v[idx])
                else:
                    samples.append(v)
        if args.autoscale == "minmax":
            lo_used = float(np.min(mins)) if mins else np.nan
            hi_used = float(np.max(maxs)) if maxs else np.nan
        else:
            all_s = np.concatenate(samples) if samples else np.array([], dtype=float)
            lo_used, hi_used = robust_min_max_from_values(all_s, "p01p99" if args.autoscale=="p01p99" else "p05p95")

        print(f"[info] autoscale -> lo={lo_used:.6g}, hi={hi_used:.6g}")

    # --------- Pass 1: elaborazione ---------
    for k in range(args.kmin, args.kmax + 1):
        print(f"[INFO] Working on kmer k={k} ...")
        center =  k // 2  # per k pari prende il centrale destro
        mean_map  = {}
        first_map = {}
        cent_map  = {}

        for chrom in chroms:
            print(f"[k={k}] Processing {chrom} ...")
            L = bw_chroms[prefix + chrom]
            vals = bw.values(prefix + chrom, 0, L, numpy=True)  # np.ndarray con NaN
            vals = np.asarray(vals, dtype=float)

            # normalizzazione globale tra lo_used/hi_used
            # vals = rescale(vals, lo_used, hi_used)
            vals = rescale_centered_on_zero(vals, lo_used, hi_used)

            # mean intra-finestra
            mean_win = rolling_mean(vals, k)

            # primo valore e centrale della finestra (len = L-k+1)
            if L >= k:
                first_win = vals[0 : L - k + 1]
                cent_win  = vals[center : L - (k - 1 - center)]
            else:
                first_win = np.empty(0, dtype=float)
                cent_win  = np.empty(0, dtype=float)

            # quantizzazione 0..1 -> uint8 (0..100), NaN->255
            mean_q  = quantize01_to_uint8(mean_win)
            first_q = quantize01_to_uint8(first_win)
            cent_q  = quantize01_to_uint8(cent_win)

            mean_map['chr' + chrom]  = mean_q
            first_map['chr' + chrom] = first_q
            cent_map['chr' + chrom]  = cent_q

            print(f"[k={k}] {chrom}: L={L:,} -> win={mean_q.size:,}")

        # salva tre NPZ per questo k
        base = os.path.join(args.outdir, f"k{k}")
        os.makedirs(base, exist_ok=True)
        np.savez_compressed(os.path.join(base, f"bw_mean_k{k}.npz"),  **mean_map)
        np.savez_compressed(os.path.join(base, f"bw_first_k{k}.npz"), **first_map)
        np.savez_compressed(os.path.join(base, f"bw_center_k{k}.npz"), **cent_map)

        with open(os.path.join(base, f"README_k{k}.txt"), "w") as fh:
            fh.write(
f"""Valori per finestra di lunghezza k (solo chr1..22):
- mean  : media dei k valori interni alla finestra (NaN se la finestra contiene NaN)
- first : primo valore della finestra (posizione i)
- center: valore al centro (i + {center}; per k pari è il centrale destro)
Normalizzazione: lineare su [lo, hi] = [{lo_used}, {hi_used}] (globali), con NaN preservati
Quantizzazione: uint8 0..100 (passo 0.01); 255 = NaN/sentinella
File:
  bw_mean_k{k}.npz   (keys=chr1..22, values=array uint8 len=L-k+1)
  bw_first_k{k}.npz
  bw_center_k{k}.npz
"""
            )

    bw.close()
    print(f"[DONE] NPZ salvati in {args.outdir}. Normalizzazione su lo={lo_used} hi={hi_used} (autoscale={args.autoscale}).")

if __name__ == "__main__":
    main()
