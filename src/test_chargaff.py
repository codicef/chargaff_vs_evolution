#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)

def invert_index(kmer_to_idx):
    V = max(kmer_to_idx.values()) + 1
    inv = [""] * V
    for km, idx in kmer_to_idx.items():
        inv[idx] = km
    return inv

def rc_kmer(kmer: str) -> str:
    comp = str.maketrans("ACGT", "TGCA")
    return kmer.translate(comp)[::-1]

def build_rc_index(inv_kmer):
    """rc_idx[i] = indice del reverse complement di inv_kmer[i]"""
    k2i = {km: i for i, km in enumerate(inv_kmer)}
    rc_idx = np.empty(len(inv_kmer), dtype=np.int64)
    for i, km in enumerate(inv_kmer):
        rck = rc_kmer(km)
        rc_idx[i] = k2i[rck]
    return rc_idx

def unique_pairs(rc_idx: np.ndarray):
    """coppie (i, j=rc[i]) senza doppioni; palindromi come (i,i)."""
    seen = np.zeros(rc_idx.size, dtype=bool)
    pairs_i = []
    pairs_j = []
    for i in range(rc_idx.size):
        if seen[i]:
            continue
        j = int(rc_idx[i])
        if j == i:
            pairs_i.append(i); pairs_j.append(j)
            seen[i] = True
        elif j > i:
            pairs_i.append(i); pairs_j.append(j)
            seen[i] = True; seen[j] = True
        else:
            continue
    return np.array(pairs_i, dtype=np.int64), np.array(pairs_j, dtype=np.int64)

def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; np.nan se varianza nulla o <2 punti."""
    if x.size < 2:
        return float("nan")
    vx = np.var(x); vy = np.var(y)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    x_mean = x.mean(); y_mean = y.mean()
    xm = x - x_mean; ym = y - y_mean
    num = np.dot(xm, ym)
    den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
    if den == 0.0:
        return float("nan")
    return float(num / den)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- main logic ----------
def process_k(k, base_dir, kmer_index_pattern, outdir):
    freq_npz_path = os.path.join(base_dir, f"k{k}", f"freq_matrices_k{k}.npz")
    if not os.path.exists(freq_npz_path):
        raise FileNotFoundError(f"Matrice frequenze non trovata: {freq_npz_path}. Esegui prima kmer_bin_counts.py.")
    data = np.load(freq_npz_path)
    # counts (uint64) shape: (10, V)
    counts_coding = data["coding"].astype(np.float64)
    counts_nonc   = data["noncoding"].astype(np.float64)
    bin_edges = data["bins"]  # [0,10,20,...,100]

    # average before 40
    # counts_coding[0,:] = (counts_coding[0,:] + counts_coding[1,:] + counts_coding[2,:] + counts_coding[3,:] + counts_coding[4,:]) / 5.0
    # counts_nonc[0,:]   = (counts_nonc[0,:]   + counts_nonc[1,:]   + counts_nonc[2,:]   + counts_nonc[3,:]   + counts_nonc[4,:]) / 5.0

    # counts_coding = counts_coding[0] + counts_coding[5:]
    # counts_nonc   = counts_nonc[0]   + counts_nonc[5:]

    B, V = counts_coding.shape

    # normalizza a frequenze per bin (se bin vuoto rimane 0)
    sums_c = counts_coding.sum(axis=1, keepdims=True)  # (10,1) => #kmer nel bin, coding
    sums_n = counts_nonc.sum(axis=1, keepdims=True)    # (10,1) => #kmer nel bin, noncoding
    sums_a = sums_c + sums_n

    freqs_coding = np.divide(counts_coding, sums_c, out=np.zeros_like(counts_coding), where=(sums_c>0))
    freqs_nonc   = np.divide(counts_nonc,   sums_n, out=np.zeros_like(counts_nonc),   where=(sums_n>0))
    freqs_all    = np.divide(counts_coding + counts_nonc, sums_a, out=np.zeros_like(counts_coding), where=(sums_a>0))

    # k-mer index & reverse-complement pairing
    kmer_index_json = kmer_index_pattern.format(k=k)
    k2i = load_json(kmer_index_json)
    inv = invert_index(k2i)
    rc_idx = build_rc_index(inv)
    pi, pj = unique_pairs(rc_idx)  # indici unici

    # calcola Pearson r per ogni bin e regione
    rows = []
    rs_c = []  # per plot
    rs_n = []
    rs_all = []
    for b in range(B):
        # coding
        x_c = freqs_coding[b, pi]
        y_c = freqs_coding[b, pj]
        r_c = pearson_safe(x_c, y_c)
        rs_c.append(r_c)
        rows.append(("coding", k, int(bin_edges[b]), int(bin_edges[b+1]), pi.size, int(sums_c[b,0]), r_c))
        # noncoding
        x_n = freqs_nonc[b, pi]
        y_n = freqs_nonc[b, pj]
        r_n = pearson_safe(x_n, y_n)
        rs_n.append(r_n)
        rows.append(("noncoding", k, int(bin_edges[b]), int(bin_edges[b+1]), pi.size, int(sums_n[b,0]), r_n))


        # all
        x_a = np.concatenate([x_c, x_n])
        y_a = np.concatenate([y_c, y_n])
        r_a = pearson_safe(x_a, y_a)
        rs_all.append(r_a)
        rows.append(("all", k, int(bin_edges[b]), int(bin_edges[b+1]), pi.size*2, int(sums_c[b,0]+sums_n[b,0]), r_a))


    # salva CSV (aggiungo n_bin_total)
    out_k_dir = os.path.join(outdir, f"k{k}")
    ensure_dir(out_k_dir)
    csv_path = os.path.join(out_k_dir, f"chargaff_pearson_k{k}.csv")
    with open(csv_path, "w") as fh:
        fh.write("region,k,bin_lo,bin_hi,n_pairs,n_in_bin,pearson_r\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]:.6f}\n")

    # --- multiplot: Pearson (top) + counts per bin (bottom, log scale) ---
    x_centers = [(int(bin_edges[b]) + int(bin_edges[b+1]))/2 for b in range(B)]
    n_c = sums_c[:,0].astype(np.float64)
    n_n = sums_n[:,0].astype(np.float64)
    n_a = sums_a[:,0].astype(np.float64)


    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                             gridspec_kw={"height_ratios":[2.0, 1.5], "hspace": 0.08})

    from scipy import stats
    # top: Pearson
    ax = axes[0]
    ax.plot(x_centers, rs_c, marker="o", label=f"Coding : {stats.pearsonr(x_centers, rs_c)[0]:.3f} (p={stats.pearsonr(x_centers, rs_c)[1]:.3g})")
    ax.plot(x_centers, rs_n, marker="s", label=f"Non-coding : {stats.pearsonr(x_centers, rs_n)[0]:.3f} (p={stats.pearsonr(x_centers, rs_n)[1]:.3g})")
    ax.plot(x_centers, rs_all, marker="^", label=f"All : {stats.pearsonr(x_centers, rs_all)[0]:.3f} (p={stats.pearsonr(x_centers, rs_all)[1]:.3g})")
    ax.set_ylabel("Pearson r (freq vs RC)")
    ax.set_title(f"Chargaff per bin — k={k}")
    ax.set_ylim(max(-1.05, np.nanmin([rs_c, rs_n]) - 0.05), 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # bottom: counts
    ax2 = axes[1]
    width = 4.0  # metà-bin per due barre affiancate
    ax2.bar(np.array(x_centers) - width/2, n_c, width=width, alpha=0.8, label="Coding")
    ax2.bar(np.array(x_centers) + width/2, n_n, width=width, alpha=0.8, label="Non-coding")
    ax2.set_yscale("log")
    ax2.set_xlabel("Score bin (0–100)")
    ax2.set_ylabel("# k-mer nel bin (log)")
    ax2.set_xticks(bin_edges[:-1])
    ax2.grid(True, which="both", axis="y", alpha=0.3)
    ax2.legend()

    fig_path = os.path.join(out_k_dir, f"chargaff_pearson_counts_k{k}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)

    # salvo anche un NPZ comodo per il grafico
    np.savez_compressed(
        os.path.join(out_k_dir, f"chargaff_pearson_k{k}.npz"),
        bins=bin_edges,
        r_coding=np.array(rs_c, dtype=np.float64),
        r_noncoding=np.array(rs_n, dtype=np.float64),
        n_in_bin_coding=n_c,
        n_in_bin_noncoding=n_n,
        n_pairs=pi.size,
    )

    print(f"[k={k}] salvato: {csv_path}  &  {fig_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Pearson(freq(k-mer) vs RC) per 10 bin (0–10,...,90–100), coding vs noncoding; multiplot con conte per bin."
    )
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--bins-base-dir", default="./kmer_bin_stats_center",
                    help="Cartella con k*/freq_matrices_k*.npz (da kmer_bin_counts.py).")
    ap.add_argument("--kmer-index-pattern", default="data/kmer_out/k{k}/kmer_index_k{k}.json",
                    help="Percorso JSON indice k-mer (usa {k}).")
    ap.add_argument("--outdir", required=True, help="Dove salvare CSV e figure.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for k in range(args.kmin, args.kmax + 1):
        process_k(k, args.bins_base_dir, args.kmer_index_pattern, args.outdir)

    print(f"[DONE] Output in {args.outdir}")

if __name__ == "__main__":
    main()
