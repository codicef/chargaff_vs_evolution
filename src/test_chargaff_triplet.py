#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    pairs_i, pairs_j = [], []
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

    # counts (uint64) shape: (B, V) con B variabile (dispari)
    counts_coding = data["coding"].astype(np.float64)
    counts_nonc   = data["noncoding"].astype(np.float64)

    # edges può chiamarsi 'edges' (nuovo) o 'bins' (retro-compat: vettore di edge)
    if "edges" in data:
        edges = data["edges"].astype(np.float64)
    elif "bins" in data:
        edges = data["bins"].astype(np.float64)
    else:
        raise KeyError("Nel NPZ manca 'edges'/'bins' (attesi gli estremi dei bin).")

    B, V = counts_coding.shape
    if edges.size != B + 1:
        raise ValueError(f"Incongruenza: edges ha len={edges.size}, ma counts hanno B={B} ⇒ edges deve avere B+1 elementi.")

    # se già salvati, li usiamo; altrimenti li ricaviamo
    centers = data["centers"].astype(np.float64) if "centers" in data else (edges[:-1] + edges[1:]) / 2.0

    # normalizza a frequenze per bin (se bin vuoto rimane 0)
    sums_c = counts_coding.sum(axis=1, keepdims=True)  # (B,1)
    sums_n = counts_nonc.sum(axis=1, keepdims=True)    # (B,1)
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
    rs_c, rs_n, rs_all = [], [], []
    for b in range(B):
        # coding
        x_c = freqs_coding[b, pi]; y_c = freqs_coding[b, pj]
        r_c = pearson_safe(x_c, y_c); rs_c.append(r_c)
        rows.append(("coding", k, float(edges[b]), float(edges[b+1]), pi.size, int(sums_c[b,0]), r_c))
        # noncoding
        x_n = freqs_nonc[b, pi]; y_n = freqs_nonc[b, pj]
        r_n = pearson_safe(x_n, y_n); rs_n.append(r_n)
        rows.append(("noncoding", k, float(edges[b]), float(edges[b+1]), pi.size, int(sums_n[b,0]), r_n))
        # all
        x_a = np.concatenate([x_c, x_n]); y_a = np.concatenate([y_c, y_n])
        r_a = pearson_safe(x_a, y_a); rs_all.append(r_a)
        rows.append(("all", k, float(edges[b]), float(edges[b+1]), pi.size*2, int(sums_c[b,0]+sums_n[b,0]), r_a))

    rs_c = np.array(rs_c, dtype=np.float64)
    rs_n = np.array(rs_n, dtype=np.float64)
    rs_all = np.array(rs_all, dtype=np.float64)

    # salva CSV
    out_k_dir = os.path.join(outdir, f"k{k}")
    ensure_dir(out_k_dir)
    csv_path = os.path.join(out_k_dir, f"chargaff_pearson_k{k}.csv")
    with open(csv_path, "w") as fh:
        fh.write("region,k,bin_lo,bin_hi,n_pairs,n_in_bin,pearson_r\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]},{r[5]},{r[6]:.6f}\n")

    # ---------- GRAFICO ORIGINALE (funziona per B variabile) ----------
    x_centers = centers.tolist()
    n_c = sums_c[:,0].astype(np.float64)
    n_n = sums_n[:,0].astype(np.float64)
    n_a = (sums_c[:,0] + sums_n[:,0]).astype(np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                             gridspec_kw={"height_ratios":[2.0, 1.5], "hspace": 0.08})
    ax = axes[0]
    def rp(x, y):
        if len(x) >= 2 and np.isfinite(y).sum() >= 2:
            r, p = stats.pearsonr(x, y)
            return f"r={r:.3f} (p={p:.3g})"
        return "r=n/a"
    ax.plot(x_centers, rs_c, marker="o", label=f"Coding : {rp(x_centers, rs_c)}")
    ax.plot(x_centers, rs_n, marker="s", label=f"Non-coding : {rp(x_centers, rs_n)}")
    ax.plot(x_centers, rs_all, marker="^", label=f"All : {rp(x_centers, rs_all)}")
    ax.set_ylabel("Pearson r (freq vs RC)")
    ax.set_title(f"Chargaff per bin — k={k}")
    y_min = np.nanmin([np.nanmin(rs_c), np.nanmin(rs_n)])
    ax.set_ylim(max(-1.05, y_min - 0.05), 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    widths = edges[1:] - edges[:-1]
    ax2.bar(centers - widths/4, n_c, width=widths/2, alpha=0.8, label="Coding", align="center")
    ax2.bar(centers + widths/4, n_n, width=widths/2, alpha=0.8, label="Non-coding", align="center")
    ax2.set_yscale("log")
    ax2.set_xlabel("Score bin (0–100)")
    ax2.set_ylabel("# k-mer nel bin (log)")
    ax2.set_xticks(edges[:-1])
    ax2.grid(True, which="both", axis="y", alpha=0.3)
    ax2.legend()

    fig_path = os.path.join(out_k_dir, f"chargaff_pearson_counts_k{k}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)

    # ---------- NUOVO GRAFICO: 3 pannelli + barre, con bin centrale = 0 ----------
    mid_b  = (B - 1) // 2          # indice del bin centrale
    last_b = B - 1

    # 1) CONSERVATION (destra, includendo il centrale). X in [0,1] con 0=centrale
    idx_cons = np.arange(mid_b, last_b + 1)                 # mid..last
    denom_r  = max(1, last_b - mid_b)                       # evita div/0 se B=1
    x_cons   = (idx_cons - mid_b) / denom_r
    rc_cons  = rs_c[idx_cons]; rn_cons = rs_n[idx_cons]; ra_cons = rs_all[idx_cons]
    nc_cons  = n_c[idx_cons];  nn_cons = n_n[idx_cons]

    # 2) ACCELERATION (sinistra, includendo il centrale). X in [0,1] con 0=centrale
    idx_accel = np.arange(mid_b, -1, -1)                    # mid..0
    denom_l   = max(1, mid_b)
    x_accel   = (mid_b - idx_accel) / denom_l
    rc_accel  = rs_c[idx_accel]; rn_accel = rs_n[idx_accel]; ra_accel = rs_all[idx_accel]
    nc_accel  = n_c[idx_accel];  nn_accel = n_n[idx_accel]

    # 3) PRESSURE: media dei r per coppie equidistanti dal centro; conteggi sommati
    max_d = min(mid_b, last_b - mid_b)
    dists = np.arange(0, max_d + 1)
    x_press = dists / max(1, max_d)
    rc_press = np.empty_like(x_press, dtype=float)
    rn_press = np.empty_like(x_press, dtype=float)
    ra_press = np.empty_like(x_press, dtype=float)
    nc_press = np.empty_like(x_press, dtype=float)
    nn_press = np.empty_like(x_press, dtype=float)
    for i, d in enumerate(dists):
        if d == 0:
            rc_press[i] = rs_c[mid_b]; rn_press[i] = rs_n[mid_b]; ra_press[i] = rs_all[mid_b]
            nc_press[i] = n_c[mid_b];  nn_press[i] = n_n[mid_b]
        else:
            rc_press[i] = np.nanmean([rs_c[mid_b-d], rs_c[mid_b+d]])
            rn_press[i] = np.nanmean([rs_n[mid_b-d], rs_n[mid_b+d]])
            ra_press[i] = np.nanmean([rs_all[mid_b-d], rs_all[mid_b+d]])
            nc_press[i] = n_c[mid_b-d] + n_c[mid_b+d]
            nn_press[i] = n_n[mid_b-d] + n_n[mid_b+d]

    # Limite Y condiviso tra i tre pannelli superiori
    all_vals = np.concatenate([
        rc_cons, rn_cons, ra_cons,
        rc_accel, rn_accel, ra_accel,
        rc_press, rn_press, ra_press
    ])
    y_min = np.nanmin(all_vals)
    y_lo = max(-1.05, y_min - 0.05)
    y_hi = 1.05

    fig3, axes3 = plt.subplots(
        2, 3,
        figsize=(12.5, 8.6),
        sharey='row',
        constrained_layout=True,
        sharex='col',
        gridspec_kw={"height_ratios": [1.0, 0.4]}
    )

    def rp_label(x, y, name):
        if len(x) >= 2 and np.isfinite(y).sum() >= 2:
            r, p = stats.pearsonr(x, y)
            return f"{name} : {r:.3f} (p={p:.3g})"
        return f"{name} : n/a"

    def style_panel(ax, x, yc, yn, ya, title, xlabel, show_ylabel=False):
        ax.plot(x, yc, marker="o", label=rp_label(x, yc, "Coding"))
        ax.plot(x, yn, marker="s", label=rp_label(x, yn, "Non-coding"))
        ax.plot(x, ya, marker="^", label=rp_label(x, ya, "All"))
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(xlabel)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        if show_ylabel:
            ax.set_ylabel("Pearson r (freq vs RC)")
        ax.set_aspect('equal', adjustable='box')
        ax.legend()

    def bar_panel(ax, x, n_cod, n_nonc, xlabel):
        if len(x) > 1:
            dx = np.min(np.diff(np.sort(x)))
        else:
            dx = 0.15
        width = min(0.15, dx * 0.2)
        ax.bar(np.array(x) - width/2, n_cod, width=width, alpha=0.85, label="Coding")
        ax.bar(np.array(x) + width/2, n_nonc, width=width, alpha=0.85, label="Non-coding")
        ax.set_yscale("log")
        ax.set_xlim(-0.10, 1.1)
        ax.grid(True, which="both", axis="y", alpha=0.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("# k-mer nel bin (log)")
        # ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend()

    # riga superiore
    style_panel(
        axes3[0,0], x_cons, rc_cons, rn_cons, ra_cons,
        title="Conservation score\n0 = neutral",
        xlabel="Conservation (0→1)",
        show_ylabel=True
    )
    style_panel(
        axes3[0,1], x_accel, rc_accel, rn_accel, ra_accel,
        title="Mutation acceleration score\n0 = neutral",
        xlabel="Acceleration (0→1)"
    )
    style_panel(
        axes3[0,2], x_press, rc_press, rn_press, ra_press,
        title="Overall evolutionary pressure\n0 = neutral",
        xlabel="Normalized distance from center (0→1)"
    )

    # riga inferiore
    bar_panel(axes3[1,0], x_cons,   nc_cons,   nn_cons,   xlabel="Conservation (0→1)")
    bar_panel(axes3[1,1], x_accel,  nc_accel,  nn_accel,  xlabel="Acceleration (0→1)")
    bar_panel(axes3[1,2], x_press,  nc_press,  nn_press,  xlabel="Normalized distance (0→1)")

    fig3_path = os.path.join(out_k_dir, f"chargaff_pearson_threepanel_side_withcounts_k{k}.png")
    fig3.savefig(fig3_path, dpi=160)
    plt.close(fig3)
    print(f"[k={k}] salvato: {fig3_path}")

    # salvo anche un NPZ con le serie trasformate utili per replottare
    np.savez_compressed(
        os.path.join(out_k_dir, f"chargaff_pearson_k{k}.npz"),
        edges=edges,
        centers=centers,
        r_coding=rs_c,
        r_noncoding=rs_n,
        n_in_bin_coding=n_c,
        n_in_bin_noncoding=n_n,
        n_pairs=pi.size,
        # pannelli trasformati
        x_conservation=x_cons, r_coding_conservation=rc_cons, r_noncoding_conservation=rn_cons, r_all_conservation=ra_cons,
        x_acceleration=x_accel, r_coding_acceleration=rc_accel, r_noncoding_acceleration=rn_accel, r_all_acceleration=ra_accel,
        x_pressure=x_press, r_coding_pressure=rc_press, r_noncoding_pressure=rn_press, r_all_pressure=ra_press,
    )

    print(f"[k={k}] salvato: {csv_path}")
    print(f"[k={k}] salvato: {fig_path}")
    print(f"[k={k}] salvato: {fig3_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Pearson(freq(k-mer) vs RC) per B bin variabili; include grafico a 3 pannelli con centro (bin centrale) mappato a 0."
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
    for k in range(args.kmin, args.kmax + 1, 2):
        process_k(k, args.bins_base_dir, args.kmer_index_pattern, args.outdir)

    print(f"[DONE] Output in {args.outdir}")

if __name__ == "__main__":
    main()
