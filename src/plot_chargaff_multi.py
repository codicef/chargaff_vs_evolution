#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, glob
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
        rc_idx[i] = k2i[rc_kmer(km)]
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

# ---------- NEW: helpers skew & plotting ----------
def pair_skew(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return abs((x - y) / (x + y + eps))

def reg_label(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2 or np.var(x) == 0 or np.var(y) == 0:
        return "n/a"
    slope, intercept, r, p, _ = stats.linregress(x, y)
    rho, p_s = stats.spearmanr(x, y)
    return f"OLS: y={slope:.3g}x+{intercept:.3g} | r={r:.3f} (p={p:.2g}) | ρ={rho:.3f} (p={p_s:.2g})"

def line_with_fit(ax, x, y, label=None, marker="o"):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    ax.plot(x, y, marker=marker)
    if x.size >= 2 and np.var(x) > 0 and np.var(y) > 0:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xs = np.array([np.min(x), np.max(x)])
        ys = intercept + slope * xs
        ax.plot(xs, ys, linewidth=2, alpha=0.9, label=label if label else None)
    elif label:
        ax.plot([], [], label=label)  # placeholder per legenda


def scatter_with_reg(ax, x, y, title, xlabel, ylabel="Skew (k-mer vs RC)", alpha=0.08):
    """
    Scatter x vs y con retta di regressione (OLS) + r,p Pearson e rho,p Spearman.
    x,y: array 1D (stesso size). Non usa seaborn.
    """
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    ax.scatter(x, y, s=8, alpha=alpha)
    txt = "n/a"
    if x.size >= 2 and np.var(x) > 0 and np.var(y) > 0:
        # OLS
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.array([np.min(x), np.max(x)])
        ys = intercept + slope * xs
        ax.plot(xs, ys, linewidth=2, alpha=0.9)
        # Spearman
        rho, p_s = stats.spearmanr(x, y)
        txt = f"OLS: y={slope:.3g}x+{intercept:.3g} | r={r:.3f} (p={p:.2g}) | ρ={rho:.3f} (p={p_s:.2g})"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.text(0.02, 0.98, txt, ha="left", va="top", transform=ax.transAxes, fontsize=9)


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

# ---------- core plotting for one freq_matrices file ----------
def process_freq_npz(k, freq_npz_path, kmer_index_pattern, outdir, bins, asm_id, sample_norm):
    if not os.path.exists(freq_npz_path):
        raise FileNotFoundError(f"Matrice frequenze non trovata: {freq_npz_path}")

    if not os.path.exists(kmer_index_pattern.format(k=k, id=asm_id)):
        print(f"[ERROR] Indice k-mer non trovato: {kmer_index_pattern.format(k=k, id=asm_id)}")
        return
    data = np.load(freq_npz_path)

    # counts (uint64) shape: (B, V)
    counts_coding = data["coding"].astype(np.float64)
    counts_nonc   = data["noncoding"].astype(np.float64)

    if "edges" in data:
        edges = data["edges"].astype(np.float64)
    elif "bins" in data:
        edges = data["bins"].astype(np.float64)
    else:
        raise KeyError("Nel NPZ manca 'edges'/'bins' (attesi gli estremi dei bin).")

    B, V = counts_coding.shape
    if edges.size != B + 1:
        raise ValueError(f"Incongruenza: edges len={edges.size}, counts B={B} ⇒ edges deve avere B+1 elementi.")

    centers = data["centers"].astype(np.float64) if "centers" in data else (edges[:-1] + edges[1:]) / 2.0

    # normalizza a frequenze per bin
    sums_c = counts_coding.sum(axis=1, keepdims=True)
    sums_n = counts_nonc.sum(axis=1, keepdims=True)
    sums_a = sums_c + sums_n

    freqs_coding = np.divide(counts_coding, sums_c, out=np.zeros_like(counts_coding), where=(sums_c>0))
    freqs_nonc   = np.divide(counts_nonc,   sums_n, out=np.zeros_like(counts_nonc),   where=(sums_n>0))
    freqs_all    = np.divide(counts_coding + counts_nonc, sums_a, out=np.zeros_like(counts_coding), where=(sums_a>0))

    # k-mer index & RC pairing
    kmer_index_json = kmer_index_pattern.format(k=k, id=asm_id)
    k2i = load_json(kmer_index_json)
    inv = invert_index(k2i)
    rc_idx = build_rc_index(inv)
    pi, pj = unique_pairs(rc_idx)

    # Pearson r per bin
    rows = []
    rs_c, rs_n, rs_all = [], [], []
    n_in_c = []; n_in_n = []
    for b in range(B):
        x_c = freqs_coding[b, pi]; y_c = freqs_coding[b, pj]
        r_c = pearson_safe(x_c, y_c); rs_c.append(r_c)
        rows.append(("coding", k, float(edges[b]), float(edges[b+1]), pi.size, int(sums_c[b,0]), r_c))
        x_n = freqs_nonc[b, pi]; y_n = freqs_nonc[b, pj]
        r_n = pearson_safe(x_n, y_n); rs_n.append(r_n)
        rows.append(("noncoding", k, float(edges[b]), float(edges[b+1]), pi.size, int(sums_n[b,0]), r_n))
        x_a = np.concatenate([x_c, x_n]); y_a = np.concatenate([y_c, y_n])
        r_a = pearson_safe(x_a, y_a); rs_all.append(r_a)
        rows.append(("all", k, float(edges[b]), float(edges[b+1]), pi.size*2, int(sums_c[b,0]+sums_n[b,0]), r_a))
        n_in_c.append(int(sums_c[b,0])); n_in_n.append(int(sums_n[b,0]))

    rs_c = np.array(rs_c, dtype=np.float64)
    rs_n = np.array(rs_n, dtype=np.float64)
    rs_all = np.array(rs_all, dtype=np.float64)
    n_c = np.array(n_in_c, dtype=np.float64)
    n_n = np.array(n_in_n, dtype=np.float64)

    # salva CSV
    ensure_dir(outdir)
    csv_path = os.path.join(outdir, f"chargaff_pearson_k{k}_{sample_norm}sample.csv")
    with open(csv_path, "w") as fh:
        fh.write("region,k,bin_lo,bin_hi,n_pairs,n_in_bin,pearson_r\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]},{r[5]},{r[6]:.6f}\n")


    # ---------- FIXED: scatter skew normalizzato (3x3) ----------
    mid_b  = (B - 1) // 2
    last_b = B - 1
    denom_r = max(1, last_b - mid_b)  # destra (conservation)
    denom_l = max(1, mid_b)           # sinistra (acceleration)
    max_d   = min(mid_b, last_b - mid_b)  # pressure (distanza)

    def make_xy(freqs):
        Xc, Yc = [], []   # Conservation: b >= mid_b
        Xa, Ya = [], []   # Acceleration: b <= mid_b
        Xp, Yp = [], []   # Pressure: distanza simmetrica dal centro
        for b in range(B):
            sk = pair_skew(freqs[b, pi], freqs[b, pj])
            if sk.size == 0:
                continue
            # conservation half
            if b >= mid_b:
                x = (b - mid_b) / denom_r
                Xc.extend([x] * sk.size)
                Yc.extend(sk.tolist())
            # acceleration half
            if b <= mid_b:
                x = (mid_b - b) / denom_l
                Xa.extend([x] * sk.size)
                Ya.extend(sk.tolist())
            # pressure (distanza)
            d = abs(b - mid_b)
            if d <= max_d and max_d > 0:
                x = d / max_d
                Xp.extend([x] * sk.size)
                Yp.extend(sk.tolist())
            elif max_d == 0:
                # caso B=1: tutto al centro
                Xp.extend([0.0] * sk.size)
                Yp.extend(sk.tolist())
        return (np.array(Xc), np.array(Yc),
                np.array(Xa), np.array(Ya),
                np.array(Xp), np.array(Yp))

    Xc_c, Yc_c, Xa_c, Ya_c, Xp_c, Yp_c = make_xy(freqs_coding)
    Xc_n, Yc_n, Xa_n, Ya_n, Xp_n, Yp_n = make_xy(freqs_nonc)
    Xc_a, Yc_a, Xa_a, Ya_a, Xp_a, Yp_a = make_xy(freqs_all)

    fig_sc, ax_sc = plt.subplots(3, 3, figsize=(12.5, 10.0), sharey=True)
    fig_sc.subplots_adjust(hspace=0.3, wspace=0.25)

    def scatter_fit(ax, x, y, title, xlabel, show_ylabel=False):
        mask = np.isfinite(x) & np.isfinite(y)
        xv = x[mask]; yv = y[mask]
        ax.scatter(xv, yv, s=6, alpha=0.05)
        if xv.size >= 2 and np.var(xv) > 0 and np.var(yv) > 0:
            slope, intercept, r, p, _ = stats.linregress(xv, yv)
            xs = np.linspace(0, 1, 100)
            ax.plot(xs, intercept + slope*xs, linewidth=2)
            rho, p_s = stats.spearmanr(xv, yv)
            ax.text(0.02, 0.98,
                    f"r={r:.3f} (p={p:.2g}) | ρ={rho:.3f} (p={p_s:.2g})",
                    ha="left", va="top", transform=ax.transAxes, fontsize=8)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, ls="--", lw=0.8)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if show_ylabel: ax.set_ylabel("Skew (k vs RC)")
        ax.grid(True, alpha=0.2)

    # coding
    scatter_fit(ax_sc[0,0], Xc_c, Yc_c, "Coding — Conservation", "Conservation (0→1)", True)
    scatter_fit(ax_sc[0,1], Xa_c, Ya_c, "Coding — Acceleration", "Acceleration (0→1)")
    scatter_fit(ax_sc[0,2], Xp_c, Yp_c, "Coding — Pressure", "Distance from center (0→1)")

    # noncoding
    scatter_fit(ax_sc[1,0], Xc_n, Yc_n, "Noncoding — Conservation", "Conservation (0→1)", True)
    scatter_fit(ax_sc[1,1], Xa_n, Ya_n, "Noncoding — Acceleration", "Acceleration (0→1)")
    scatter_fit(ax_sc[1,2], Xp_n, Yp_n, "Noncoding — Pressure", "Distance from center (0→1)")

    # all
    scatter_fit(ax_sc[2,0], Xc_a, Yc_a, "All — Conservation", "Conservation (0→1)", True)
    scatter_fit(ax_sc[2,1], Xa_a, Ya_a, "All — Acceleration", "Acceleration (0→1)")
    scatter_fit(ax_sc[2,2], Xp_a, Yp_a, "All — Pressure", "Distance from center (0→1)")

    fig_sc_path = os.path.join(outdir, f"chargaff_skew_scatter_threebythree_k{k}_{sample_norm}sample.png")
    fig_sc.savefig(fig_sc_path, dpi=160)
    plt.close(fig_sc)



    # ---------- GRAFICO 2 pannelli ----------
    x_centers = centers.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                             gridspec_kw={"height_ratios":[2.0, 1.5], "hspace": 0.08})
    ax = axes[0]
    def rp(x, y):
        y_ok = np.asarray(y)
        mask = np.isfinite(y_ok)
        xv = np.asarray(x)[mask]
        yv = y_ok[mask]
        if len(xv) >= 2 and len(yv) >= 2 and np.var(yv) > 0 and np.var(xv) > 0:
            r, p = stats.pearsonr(xv, yv)
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
    ax.set_xticks(edges[:-1])           # ticks anche sopra
    ax.tick_params(labelbottom=True)    # e mostra etichette

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

    fig_path = os.path.join(outdir, f"chargaff_pearson_counts_k{k}_{sample_norm}sample.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)

    # ---------- GRAFICO 3 pannelli + barre (mappando il bin centrale a 0) ----------
    B = len(centers)
    mid_b  = (B - 1) // 2
    last_b = B - 1

    idx_cons = np.arange(mid_b, last_b + 1)
    denom_r  = max(1, last_b - mid_b)
    x_cons   = (idx_cons - mid_b) / denom_r
    rc_cons  = rs_c[idx_cons]; rn_cons = rs_n[idx_cons]; ra_cons = rs_all[idx_cons]
    nc_cons  = n_c[idx_cons];  nn_cons = n_n[idx_cons]

    idx_accel = np.arange(mid_b, -1, -1)
    denom_l   = max(1, mid_b)
    x_accel   = (mid_b - idx_accel) / denom_l
    rc_accel  = rs_c[idx_accel]; rn_accel = rs_n[idx_accel]; ra_accel = rs_all[idx_accel]
    nc_accel  = n_c[idx_accel];  nn_accel = n_n[idx_accel]

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
        y_ok = np.asarray(y)
        mask = np.isfinite(y_ok)
        xv = np.asarray(x)[mask]
        yv = y_ok[mask]
        if len(xv) >= 2 and len(yv) >= 2 and np.var(yv) > 0 and np.var(xv) > 0:
            r, p = stats.pearsonr(xv, yv)
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
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend()

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

    bar_panel(axes3[1,0], x_cons,   nc_cons,   nn_cons,   xlabel="Conservation (0→1)")
    bar_panel(axes3[1,1], x_accel,  nc_accel,  nn_accel,  xlabel="Acceleration (0→1)")
    bar_panel(axes3[1,2], x_press,  nc_press,  nn_press,  xlabel="Normalized distance (0→1)")

    fig3_path = os.path.join(outdir, f"chargaff_pearson_threepanel_side_withcounts_k{k}_{sample_norm}sample.png")
    fig3.savefig(fig3_path, dpi=160)
    plt.close(fig3)

    # NPZ con serie trasformate
    np.savez_compressed(
        os.path.join(outdir, f"chargaff_pearson_k{k}_{sample_norm}sample.npz"),
        edges=edges,
        centers=centers,
        r_coding=rs_c,
        r_noncoding=rs_n,
        n_in_bin_coding=n_c,
        n_in_bin_noncoding=n_n,
        n_pairs=len(pi),
        x_conservation=x_cons, r_coding_conservation=rc_cons, r_noncoding_conservation=rn_cons, r_all_conservation=ra_cons,
        x_acceleration=x_accel, r_coding_acceleration=rc_accel, r_noncoding_acceleration=rn_accel, r_all_acceleration=ra_accel,
        x_pressure=x_press, r_coding_pressure=rc_press, r_noncoding_pressure=rn_press, r_all_pressure=ra_press,
    )

    print(f"[k={k}] salvato: {csv_path}")
    print(f"[k={k}] salvato: {fig_path}")
    print(f"[k={k}] salvato: {fig3_path}")

# ---------- discovery utilities ----------
def discover_methods_for_id(processed_root, asm_id):
    """Se method=auto, trova directory tipo {method}_scores sotto data/processed/{id}/"""
    base = os.path.join(processed_root, asm_id)
    if not os.path.isdir(base):
        return []
    methods = []
    for name in os.listdir(base):
        if name.endswith("_scores") and os.path.isdir(os.path.join(base, name)):
            methods.append(name.replace("_scores", ""))
    return sorted(set(methods))

def discover_kinds_for_k(processed_root, asm_id, method, k, bins, sample_norm):
    """Trova i kinds presenti per un certo k: cartelle in bins_scores/k{k}/* che contengono freq_matrices_k{k}.npz"""
    base = os.path.join(processed_root, asm_id, f"{method}_scores", "bins_scores", f"k{k}")
    kinds = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            freq_path = os.path.join(base, name, f"freq_matrices_k{k}_{bins}bins_{sample_norm}sample.npz")
            if os.path.exists(freq_path):
                kinds.append(name)
    if len(kinds) == 0:
        print(f"[WARN] {asm_id} {method} k={k}: nessun kind trovato in {base}/")
        print(f"[WARN] Paths: {glob.glob(os.path.join(base, '*', f'freq_matrices_k{k}_*'))}")

    return sorted(set(kinds))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Plot Chargaff (Pearson freq vs RC) per assemblies/methods/kinds. Legge assemblies.json, scopre metodi e kinds dalle cartelle."
    )
    ap.add_argument("--assemblies-json", required=True,
                    help="Lista assembly con campo 'id'.")
    ap.add_argument("--ids", default="",
                    help="Subset comma-separated di id da processare (default: tutti).")
    ap.add_argument("--method", default="phylop",
                    help="Metodo degli score (es. phylop). Usa 'auto' per scoprire tutti i metodi disponibili.")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--processed-root", default="data/processed",
                    help="Radice input: data/processed/{id}/{method}_scores/bins_scores/k{k}/{kind}/freq_matrices_k{k}_{bins}bins.npz")
    ap.add_argument("--kmer-index-pattern", default="data/processed/{id}/kmers_info/k{k}/kmer_index_k{k}.json",
                    help="Percorso JSON indice k-mer (usa {k}).")
    ap.add_argument("--results-root", default="results",
                    help="Dove salvare i risultati: results/{id}/{method}/{kind}/k{k}/")
    ap.add_argument("--bins", type=int, default=11)

    args = ap.parse_args()


    assemblies = load_json(args.assemblies_json)
    wanted_ids = set([s.strip() for s in args.ids.split(",") if s.strip()]) if args.ids else None

    ids = []
    for entry in assemblies:
        asm_id = entry.get("id")
        if asm_id and (not wanted_ids or asm_id in wanted_ids):
            ids.append(asm_id)

    if not ids:
        raise SystemExit("Nessun assembly da processare (controlla --assemblies-json e/o --ids).")

    for asm_id in ids:
        if args.method == "auto":
            methods = discover_methods_for_id(args.processed_root, asm_id)
            if not methods:
                print(f"[SKIP] {asm_id}: nessun metodo trovato in {args.processed_root}/{asm_id}/")
                continue
        else:
            methods = [args.method]

        for method in methods:
            print(f"\n=== Assembly: {asm_id} | method: {method} ===")
            for k in range(args.kmin, args.kmax + 1, 2):
                    for sample_norm in [False, True]:
                        # check freq npz_path


                            kinds = discover_kinds_for_k(args.processed_root, asm_id, method, k, bins=args.bins, sample_norm=sample_norm)
                            if not kinds:
                                print(f"[WARN] {asm_id} {method} k={k}: nessun kind trovato.")
                                continue
                            for kind in kinds:
                                print(f"[RUN] {asm_id} | {method} | kind={kind} | k={k}")

                                base_dir = os.path.join(args.processed_root, asm_id, f"{method}_scores", "bins_scores", f"k{k}", kind)
                                if os.path.exists(os.path.join(base_dir, f"freq_matrices_k{k}_{args.bins}bins_{sample_norm}sample.npz")):
                                    freq_npz_path = os.path.join(base_dir, f"freq_matrices_k{k}_{args.bins}bins_{sample_norm}sample.npz")
                                    outdir = os.path.join(args.results_root, asm_id, method, kind, f"k{k}")
                                    if os.path.exists(os.path.join(outdir, f"chargaff_pearson_k{k}.png")):
                                        print(f"[SKIP] {asm_id} | {method} | kind={kind} | k={k} (già esistente)")
                                        continue
                                    print(f"      sample_norm={sample_norm} (found)")
                                    process_freq_npz(k, freq_npz_path, args.kmer_index_pattern, outdir, args.bins, asm_id, sample_norm)
                                else:
                                    print(f"[SKIP] {asm_id} | {method} | kind={kind} | k={k} | sample_norm={sample_norm} (freq npz mancante)")

    print("\n[DONE] Tutti i plot generati.")

if __name__ == "__main__":
    main()
