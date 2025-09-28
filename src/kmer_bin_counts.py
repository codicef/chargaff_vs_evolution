#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, math
import numpy as np

# ----------------------------
# helper
# ----------------------------
def load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def write_bin_csv(out_csv, counts, inv_kmer):
    """Scrive CSV kmer,index,count,frequency per un bin."""
    total = int(counts.sum())
    with open(out_csv, "w") as fh:
        fh.write("kmer,index,count,frequency\n")
        if total == 0:
            return
        nz = np.nonzero(counts)[0]
        if nz.size == 0:
            return
        # ordino per conteggio desc
        order = nz[np.argsort(counts[nz])[::-1]]
        for i in order:
            km = inv_kmer[i] if 0 <= i < len(inv_kmer) and inv_kmer[i] else f"IDX_{i}"
            c = int(counts[i])
            f = c / total if total > 0 else 0.0
            fh.write(f"{km},{i},{c},{f:.8f}\n")

def invert_index(kmer_to_idx):
    V = max(kmer_to_idx.values()) + 1
    inv = [""] * V
    for km, idx in kmer_to_idx.items():
        inv[idx] = km
    return inv

def fmt_range(lo, hi):
    """Stringa pulita per range cartella (interi se possibile, altrimenti 1 decimale)."""
    def _fmt(x):
        if abs(x - round(x)) < 1e-6:
            return f"{int(round(x)):02d}"
        else:
            return f"{x:05.1f}"
    return f"{_fmt(lo)}-{_fmt(hi)}"

# ----------------------------
# core pass (O(n))
# ----------------------------
def process_k(
    k,
    kmer_npz_path,
    kmer_index_json,
    scores_npz_path,
    coding_npz_path,
    outdir_for_k,
    bins,                      # num bin (dispari)
    sample_norm=False,         # <-- nuovo
    rng=None,                  # numpy Generator (per riproducibilità)
):
    print(f"[INFO] k={k}  bins={bins}  sample_norm={sample_norm}")
    if bins % 2 == 0 or bins <= 0:
        raise ValueError("--bins deve essere un intero dispari > 0")

    # carica dati
    kmer_npz   = np.load(kmer_npz_path)          # chr -> uint8/uint16 (len=L-k+1)
    scores_npz = np.load(scores_npz_path)        # chr -> uint8 (0..100, 255=invalid), len=L-k+1
    coding_npz = np.load(coding_npz_path)        # chr -> bool (len=L)
    k2i = load_json(kmer_index_json)
    inv_kmer = invert_index(k2i)
    V = 4 ** k
    B = bins

    # definizione bin: B dispari, larghezza uniforme, centro su 50
    edges = np.linspace(0.0, 100.0, B + 1, dtype=np.float64)
    w = 100.0 / B
    centers = (edges[:-1] + edges[1:]) / 2.0

    freqs_coding    = np.zeros((B, V), dtype=np.uint64)
    freqs_noncoding = np.zeros((B, V), dtype=np.uint64)

    center = k // 2

    # cromosomi in comune (autosomes)
    chroms = [f"chr{i}" for i in range(1, 23)
              if f"chr{i}" in kmer_npz.files and f"chr{i}" in scores_npz.files and f"chr{i}" in coding_npz.files]

    # ------------- PASS 1: conta per bin (per calcolare la cap) -------------
    # totali di posizioni valide per bin e regione su tutto il genoma
    totals_coding = np.zeros(B, dtype=np.int64)
    totals_noncoding = np.zeros(B, dtype=np.int64)

    for chrom in chroms:
        idx_arr = kmer_npz[chrom]            # shape Lk
        scr_q   = scores_npz[chrom]          # shape Lk, uint8 (0..100 o 255)
        flags   = coding_npz[chrom]          # shape L
        Lk = idx_arr.size
        if Lk == 0:
            continue

        sent = np.iinfo(idx_arr.dtype).max
        valid = (scr_q <= 100) & (idx_arr != sent)

        if center + Lk > flags.size:
            maxLk = flags.size - center
            if maxLk <= 0:
                continue
            idx_arr = idx_arr[:maxLk]
            scr_q   = scr_q[:maxLk]
            valid   = valid[:maxLk]
            Lk      = maxLk

        region_is_coding = flags[center : center + Lk]
        scr_q_float = scr_q.astype(np.float64, copy=False)
        bin_idx = (scr_q_float / w).astype(np.int64, copy=False)
        bin_idx = np.clip(bin_idx, 0, B - 1)

        # maschere
        m_valid_coding    = valid &  region_is_coding
        m_valid_noncoding = valid & (~region_is_coding)

        # aggiorna conteggi per-bin (senza accumulare k-mer)
        if np.any(m_valid_coding):
            bc = bin_idx[m_valid_coding]
            # bincount con lunghezza B
            totals_coding += np.bincount(bc, minlength=B)
        if np.any(m_valid_noncoding):
            bn = bin_idx[m_valid_noncoding]
            totals_noncoding += np.bincount(bn, minlength=B)

    # cap = dimensione del bin più a destra (B-1), separato per regione
    cap_c = int(totals_coding[B-1])
    cap_n = int(totals_noncoding[B-1])

    # target per ogni bin
    target_c = np.minimum(totals_coding, cap_c) if sample_norm else totals_coding
    target_n = np.minimum(totals_noncoding, cap_n) if sample_norm else totals_noncoding

    # quote residue globali da riempire nel PASS 2
    rem_c = target_c.astype(np.int64).copy()
    rem_n = target_n.astype(np.int64).copy()

    # ------------- PASS 2: accumulo (con eventuale downsampling) -------------
    def accumulate_selected(sel_positions_mask, idx_arr_local, target_matrix, bin_idx_local, bins_to_touch, rem_vec):
        """Accumulatore che seleziona per bin rispettando la quota residua rem_vec."""
        if not np.any(sel_positions_mask):
            return
        # per efficienza, lavora solo sui bin che compaiono
        present_bins = np.unique(bin_idx_local[sel_positions_mask])
        # restringi ai bin di interesse (es. se vogliamo solo quelli dove mask True)
        if bins_to_touch is not None:
            present_bins = present_bins[np.isin(present_bins, bins_to_touch)]

        for b in present_bins:
            if rem_vec[b] <= 0:
                continue
            # posizioni in questo bin
            in_b = sel_positions_mask & (bin_idx_local == b)
            n_in_b = int(in_b.sum())
            if n_in_b == 0:
                continue
            take = n_in_b if not sample_norm else min(n_in_b, int(rem_vec[b]))
            if take <= 0:
                continue
            if take == n_in_b:
                # prendi tutto
                km = idx_arr_local[in_b].astype(np.int64, copy=False)
                # np.add.at su tutte le k-mer in questo bin
                np.add.at(target_matrix, (b, km), 1)
                rem_vec[b] -= n_in_b if sample_norm else 0
            else:
                # campiona senza reinserimento 'take' posizioni dall'insieme in_b
                # ricava gli indici assoluti (posizioni) dove in_b True
                pos = np.flatnonzero(in_b)
                choose = rng.choice(pos, size=take, replace=False)
                km = idx_arr_local[choose].astype(np.int64, copy=False)
                np.add.at(target_matrix, (b, km), 1)
                rem_vec[b] -= take

    for chrom in chroms:
        print(f"[k={k}] Processing {chrom} ...")
        idx_arr = kmer_npz[chrom]            # shape Lk
        scr_q   = scores_npz[chrom]          # shape Lk, uint8 (0..100 o 255)
        flags   = coding_npz[chrom]          # shape L
        Lk = idx_arr.size
        if Lk == 0:
            continue

        sent = np.iinfo(idx_arr.dtype).max
        valid = (scr_q <= 100) & (idx_arr != sent)

        if center + Lk > flags.size:
            maxLk = flags.size - center
            if maxLk <= 0:
                continue
            idx_arr = idx_arr[:maxLk]
            scr_q   = scr_q[:maxLk]
            valid   = valid[:maxLk]
            Lk      = maxLk

        region_is_coding = flags[center : center + Lk]
        scr_q_float = scr_q.astype(np.float64, copy=False)
        bin_idx = (scr_q_float / w).astype(np.int64, copy=False)
        bin_idx = np.clip(bin_idx, 0, B - 1)

        m_valid_coding    = valid &  region_is_coding
        m_valid_noncoding = valid & (~region_is_coding)

        # accumula con (eventuale) downsampling controllato da rem_c / rem_n
        accumulate_selected(m_valid_coding,    idx_arr, freqs_coding,    bin_idx, None, rem_c)
        accumulate_selected(m_valid_noncoding, idx_arr, freqs_noncoding, bin_idx, None, rem_n)

        print(f"  {chrom}: valid={int(valid.sum()):,}  coding_used={int((freqs_coding.sum(axis=1)).sum()):,}  noncoding_used={int((freqs_noncoding.sum(axis=1)).sum()):,}")

    # salva CSV per ciascun bin e regione
    # struttura: outdir/k{k}/{coding|noncoding}/bin_{lo-hi}/counts.csv
    for region_name, F in [("coding", freqs_coding), ("noncoding", freqs_noncoding)]:
        base_dir = os.path.join(outdir_for_k, region_name)
        ensure_dir(base_dir)
        for b in range(B):
            b_lo = float(edges[b])
            b_hi = float(edges[b+1])
            sub = os.path.join(base_dir, f"bin_{fmt_range(b_lo, b_hi)}")
            ensure_dir(sub)
            csv_path = os.path.join(sub, f"k{k}_{region_name}_bin_{fmt_range(b_lo, b_hi)}.csv")
            write_bin_csv(csv_path, F[b], inv_kmer)

    # opzionale: salvataggi matrici per uso Python veloce
    np.savez_compressed(
        os.path.join(outdir_for_k, f"freq_matrices_k{k}.npz"),
        coding=freqs_coding,
        noncoding=freqs_noncoding,
        edges=edges,
        centers=centers,
        totals_coding=totals_coding,
        totals_noncoding=totals_noncoding,
        target_coding=target_c,
        target_noncoding=target_n,
        sample_norm=np.array([int(sample_norm)], dtype=np.int8),
    )

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Conte e frequenze dei k-mer per B bin di score su [0,100] (B dispari, bin centrale centrato su 50), separando coding vs noncoding. O(n)."
    )
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--outdir", required=True)

    # numero di bin (dispari)
    ap.add_argument("--bins", type=int, default=11,
                    help="Numero di bin su [0,100]. Deve essere dispari (uno centrato su 50). Default: 11")

    # normalizzazione per campionamento
    ap.add_argument("--sample_norm", action="store_true",
                    help="Se attivo, in ciascun bin si usa al più il numero di k-mer del bin più a destra (per regione). Se un bin eccede, si esegue downsampling casuale senza reinserimento.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Seed RNG per il campionamento (usato solo con --sample_norm).")

    # dove sono i dati (pattern con {k})
    ap.add_argument("--kmer-npz-pattern",
                    default="data/kmer_out/k{k}/hg38.k{k}.perchrom.npz",
                    help="Percorso NPZ indici k-mer per-chrom (usa {k}).")
    ap.add_argument("--kmer-index-pattern",
                    default="data/kmer_out/k{k}/kmer_index_k{k}.json",
                    help="Percorso JSON indice k-mer (usa {k}).")
    ap.add_argument("--scores-npz-pattern",
                    default="data/kmers_scores_phylo/k{k}/bw_center_k{k}.npz",
                    help="Percorso NPZ score (usa {k}); attesi uint8 0..100, 255=invalid.")
    ap.add_argument("--coding-npz",
                    default="data/coding_flags.autosomes.bool.npz",
                    help="NPZ con flags coding bool (chr1..chr22).")

    args = ap.parse_args()

    if args.bins % 2 == 0 or args.bins <= 0:
        raise SystemExit("--bins deve essere un intero dispari > 0")

    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for k in range(args.kmin, args.kmax + 1, 2):
        out_k = os.path.join(args.outdir, f"k{k}")
        os.makedirs(out_k, exist_ok=True)
        process_k(
            k=k,
            kmer_npz_path=args.kmer_npz_pattern.format(k=k),
            kmer_index_json=args.kmer_index_pattern.format(k=k),
            scores_npz_path=args.scores_npz_pattern.format(k=k),
            coding_npz_path=args.coding_npz,
            outdir_for_k=out_k,
            bins=args.bins,
            sample_norm=args.sample_norm,
            rng=rng,
        )

    print(f"[DONE] Output in {args.outdir}")

if __name__ == "__main__":
    main()
