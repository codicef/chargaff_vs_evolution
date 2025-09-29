#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, math, re
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
    ensure_dir(os.path.dirname(out_csv))
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
# contig filtering (mito / non-primary)
# ----------------------------
MITO_PATTERNS = [
    r"(?i)^chr?m(t)?$", r"(?i)^m(t)?$", r"(?i)mitochond", r"(?i)\bmito\b", r"(?i)^chrmt$"
]

NON_PRIMARY_HINTS = [
    r"(?i)random", r"(?i)alt", r"(?i)fix", r"(?i)hap", r"(?i)\bun\b", r"(?i)unlocalized",
    r"(?i)scaff", r"(?i)patch", r"(?i)decoy", r"(?i)contig",
    r"(?i)^GL\d", r"(?i)^KI\d", r"(?i)^NW_\d", r"(?i)^NT_\d"
]

def is_mito(name: str) -> bool:
    return any(re.search(p, name) for p in MITO_PATTERNS)

def is_non_primary(name: str) -> bool:
    if any(re.search(p, name) for p in NON_PRIMARY_HINTS):
        return True
    # euristiche extra
    if "_" in name and not re.fullmatch(r"(?i)chr?\d+|^\d+$", name):
        return True
    return False

def select_primary_contigs(npz_list):
    """Intersezione dei contig tra più npz, filtrando mito e non-primari."""
    if not npz_list:
        return []
    common = set(npz_list[0].files)
    for z in npz_list[1:]:
        common &= set(z.files)
    kept = [c for c in sorted(common) if not is_mito(c) and not is_non_primary(c)]
    return kept


# helpers vari x downsample
def binomial_thin_counts(counts_row: np.ndarray, keep: int, rng) -> np.ndarray:
    """Restituisce una nuova riga con ~keep conteggi totali, thinning binomiale per k-mer."""
    total = int(counts_row.sum())
    if total <= 0 or keep >= total:
        return counts_row.copy()
    p = keep / total
    # Binomiale indipendente per k-mer: preserva in media le proporzioni
    return rng.binomial(counts_row.astype(np.int64, copy=False), p).astype(np.int64)

def rebalance_bins_towards_zero(F: np.ndarray, totals: np.ndarray, cap: int, rng) -> np.ndarray:
    """
    F: (B, V) conteggi grezzi per bin
    totals: (B,) somma per bin
    cap: target per ogni bin (= conteggi del bin più alto)
    Restituisce una nuova matrice (B, V) riequilibrata.
    """
    B, V = F.shape
    F_bal = F.astype(np.int64, copy=True)
    tot = totals.astype(np.int64, copy=True)

    for b in range(B-1, -1, -1):
        if cap <= 0:
            # niente da fare: tutto si azzera
            F_bal[b, :] = 0
            tot[b] = 0
            continue

        if tot[b] > cap:
            # Downsample nel bin b
            F_bal[b, :] = binomial_thin_counts(F_bal[b, :], cap, rng)
            tot[b] = int(F_bal[b, :].sum())

        elif tot[b] < cap:
            need = cap - tot[b]
            # Prendi dai bin più bassi (b-1, b-2, ..., 0)
            src = b - 1
            while need > 0 and src >= 0:
                avail = int(tot[src])
                if avail > 0:
                    take = min(avail, need)
                    # quota proporzionale per k-mer dal bin src
                    # fattore p = take / avail (binomiale per ogni k-mer)
                    p = take / avail
                    moved = rng.binomial(F_bal[src, :], p).astype(np.int64)
                    # aggiorna sorgente e destinazione
                    F_bal[src, :] -= moved
                    F_bal[b,   :] += moved
                    tot[src] -= int(moved.sum())
                    tot[b]   += int(moved.sum())
                    need = cap - tot[b]
                src -= 1
            # se non siamo riusciti a raggiungere cap (p.e. dati insufficienti), accettiamo il massimo possibile
        # se tot[b] == cap: perfetto, non facciamo nulla

    return F_bal





# ----------------------------
# core pass (O(n))
# ----------------------------
def process_k_for_kind(
    k,
    kmer_npz_path,
    kmer_index_json,
    scores_npz_path,
    coding_npz_path,
    outdir_for_kind_and_k,
    bins,                      # num bin (dispari)
    sample_norm=False,
    rng=None,
):
    print(f"[INFO] k={k}  bins={bins}  sample_norm={sample_norm}")

    if bins % 2 == 0 or bins <= 0:
        raise ValueError("--bins deve essere un intero dispari > 0")

    # carica dati
    kmer_npz   = np.load(kmer_npz_path)          # contig -> uint dtype (len = L-k+1)
    scores_npz = np.load(scores_npz_path)        # contig -> uint8 (0..100, 255=invalid), len = L-k+1
    coding_npz = np.load(coding_npz_path)        # contig -> bool (len = L)
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

    # contig in comune tra tutti i file, filtrati
    chroms = select_primary_contigs([kmer_npz, scores_npz, coding_npz])
    if not chroms:
        raise RuntimeError("Nessun contig valido dopo il filtraggio (controlla path e formati NPZ).")

    # ------------- PASS 1: quota per bin -------------
    totals_coding = np.zeros(B, dtype=np.int64)
    totals_noncoding = np.zeros(B, dtype=np.int64)

    for chrom in chroms:
        idx_arr = kmer_npz[chrom]            # shape Lk
        scr_q   = scores_npz[chrom]          # shape Lk, uint8 (0..100 o 255)
        flags   = coding_npz[chrom]          # shape L
        Lk = idx_arr.size
        if Lk == 0:
            continue

        sentinel = np.iinfo(idx_arr.dtype).max
        valid = (scr_q <= 100) & (idx_arr != sentinel)

        if center + Lk > flags.size:
            maxLk = flags.size - center
            if maxLk <= 0:
                continue
            idx_arr = idx_arr[:maxLk]
            scr_q   = scr_q[:maxLk]
            valid   = valid[:maxLk]
            Lk      = maxLk

        region_is_coding = flags[center : center + Lk]
        bin_idx = np.clip((scr_q.astype(np.float64, copy=False) / w).astype(np.int64, copy=False), 0, B - 1)

        m_valid_coding    = valid &  region_is_coding
        m_valid_noncoding = valid & (~region_is_coding)

        if np.any(m_valid_coding):
            totals_coding += np.bincount(bin_idx[m_valid_coding], minlength=B)
        if np.any(m_valid_noncoding):
            totals_noncoding += np.bincount(bin_idx[m_valid_noncoding], minlength=B)

    cap_c = int(totals_coding[B-1])
    cap_n = int(totals_noncoding[B-1])

    target_c = np.minimum(totals_coding, cap_c) if sample_norm else totals_coding
    target_n = np.minimum(totals_noncoding, cap_n) if sample_norm else totals_noncoding

    rem_c = target_c.astype(np.int64).copy()
    rem_n = target_n.astype(np.int64).copy()

    # ------------- PASS 2: accumulo (con eventuale downsampling) -------------
    def accumulate_selected(sel_mask, idx_arr_local, target_matrix, bin_idx_local, rem_vec):
        if not np.any(sel_mask):
            return
        present_bins = np.unique(bin_idx_local[sel_mask])
        for b in present_bins:
            if sample_norm and rem_vec[b] <= 0:
                continue
            in_b = sel_mask & (bin_idx_local == b)
            n_in_b = int(in_b.sum())
            if n_in_b == 0:
                continue
            take = n_in_b if not sample_norm else min(n_in_b, int(rem_vec[b]))
            if take <= 0:
                continue
            if take == n_in_b:
                km = idx_arr_local[in_b].astype(np.int64, copy=False)
            else:
                pos = np.flatnonzero(in_b)
                choose = rng.choice(pos, size=take, replace=False)
                km = idx_arr_local[choose].astype(np.int64, copy=False)
            np.add.at(target_matrix, (b, km), 1)
            if sample_norm:
                rem_vec[b] -= take
                if rem_vec[b] < 0:
                    rem_vec[b] = 0

    for chrom in chroms:
        print(f"[k={k}] Processing {chrom} ...")
        idx_arr = kmer_npz[chrom]
        scr_q   = scores_npz[chrom]
        flags   = coding_npz[chrom]
        Lk = idx_arr.size
        if Lk == 0:
            continue

        sentinel = np.iinfo(idx_arr.dtype).max
        valid = (scr_q <= 100) & (idx_arr != sentinel)

        if center + Lk > flags.size:
            maxLk = flags.size - center
            if maxLk <= 0:
                continue
            idx_arr = idx_arr[:maxLk]
            scr_q   = scr_q[:maxLk]
            valid   = valid[:maxLk]
            Lk      = maxLk

        region_is_coding = flags[center : center + Lk]
        bin_idx = np.clip((scr_q.astype(np.float64, copy=False) / w).astype(np.int64, copy=False), 0, B - 1)

        m_valid_coding    = valid &  region_is_coding
        m_valid_noncoding = valid & (~region_is_coding)

        accumulate_selected(m_valid_coding,    idx_arr, freqs_coding,    bin_idx, rem_c)
        accumulate_selected(m_valid_noncoding, idx_arr, freqs_noncoding, bin_idx, rem_n)


        if sample_norm:
            # target = conteggi del bin più alto (per area: coding/noncoding)
            cap_c = int(totals_coding[-1])
            cap_n = int(totals_noncoding[-1])

            freqs_coding_bal    = rebalance_bins_towards_zero(freqs_coding,    totals_coding,    cap_c, rng)
            freqs_noncoding_bal = rebalance_bins_towards_zero(freqs_noncoding, totals_noncoding, cap_n, rng)

            # ricalcola i totals/target effettivi post-rebalance
            totals_coding_eff    = freqs_coding_bal.sum(axis=1).astype(np.int64)
            totals_noncoding_eff = freqs_noncoding_bal.sum(axis=1).astype(np.int64)
            target_c = np.full(B, cap_c, dtype=np.int64)
            target_n = np.full(B, cap_n, dtype=np.int64)

            # usa le matrici riequilibrate per l’export
            out_mats = {
                "coding":    freqs_coding_bal,
                "noncoding": freqs_noncoding_bal,
            }
        else:
            # comportamento precedente (nessun rebalance)
            totals_coding_eff    = totals_coding
            totals_noncoding_eff = totals_noncoding
            target_c = totals_coding
            target_n = totals_noncoding
            out_mats = {
                "coding":    freqs_coding,
                "noncoding": freqs_noncoding,
            }

        # salva CSV per ciascun bin e regione (usando out_mats)
        for region_name, F in out_mats.items():
            base_dir = os.path.join(outdir_for_kind_and_k, region_name)
            ensure_dir(base_dir)
            for b in range(B):
                b_lo = float(edges[b])
                b_hi = float(edges[b+1])
                sub = os.path.join(base_dir, f"bin_{fmt_range(b_lo, b_hi)}")
                csv_path = os.path.join(sub, f"k{k}_{region_name}_bin_{fmt_range(b_lo, b_hi)}.csv")
                write_bin_csv(csv_path, F[b], inv_kmer)

        # salva NPZ riepilogo (incluse matrici riequilibrate se sample_norm)
        np.savez_compressed(
            os.path.join(outdir_for_kind_and_k, f"freq_matrices_k{k}_{bins}bins_{sample_norm}sample.npz"),
            coding=out_mats["coding"],
            noncoding=out_mats["noncoding"],
            edges=edges,
            centers=centers,
            totals_coding=totals_coding_eff,
            totals_noncoding=totals_noncoding_eff,
            target_coding=target_c,
            target_noncoding=target_n,
            sample_norm=np.array([int(sample_norm)], dtype=np.int8),
        )

# ----------------------------
# CLI (multi-assembly, multi-kind)
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Conte/frequenze k-mer per bin di score su [0,100] (B dispari, bin centrale su 50), separando coding vs noncoding. Supporto assemblies.json e {center,first,mean}."
    )
    ap.add_argument("--assemblies-json", required=True,
                    help="JSON con lista di assembly (campo 'id' usato come slug).")
    ap.add_argument("--ids", default="",
                    help="Comma-separated subset di id da processare (default: tutti quelli in JSON).")
    ap.add_argument("--method", default="phylop",
                    help="Metodo degli score (default: phylop).")
    ap.add_argument("--score-kinds", default="center,first,mean",
                    help="Tipi di score da processare (csv tra: center,first,mean).")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--out-root", default="data/processed",
                    help="Root per l’output: data/processed/{id}/{method}_scores/bins_scores/k{k}/{kind}/")
    ap.add_argument("--bins", type=int, default=11,
                    help="Numero di bin su [0,100]. Dispari. Default: 11")
    ap.add_argument("--sample_norm", action="store_true", default=None,
                    help="In ciascun bin usa al più il numero di esempi del bin più a destra (per regione); downsampling casuale senza reinserimento.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Seed RNG usato con --sample_norm.")

    # pattern input (override se necessario)
    ap.add_argument("--kmer-perchrom-pattern",
                    default="data/processed/{id}/kmers_info/k{k}/genome.k{k}.perchrom.npz",
                    help="NPZ per-chrom di indici k-mer (usa {id},{k}).")
    ap.add_argument("--kmer-index-pattern",
                    default="data/processed/{id}/kmers_info/k{k}/kmer_index_k{k}.json",
                    help="JSON indice k-mer (usa {k}).")
    ap.add_argument("--coding-npz-pattern",
                    default="data/processed/{id}/cds_flags.primary.bool.npz",
                    help="NPZ booleani coding per contig (usa {id}).")
    ap.add_argument("--scores-root-pattern",
                    default="data/processed/{id}/{method}_scores/kmers_scores/k{k}/{method}_{kind}_k{k}.npz",
                    help="NPZ di score quantizzati 0..100 (usa {id},{method},{k},{kind}).")

    args = ap.parse_args()

    # validazione bins
    if args.bins % 2 == 0 or args.bins <= 0:
        raise SystemExit("--bins deve essere un intero dispari > 0")

    # carica assemblies
    assemblies = load_json(args.assemblies_json)
    wanted_ids = set([s.strip() for s in args.ids.split(",") if s.strip()]) if args.ids else None
    kinds = [s.strip() for s in args.score_kinds.split(",") if s.strip()]
    invalid = [k for k in kinds if k not in {"center","first","mean"}]
    if invalid:
        raise SystemExit(f"--score-kinds contiene valori non supportati: {invalid}")

    rng = np.random.default_rng(args.seed)

    for entry in assemblies:
        asm_id = entry.get("id")
        if not asm_id:
            continue
        if wanted_ids and asm_id not in wanted_ids:
            continue

        print(f"\n=== Assembly: {asm_id} (method={args.method}) ===")
        for k in range(args.kmin, args.kmax + 1, 2):
            kmer_npz_path   = args.kmer_perchrom_pattern.format(id=asm_id, k=k)
            kmer_index_json = args.kmer_index_pattern.format(k=k, id=asm_id)
            coding_npz_path = args.coding_npz_pattern.format(id=asm_id)

            # verifica esistenza base (lascio eventuali errori espliciti se mancanti)
            if not os.path.exists(kmer_npz_path):
                print(kmer_npz_path)
                raise FileNotFoundError(f"Manca kmer per-chrom: {kmer_npz_path}")
            if not os.path.exists(kmer_index_json):
                raise FileNotFoundError(f"Manca indice k-mer: {kmer_index_json}")
            if not os.path.exists(coding_npz_path):
                raise FileNotFoundError(f"Manca coding flags: {coding_npz_path}")

            for kind in kinds:
                scores_npz_path = args.scores_root_pattern.format(
                    id=asm_id, method=args.method, k=k, kind=kind, bins=args.bins
                )
                if not os.path.exists(scores_npz_path):
                    raise FileNotFoundError(f"Manca scores NPZ: {scores_npz_path}")

                outdir_for_kind_and_k = os.path.join(
                    args.out_root, asm_id, f"{args.method}_scores", "bins_scores", f"k{k}", kind
                )
                ensure_dir(outdir_for_kind_and_k)

                SAMPLE_WAYS = [False]

                if args.sample_norm:
                    SAMPLE_WAYS = [True]

                if args.sample_norm is None:
                    print("[INFO] Eseguo sia con che senza --sample_norm (impostato a None)")
                    SAMPLE_WAYS = [False, True]


                for sample_norm in SAMPLE_WAYS:
                    if os.path.exists(os.path.join(outdir_for_kind_and_k, f"freq_matrices_k{k}_{args.bins}bins_{args.sample_norm}sample.npz")):
                        print(f"[SKIP] k={k} kind={kind} già fatto in {outdir_for_kind_and_k}")
                        continue
                    args.sample_norm = sample_norm
                    print(f"[RUN] id={asm_id} k={k} kind={kind}")
                    process_k_for_kind(
                        k=k,
                        kmer_npz_path=kmer_npz_path,
                        kmer_index_json=kmer_index_json,
                        scores_npz_path=scores_npz_path,
                        coding_npz_path=coding_npz_path,
                        outdir_for_kind_and_k=outdir_for_kind_and_k,
                        bins=args.bins,
                        sample_norm=args.sample_norm,
                        rng=rng,
                    )

    print("\n[DONE] Tutti i job completati.")

if __name__ == "__main__":
    main()
