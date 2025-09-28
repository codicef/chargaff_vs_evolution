#!/usr/bin/env python3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import sys
from typing import Dict, Iterator, Tuple, List, Optional

import numpy as np


# ----------------------------
# FASTA reader (streaming)
# ----------------------------
def fasta_reader_gz(path: str) -> Iterator[Tuple[str, str]]:
    """
    Legge un FASTA .gz producendo (header, sequence) per ogni record.
    Concatena le righe di sequenza e le restituisce in maiuscolo.
    """
    name = None
    seq_chunks = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_chunks).upper()
                name = line[1:].strip().split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if name is not None:
            yield name, "".join(seq_chunks).upper()


# ----------------------------
# K-mer indexing
# ----------------------------
BASE2VAL = {"A": 0, "C": 1, "G": 2, "T": 3}
VALID = set(BASE2VAL.keys())


def kmer_to_index(kmer: str) -> Optional[int]:
    """Converte un k-mer A/C/G/T in indice base-4 (A<C<G<T). Ritorna None se contiene basi non valide."""
    idx = 0
    for ch in kmer:
        v = BASE2VAL.get(ch)
        if v is None:
            return None
        idx = (idx << 2) | v  # equivalente a idx*4 + v
    return idx


def build_kmer_index(k: int) -> Dict[str, int]:
    """
    Costruisce il dizionario k-mer -> indice (lessicografico A<C<G<T).
    Nota: non genera esplicitamente tutte le combinazioni (potrebbe essere heavy),
    ma è deterministico: indice = interpretazione in base-4 di A=0,C=1,G=2,T=3.
    Per salvataggio JSON, generiamo le chiavi in ordine.
    """
    # Generiamo in modo iterativo, senza ricorsione profonda.
    alphabet = ["A", "C", "G", "T"]
    kmers = [""]
    for _ in range(k):
        kmers = [p + a for p in kmers for a in alphabet]
    return {km: kmer_to_index(km) for km in kmers}


def dtype_for_k(k: int) -> np.dtype:
    max_val = 4 ** k - 1
    if max_val <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_val <= np.iinfo(np.uint16).max:
        return np.uint16
    else:
        # Non serve per k<=7, ma in generale:
        return np.uint32


def sentinel_for_dtype(dt: np.dtype) -> int:
    return int(np.iinfo(dt).max)


# ----------------------------
# Rolling encoder per k-mer indices
# ----------------------------
def encode_kmer_indices(seq: str, k: int, dt: np.dtype) -> np.ndarray:
    """
    Ritorna un array di lunghezza len(seq)-k+1 con l'indice del k-mer in ogni posizione.
    Se la finestra contiene basi non A/C/G/T, scrive la sentinella nel relativo range.
    Implementa rolling window O(n).
    """
    n = len(seq)
    if n < k:
        return np.zeros(0, dtype=dt)

    sent = sentinel_for_dtype(dt)
    out = np.full(n - k + 1, sent, dtype=dt)

    mask = (1 << (2 * k)) - 1  # per mantenere solo gli ultimi 2k bit
    val = 0
    valid_run = 0  # numero di basi valide consecutive nella finestra corrente

    for i, ch in enumerate(seq):
        v = BASE2VAL.get(ch)
        if v is None:
            # reset
            val = 0
            valid_run = 0
        else:
            val = ((val << 2) | v) & mask
            valid_run += 1
            if valid_run >= k:
                pos = i - k + 1
                out[pos] = val

    return out


# ----------------------------
# Salvataggi helper
# ----------------------------
def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def save_npy(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def save_npz_dict(arr_dict: Dict[str, np.ndarray], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arr_dict)


# ----------------------------
# Main pipeline
# ----------------------------
def process(
    fasta_gz: str,
    outdir: str,
    k_min: int = 3,
    k_max: int = 7,
    build_concat: bool = False,
    chrom_prefix_filter: Tuple[str, ...] = ("chr",),  # per hg38 UCSC-style

):
    """
    Esegue l'indicizzazione per k in [k_min, k_max].
    - Salva JSON con indice dei k-mer
    - Estrae vettori per cromosoma e li salva in .npy
    - Facoltativo: crea vettore concatenato + metadati offset per cromosoma (per ciascun k)
    """
    os.makedirs(outdir, exist_ok=True)

    canonical = {f"chr{i}" for i in range(1, 23)} #| {"chrX", "chrY", "chrM"}
    # Primo pass: leggiamo una volta e memorizziamo in RAM i cromosomi (per semplicità/velocità).
    # Se vuoi ridurre RAM, puoi processare cromosoma-per-cromosoma per ogni k (più I/O).
    print(f"[INFO] Carico sequenze da: {fasta_gz}", file=sys.stderr)
    chroms: List[Tuple[str, str]] = []
    for name, seq in fasta_reader_gz(fasta_gz):
        if name not in canonical:
            if chrom_prefix_filter and not name.startswith(chrom_prefix_filter):
                print(f"[WARN] Ignorato contig non canonico: {name}", file=sys.stderr)
                continue
            else:
                print(f"[WARN] Attenzione: contig non canonico: {name}", file=sys.stderr)
                continue
        print(f"[INFO] Caricato contig: {name} (len={len(seq):,})", file=sys.stderr)
        chroms.append((name, seq))
    print(f"[INFO] Trovati {len(chroms)} contig/cromosomi da processare.", file=sys.stderr)

    chroms = sorted(chroms, key=lambda x: int(x[0][3:]))  # ordine alfabetico
    for k in range(k_min, k_max + 1):
        print(f"[INFO] === k={k} ===", file=sys.stderr)
        dt = dtype_for_k(k)
        sent = sentinel_for_dtype(dt)

        # 1) Indice k-mer -> salva JSON (deterministico)
        kmer_index = build_kmer_index(k)
        json_path = os.path.join(outdir, f"k{k}", f"kmer_index_k{k}.json")
        save_json(kmer_index, json_path)
        print(f"[INFO] Salvato indice k-mer: {json_path}", file=sys.stderr)


        per_chr_dir = os.path.join(outdir, f"k{k}", "per_chrom")
        os.makedirs(per_chr_dir, exist_ok=True)

        arr_dict = {}          # per l'NPZ unico
        lengths_meta = {}      # per metadati (lunghezze per cromosoma)

        for chrom_name, chrom_seq in chroms:
            arr = encode_kmer_indices(chrom_seq, k, dt)

            # (opzionale) se vuoi comunque i file per-cromosoma .npy, lasciali; altrimenti commenta le due righe sotto
            out_path = os.path.join(per_chr_dir, f"{chrom_name}.k{k}.npy")
            #save_npy(arr, out_path)

            # accumulo per NPZ
            arr_dict[chrom_name] = arr
            lengths_meta[chrom_name] = int(arr.size)

        # 3) Concatenazione opzionale (può essere molto grande!)
        if build_concat:
            # 3) salva NPZ compresso: una voce per ogni cromosoma (chiave = nome cromosoma)
            npz_path = os.path.join(outdir, f"k{k}", f"hg38.k{k}.perchrom.npz")
            save_npz_dict(arr_dict, npz_path)

            # 4) metadati sidecar
            meta = {
                "dtype": str(dtype_for_k(k)),
                "sentinel": sentinel_for_dtype(dt),
                "k": k,
                "total_length": int(sum(lengths_meta.values())),
                "lengths": lengths_meta,
                "note": "Ogni entry dell'NPZ è un vettore 0-based di indici k-mer per quel cromosoma; finestra invalida = sentinella."
            }
            meta_path = os.path.join(outdir, f"k{k}", f"hg38.k{k}.perchrom.meta.json")
            save_json(meta, meta_path)

            print(f"[INFO] Salvato NPZ compresso per k={k}: {npz_path}", file=sys.stderr)

    print("[INFO] Completato.", file=sys.stderr)


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Indicizzazione k-mer (k=3..7) su hg38.fa.gz")
    ap.add_argument("--fasta", required=True, help="Percorso a hg38.fa.gz (FASTA compresso).")
    ap.add_argument("--outdir", required=True, help="Directory di output.")
    ap.add_argument("--kmin", type=int, default=3, help="k minimo (default: 3).")
    ap.add_argument("--kmax", type=int, default=7, help="k massimo (default: 7).")
    ap.add_argument(
        "--concat",
        action="store_true",
        help="Se presente, crea anche una matrice concatenata per ciascun k (ATTENZIONE memoria/spazio!).",
    )
    ap.add_argument(
        "--no-chr-filter",
        action="store_true",
        help="Se presente, non filtra per prefisso 'chr' nei nomi dei contig.",
    )
    args = ap.parse_args()

    chrom_filter = tuple() if args.no_chr_filter else ("chr",)

    process(
        fasta_gz=args.fasta,
        outdir=args.outdir,
        k_min=args.kmin,
        k_max=args.kmax,
        build_concat=args.concat,
        chrom_prefix_filter=chrom_filter,
    )


if __name__ == "__main__":
    main()
