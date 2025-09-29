#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import re
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
        idx = (idx << 2) | v
    return idx


def build_kmer_index(k: int) -> Dict[str, int]:
    """
    Costruisce il dizionario k-mer -> indice (lessicografico A<C<G<T).
    """
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

    mask = (1 << (2 * k)) - 1
    val = 0
    valid_run = 0

    for i, ch in enumerate(seq):
        v = BASE2VAL.get(ch)
        if v is None:
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


def save_npz_dict(arr_dict: Dict[str, np.ndarray], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arr_dict)


# ----------------------------
# Canonical chromosome selection & renaming
# ----------------------------

MITO_PATTERNS = [
    r"(?i)^chr?m(t)?$", r"(?i)^m(t)?$", r"(?i)mitochond", r"(?i)\bmito\b", r"(?i)^chrmt$"
]

# Tipici tag di contig non primari/decoy/patch/unlocalized
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
    Estrae un numero 'di cromosoma' se presente (es: chr1, 1, chr01 -> 1).
    Ritorna None se non trova un numero 'autonomo'.
    """
    # prendi prima sequenza numerica "principale"
    m = re.search(r"(?i)^chr?_?0*([1-9]\d*)$", name)
    if m:
        return int(m.group(1))
    # fallback: se il nome è solo numero
    if re.fullmatch(r"0*[1-9]\d*", name):
        return int(name.lstrip("0"))
    return None

import time

def build_canonical_mapping(all_records: List[Tuple[str, int]]) -> Dict[str, str]:
    """
    all_records: lista (orig_name, length) già filtrata da mito e 'non primary'.
    Ordine finale:
      1) cromosomi con numero (chr1, chr2, ..., o '1', 'chr01' -> 1) in ordine numerico crescente
      2) tutti gli altri in ordine alfabetico del nome originale
    Assegna etichette sequenziali chr1..chrN seguendo tale ordine.
    """
    if not all_records:
        return {}

    # split: numerici vs altri
    numeric = []   # (num, name)
    others = []    # name
    for name, _L in all_records:
        num = extract_numeric_token(name)
        if num is not None:
            numeric.append((num, name))
        else:
            if looks_non_primary(name):
                time.sleep(2)  # evita output multiplo identico in caso di logging concorrente
                if 'chr' in name.lower():
                    print(f"[WARN] Ignorato contig non primario/decoy in build_canonical_mapping: {name}", file=sys.stderr)
                continue

            others.append(name)

    # ordina
    numeric_sorted = [name for num, name in sorted(numeric, key=lambda x: x[0])]
    others_sorted = sorted(others)  # alfabetico puro

    ordered_names = numeric_sorted + others_sorted

    # assegna chr1..chrN
    return {nm: f"chr{i}" for i, nm in enumerate(ordered_names, start=1)}


# ----------------------------
# Main pipeline
# ----------------------------
def process(
    fasta_gz: str,
    outdir: str,
    k_min: int = 3,
    k_max: int = 7,
    build_concat: bool = False,
):
    """
    Esegue l'indicizzazione per k in [k_min, k_max].
    - Seleziona cromosomi canonici (no mitocondrio, no contig/decoy, numerici prima; altri grandi inclusi e rinominati)
    - Rinomina in chr1..chrN
    - Salva JSON con indice dei k-mer
    - Estrae vettori per cromosoma e (opz.) salva NPZ concatenato + metadati
    """
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Carico sequenze da: {fasta_gz}", file=sys.stderr)
    # Prima passata: leggi tutte le sequenze per stimare lunghezze & filtri
    all_seqs: List[Tuple[str, str]] = []
    raw_skipped = 0
    mito_skipped = 0
    nonprimary_skipped = 0

    for name, seq in fasta_reader_gz(fasta_gz):
        if is_mito(name):
            mito_skipped += 1
            print(f"[INFO] Escluso mitocondrio/MT-like: {name}", file=sys.stderr)
            continue
        if looks_non_primary(name):
            nonprimary_skipped += 1
            if 'chr' in name.lower():  # evita troppi warning
                print(f"[WARN] Escluso contig non primario/decoy: {name}", file=sys.stderr)
            continue
        if len(seq) == 0:
            raw_skipped += 1
            print(f"[WARN] Contig vuoto escluso: {name}", file=sys.stderr)
            continue
        all_seqs.append((name, seq))

    if not all_seqs:
        print("[ERROR] Nessun contig canonico trovato dopo i filtri.", file=sys.stderr)
        return

    # Costruisci mapping original_name -> chr1..chrN
    lengths = [(nm, len(s)) for nm, s in all_seqs]
    name_to_chr = build_canonical_mapping(lengths)

    # Applica mapping e ordina per 'chrN' (N crescente)
    def chr_number(label: str) -> int:
        m = re.match(r"^chr(\d+)$", label)
        return int(m.group(1)) if m else 10**9  # sicurezza, non dovrebbe accadere

    # Log mapping e statistiche
    print(f"[INFO] Canonici selezionati: {len(name_to_chr)}", file=sys.stderr)
    for nm in sorted(name_to_chr, key=lambda x: chr_number(name_to_chr[x])):
        print(f"[INFO]   {nm}  ->  {name_to_chr[nm]}", file=sys.stderr)

    # Salva mapping per riferimento
    mapping_path = os.path.join(outdir, "chrom_mapping.json")
    save_json({"mapping": name_to_chr}, mapping_path)
    print(f"[INFO] Salvato mapping cromosomi: {mapping_path}", file=sys.stderr)

    # Riordina sequenze secondo chr1..chrN
    canon_seqs = sorted(
        ((name_to_chr[nm], seq) for nm, seq in all_seqs if nm in name_to_chr),
        key=lambda x: chr_number(x[0])
    )

    print(f"[INFO] Trovati {len(canon_seqs)} cromosomi canonici da processare.", file=sys.stderr)

    for k in range(k_min, k_max + 1, 2):
        assert 3 <= k <= 7 and k % 2 == 1, "k deve essere dispari e in [3,7]"
        print(f"[INFO] === k={k} ===", file=sys.stderr)
        dt = dtype_for_k(k)

        # 1) Indice k-mer -> salva JSON
        kmer_index = build_kmer_index(k)
        json_path = os.path.join(outdir, f"k{k}", f"kmer_index_k{k}.json")
        save_json(kmer_index, json_path)
        print(f"[INFO] Salvato indice k-mer: {json_path}", file=sys.stderr)

        per_chr_dir = os.path.join(outdir, f"k{k}", "per_chrom")
        os.makedirs(per_chr_dir, exist_ok=True)

        arr_dict = {}
        lengths_meta = {}

        for chrom_label, chrom_seq in canon_seqs:
            arr = encode_kmer_indices(chrom_seq, k, dt)

            # opzionale: salva i singoli (commentato per ridurre I/O)
            # out_path = os.path.join(per_chr_dir, f"{chrom_label}.k{k}.npy")
            # np.save(out_path, arr)

            arr_dict[chrom_label] = arr
            lengths_meta[chrom_label] = int(arr.size)

        if build_concat:
            npz_path = os.path.join(outdir, f"k{k}", f"genome.k{k}.perchrom.npz")
            save_npz_dict(arr_dict, npz_path)

            meta = {
                "dtype": str(dtype_for_k(k)),
                "sentinel": sentinel_for_dtype(dt),
                "k": k,
                "total_length": int(sum(lengths_meta.values())),
                "lengths": lengths_meta,
                "note": "Ogni entry dell'NPZ è un vettore 0-based di indici k-mer per ciascun cromosoma canonico rinominato chr1..chrN; finestre invalide = sentinella."
            }
            meta_path = os.path.join(outdir, f"k{k}", f"genome.k{k}.perchrom.meta.json")
            save_json(meta, meta_path)

            print(f"[INFO] Salvato NPZ compresso per k={k}: {npz_path}", file=sys.stderr)

    print("[INFO] Completato.", file=sys.stderr)


# ----------------------------
# Batch launcher
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def fasta_path_for_id(asm_id: str) -> str:
    return os.path.join("data", "raw", asm_id, f"{asm_id}.genome.fas.gz")

def run_ids_from_json(json_path: str, processed_base: str, kmin: int, kmax: int, concat: bool, overwrite: bool = False):
    with open(json_path, "r") as fh:
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

    print(f"[INFO] Trovati {len(ids)} id nel JSON.", file=sys.stderr)

    for asm_id in ids:
        print(f"[INFO] Processing assembly: {asm_id}", file=sys.stderr)
        fasta_gz = fasta_path_for_id(asm_id)
        if not os.path.exists(fasta_gz) and not overwrite:
            print(f"[ERROR] FASTA mancante: {fasta_gz}", file=sys.stderr)
            continue
        outdir = os.path.join(processed_base, asm_id, "kmers_info")
        if os.path.exists(outdir) and False:
            print(f"[WARN] Output directory già esistente, salto: {outdir}", file=sys.stderr)
            continue
        ensure_dir(outdir)

        print(f"[INFO] {asm_id}: FASTA={fasta_gz}  OUT={outdir}", file=sys.stderr)
        process(
            fasta_gz=fasta_gz,
            outdir=outdir,
            k_min=kmin,
            k_max=kmax,
            build_concat=concat,
        )


def main():
    ap = argparse.ArgumentParser(description="K-mer indexing multi-specie: selezione cromosomi canonici, escluso mitocondrio, rinomina in chr1..chrN.")
    ap.add_argument("--assemblies-json", help="Lista assemblies; usa 'id' o stringhe. FASTA atteso in data/raw/{id}/{id}.genome.fas.gz")
    ap.add_argument("--assembly-id", help="Esegui un singolo assembly (usa path data/raw/{id}/{id}.genome.fas.gz)")
    ap.add_argument("--processed-base", default="data/processed", help="Base output (default: data/processed)")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--concat", action="store_true", default=True, help="Costruisci NPZ concatenato per ciascun k (default: True)")
    args = ap.parse_args()

    if args.assemblies_json:
        run_ids_from_json(args.assemblies_json, args.processed_base, args.kmin, args.kmax, args.concat)
        return

    if not args.assembly_id:
        print("[ERROR] Specifica --assemblies-json oppure --assembly-id.", file=sys.stderr)
        sys.exit(1)

    asm_id = args.assembly_id
    fasta_gz = os.path.join("data", "raw", asm_id, f"{asm_id}.genome.fas.gz")
    if not os.path.exists(fasta_gz):
        print(f"[ERROR] FASTA non trovato: {fasta_gz}", file=sys.stderr)
        sys.exit(1)

    outdir = os.path.join(args.processed_base, asm_id, "kmers_info")
    ensure_dir(outdir)

    print(f"[INFO] single-run {asm_id}: FASTA={fasta_gz}  OUT={outdir}", file=sys.stderr)
    process(
        fasta_gz=fasta_gz,
        outdir=outdir,
        k_min=args.kmin,
        k_max=args.kmax,
        build_concat=args.concat,
    )

if __name__ == "__main__":
    main()
