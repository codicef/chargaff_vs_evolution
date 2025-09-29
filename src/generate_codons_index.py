
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import re
import sys
from typing import Dict, Iterator, Tuple, List, Optional
from collections import defaultdict

import numpy as np

# ----------------------------
# FASTA reader (streaming intero contig)
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
# Utils
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)

def save_npz_dict(arr_dict: Dict[str, np.ndarray], path: str):
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arr_dict)

# ----------------------------
# Canonical chromosome selection & renaming (coerente col tuo script)
# ----------------------------
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
    if not re.match(r"(?i)^chr?", name):
        return True
    return any(re.search(p, name) for p in NON_PRIMARY_HINTS)

def extract_numeric_token(name: str) -> Optional[int]:
    m = re.fullmatch(r"(?i)^chr0*([1-9]\d*)$", name)
    if m: return int(m.group(1))
    m = re.fullmatch(r"0*([1-9]\d*)$", name)
    if m: return int(m.group(1))
    return None

def build_canonical_mapping(all_records: List[Tuple[str, int]]) -> Dict[str, str]:
    """
    all_records: (orig_name, length) già filtrati.
    Ordina: numerici (1..N) poi altri alfabetico. Mappa a chr1..chrK.
    """
    if not all_records:
        return {}
    numeric: List[Tuple[int, str]] = []
    others: List[str] = []
    for nm, _L in all_records:
        n = extract_numeric_token(nm)
        if n is not None: numeric.append((n, nm))
        else:             others.append(nm)
    ordered = [nm for _, nm in sorted(numeric, key=lambda x: x[0])] + sorted(others, key=str.lower)
    return {nm: f"chr{i}" for i, nm in enumerate(ordered, start=1)}

# ----------------------------
# GTF parsing (solo righe CDS)
# ----------------------------
def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

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
        out[key.strip()] = val.strip()
    return out

def read_gtf_cds(gtf_path: str) -> Iterator[Tuple[str,int,int,str,int,Dict[str,str]]]:
    """
    Yield CDS entries: (chrom, start0, end, strand, phase, attrs)
    start0 0-based, end exclusive.
    """
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"): continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9: continue
            chrom, _src, feat, start, end, _score, strand, frame, attrs = f
            if feat != "CDS": continue
            try:
                s0 = int(start) - 1
                e  = int(end)
            except ValueError:
                continue
            try:
                phase = 0 if frame == "." else int(frame)
            except ValueError:
                phase = 0
            yield chrom, s0, e, strand, phase, parse_gtf_attributes(attrs)

def derive_sizes_from_gtf(gtf_path: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"): continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5: continue
            chrom, _, _, _, end = f[:5]
            try:
                e = int(end)
            except ValueError:
                continue
            if e > sizes.get(chrom, 0):
                sizes[chrom] = e
    return sizes

# ----------------------------
# RC / alfabeti
# ----------------------------
BASE2VAL = {"A":0, "C":1, "G":2, "T":3}
COMP = str.maketrans("ACGTN", "TGCAN")

def rc_base(b: str) -> str:
    return b.translate(COMP)

def codon_to_index(codon: str) -> int:
    """AAA..TTT ordine lessicografico base4; -1 se contiene N."""
    if len(codon) != 3: return -1
    idx = 0
    for ch in codon:
        v = BASE2VAL.get(ch)
        if v is None:
            return -1
        idx = (idx << 2) | v
    return idx  # 0..63

# ----------------------------
# Core: costruzione array per-chr
# ----------------------------
def process_assembly(asm_id: str, gtf_path: str, fasta_gz: str, outdir: str) -> bool:
    print(f"\n[RUN] {asm_id}", file=sys.stderr)
    print(f"  GTF:   {gtf_path}", file=sys.stderr)
    print(f"  FASTA: {fasta_gz}", file=sys.stderr)
    print(f"  OUT:   {outdir}", file=sys.stderr)

    if not os.path.exists(gtf_path):
        print(f"[ERROR] GTF non trovato: {gtf_path}", file=sys.stderr); return False
    if not os.path.exists(fasta_gz):
        print(f"[ERROR] FASTA non trovato: {fasta_gz}", file=sys.stderr); return False

    # 1) selezione contig canonici da GTF e mapping → chr1..chrN
    sizes_by_orig = derive_sizes_from_gtf(gtf_path)
    records: List[Tuple[str,int]] = []
    for nm, L in sizes_by_orig.items():
        if L<=0: continue
        if is_mito(nm): continue
        if looks_non_primary(nm): continue
        records.append((nm, L))
    if not records:
        print("[ERROR] Nessun contig primario dopo i filtri.", file=sys.stderr); return False
    name_to_chr = build_canonical_mapping(records)

    # 2) carica FASTA in RAM SOLO per i contig canonici
    genome: Dict[str,str] = {}
    kept = set(name_to_chr.keys())
    for nm, seq in fasta_reader_gz(fasta_gz):
        if nm in kept:
            genome[nm] = seq
    missing = [nm for nm in kept if nm not in genome]
    if missing:
        print(f"[ERROR] FASTA non contiene {len(missing)} contig attesi, e.g. {missing[:3]} ...", file=sys.stderr)
        return False

    # 3) prepara array per-chr
    #    arr_idx: int32 (-1 default). arr_bp: uint8 (255 default).
    arr_idx: Dict[str, np.ndarray] = {}
    arr_bp:  Dict[str, np.ndarray] = {}
    chrom_lengths: Dict[str,int] = {}
    # ordinamento per chrN
    ordered_pairs = sorted(name_to_chr.items(), key=lambda kv: int(kv[1].replace("chr","")))
    for orig, canon in ordered_pairs:
        L = len(genome[orig])
        chrom_lengths[canon] = L
        arr_idx[canon] = np.full(L, -1, dtype=np.int32)
        arr_bp[canon]  = np.full(L, 255, dtype=np.uint8)

    # 4) raggruppa CDS per transcript, applica phase e strand, riempi array
    cds_by_tx: Dict[str, List[Tuple[str,int,int,str,int]]] = defaultdict(list)
    for chrom, s0, e, strand, phase, attrs in read_gtf_cds(gtf_path):
        if chrom not in name_to_chr:  # scarta non-canonici
            continue
        tx = attrs.get("transcript_id") or attrs.get("transcriptId") or attrs.get("transcript") or attrs.get("transcriptIdVersion")
        if not tx:
            tx = attrs.get("gene_id") or "NO_TX"
        cds_by_tx[tx].append((chrom, s0, e, strand, phase))

    def order_blocks(blocks: List[Tuple[str,int,int,str,int]], strand: str):
        return sorted(blocks, key=lambda x: x[1], reverse=(strand=="-"))

    filled_bases = 0
    codon_bases  = 0

    for tx, blocks in cds_by_tx.items():
        if not blocks: continue
        strand = blocks[0][3]
        blocks = order_blocks(blocks, strand)

        # costruiamo la lista di (canon_chr, pos1, base_in_trascritto) nel VERSO trascrizionale
        coords: List[Tuple[str,int,str]] = []  # (chr_canon, pos1, base_5to3)
        for (chrom, s0, e, st, phase) in blocks:
            seq = genome[chrom]
            if strand == "+":
                s0_trim = s0 + phase
                e_trim  = e
                if e_trim <= s0_trim: continue
                canon = name_to_chr[chrom]
                # posizioni genomiche s0_trim .. e_trim-1
                for p0 in range(s0_trim, e_trim):
                    base = seq[p0]
                    # ignora se fuori alfabeto (lascerà -1/255)
                    coords.append( (canon, p0+1, base) )
            else:
                # per '-' taglia dalla DESTRA genomica
                s0_trim = s0
                e_trim  = e - phase
                if e_trim <= s0_trim: continue
                canon = name_to_chr[chrom]
                for p0 in range(s0_trim, e_trim):
                    # la base nel verso trascrizionale è il complementare (RC) e l'ordine va invertito alla fine
                    base = rc_base(seq[p0])
                    coords.append( (canon, p0+1, base) )
        if not coords:
            continue
        # re-ordina per verso trascrizionale su strand '-'
        if strand == "-":
            coords = coords[::-1]

        # spezza in triplette e riempi le tre basi
        L_trip = (len(coords)//3)*3
        if L_trip < 3:
            continue
        i = 0
        while i < L_trip:
            (c1, p1, b1) = coords[i]
            (c2, p2, b2) = coords[i+1]
            (c3, p3, b3) = coords[i+2]
            # se i tre chr non coincidono (non dovrebbe accadere dopo trim), salta
            if not (c1 == c2 == c3):
                i += 3
                continue
            codon = (b1 + b2 + b3).upper()
            cidx = codon_to_index(codon)  # -1 se contiene N
            # scrivi solo se posizioni non già marcate (evita sovrascritture in casi patologici)
            if arr_bp[c1][p1-1] == 255:
                arr_bp[c1][p1-1]  = 0
                arr_idx[c1][p1-1] = cidx
                codon_bases += 1
            if arr_bp[c2][p2-1] == 255:
                arr_bp[c2][p2-1]  = 1
                arr_idx[c2][p2-1] = cidx
                codon_bases += 1
            if arr_bp[c3][p3-1] == 255:
                arr_bp[c3][p3-1]  = 2
                arr_idx[c3][p3-1] = cidx
                codon_bases += 1
            filled_bases += 3
            i += 3

    # 5) salva mapping, NPZ e meta (stile k-mer)
    mapping_path = os.path.join(outdir, "chrom_mapping.json")
    save_json({"mapping": name_to_chr}, mapping_path)
    print(f"[INFO] Salvato mapping cromosomi: {mapping_path}", file=sys.stderr)

    npz_payload = {}
    for canon, L in chrom_lengths.items():
        npz_payload[f"{canon}_codon_idx"] = arr_idx[canon]
        npz_payload[f"{canon}_base_pos"]  = arr_bp[canon]
    npz_path = os.path.join(outdir, "codons.primary.kappamer.npz")
    save_npz_dict(npz_payload, npz_path)
    print(f"[INFO] Salvato NPZ: {npz_path}", file=sys.stderr)

    meta = {
        "dtype_idx": str(np.dtype(np.int32)),
        "dtype_basepos": str(np.dtype(np.uint8)),
        "sentinel_idx": -1,
        "sentinel_basepos": 255,
        "note": "Per ciascun cromosoma canonico (chr1..chrN): _codon_idx (0..63, -1 se non codone/contiene N) e _base_pos (0/1/2 per 1a/2a/3a base, 255 se non codone). Coordinate 1-based nella fase di riempimento.",
        "assembly": asm_id,
        "lengths": chrom_lengths,
        "tot_filled_triplet_bases": int(codon_bases),
    }
    meta_path = os.path.join(outdir, "codons.primary.kappamer.meta.json")
    save_json(meta, meta_path)
    print(f"[DONE] Salvato meta: {meta_path}", file=sys.stderr)

    return True

# ----------------------------
# Batch launcher (stile tuo)
# ----------------------------
def run_ids_from_json(json_path: str, processed_base: str, gtf_tmpl: str, fasta_tmpl: str):
    with open(json_path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        print("[ERROR] Il JSON deve essere una lista.", file=sys.stderr)
        sys.exit(1)
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict) and "id" in item:
            ids.append(str(item["id"]))
        elif isinstance(item, str):
            ids.append(item)
    if not ids:
        print("[ERROR] Nessun id trovato nel JSON.", file=sys.stderr); sys.exit(2)

    print(f"[INFO] Assemblies da processare: {', '.join(ids)}", file=sys.stderr)
    all_ok = True
    for asm_id in ids:
        gtf_path   = gtf_tmpl.format(id=asm_id, assembly=asm_id)
        fasta_path = fasta_tmpl.format(id=asm_id, assembly=asm_id)
        outdir     = os.path.join(processed_base, asm_id, "codons_info")
        ensure_dir(outdir)
        ok = process_assembly(asm_id, gtf_path, fasta_path, outdir)
        all_ok = all_ok and ok

    if not all_ok:
        print("\n[WARN] Completato con alcuni errori.", file=sys.stderr)
        sys.exit(1)
    print("\n[ALL DONE]")

def main():
    ap = argparse.ArgumentParser(
        description="Codon indexing from GTF/CDS (triplette) con salvataggio stile k-mer: mapping, NPZ per-chr e meta."
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--assemblies-json", help="Lista assemblies (stringhe o oggetti con 'id').")
    mode.add_argument("--assembly-id", help="Esegui un singolo assembly.")

    ap.add_argument("--gtf", help="Path GTF per --assembly-id (può essere .gz).")
    ap.add_argument("--fasta", help="Path FASTA .gz per --assembly-id.")

    ap.add_argument("--gtf-template", help="Template GTF per batch, es: data/raw/{assembly}/genes.gtf.gz",
                    default="data/raw/{assembly}/{assembly}.genes.gtf.gz")
    ap.add_argument("--fasta-template", help="Template FASTA .gz per batch, es: data/raw/{assembly}/{assembly}.genome.fas.gz",
                    default="data/raw/{assembly}/{assembly}.genome.fas.gz")

    ap.add_argument("--processed-base", default="data/processed", help="Base output (default: data/processed)")

    args = ap.parse_args()

    if args.assemblies_json:
        if not args.gtf_template or not args.fasta_template:
            print("[ERROR] In batch servono --gtf-template e --fasta-template.", file=sys.stderr)
            sys.exit(2)
        run_ids_from_json(args.assemblies_json, args.processed_base, args.gtf_template, args.fasta_template)
        return

    # singolo assembly
    if not args.assembly_id or not args.gtf or not args.fasta:
        print("[ERROR] In singolo: servono --assembly-id, --gtf, --fasta.", file=sys.stderr)
        sys.exit(3)

    outdir = os.path.join(args.processed_base, args.assembly_id, "codons_info")
    ensure_dir(outdir)
    ok = process_assembly(args.assembly_id, args.gtf, args.fasta, outdir)
    if not ok:
        sys.exit(1)
    print("\n[ALL DONE]")

if __name__ == "__main__":
    main()
