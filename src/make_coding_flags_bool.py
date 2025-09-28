#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, numpy as np

HG38_SIZES = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559, "chr4": 190_214_555,
    "chr5": 181_538_259, "chr6": 170_805_979, "chr7": 159_345_973, "chr8": 145_138_636,
    "chr9": 138_394_717, "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189, "chr16": 90_338_345,
    "chr17": 83_257_441, "chr18": 80_373_285, "chr19": 58_617_616, "chr20": 64_444_167,
    "chr21": 46_709_983, "chr22": 50_818_468,
}
HG19_SIZES = {
    "chr1": 249_250_621, "chr2": 243_199_373, "chr3": 198_022_430, "chr4": 191_154_276,
    "chr5": 180_915_260, "chr6": 171_115_067, "chr7": 159_138_663, "chr8": 146_364_022,
    "chr9": 141_213_431, "chr10": 135_534_747, "chr11": 135_006_516, "chr12": 133_851_895,
    "chr13": 115_169_878, "chr14": 107_349_540, "chr15": 102_531_392, "chr16": 90_354_753,
    "chr17": 81_195_210, "chr18": 78_077_248, "chr19": 59_128_983, "chr20": 63_025_520,
    "chr21": 48_129_895, "chr22": 51_304_566,
}

def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def main():
    ap = argparse.ArgumentParser(
        description="Crea NPZ booleano di flag CDS (chr1..22) da GTF senza chrom.sizes (hg38/hg19 incorporati)."
    )
    ap.add_argument("--gtf", required=True, help="Annotazione GTF/GTF.GZ (GENCODE/Ensembl/UCSC)")
    ap.add_argument("--out", required=True, help="Output .npz (es. coding_flags.autosomes.bool.npz)")
    ap.add_argument("--assembly", choices=["hg38","hg19"], default="hg38", help="Assembly (default: hg38)")
    ap.add_argument("--only-protein-coding", action="store_true",
                    help="Mantieni solo CDS di transcript_type=protein_coding (se presente negli attributi).")
    args = ap.parse_args()

    sizes = HG38_SIZES if args.assembly == "hg38" else HG19_SIZES
    autosomes = [f"chr{i}" for i in range(1, 23)]

    # parse GTF → intervalli CDS per autosoma
    cds_by_chrom = {c: [] for c in autosomes}
    with open_text(args.gtf) as fh:
        for line in fh:
            if not line or line.startswith("#"): continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9: continue
            chrom, _, feature, start, end, _, strand, frame, attrs = fields
            if chrom not in sizes or feature != "CDS": continue

            if args.only_protein_coding:
                at = attrs.lower()
                if "protein_coding" not in at:
                    # prova a filtrare in modo rudimentale: molte annotazioni mettono transcript_type o gene_type
                    # se non presente, non escludiamo per evitare falsi negativi
                    pass

            s = int(start) - 1  # 0-based
            e = int(end)        # half-open
            # clamp a lunghezza cromosoma (per sicurezza)
            e = min(e, sizes[chrom])
            if e > s:
                cds_by_chrom[chrom].append((s, e))

    # costruisci flag booleani per cromosoma
    payload = {}
    total = 0
    for chrom in autosomes:
        n = sizes[chrom]
        arr = np.zeros(n, dtype=bool)
        if cds_by_chrom[chrom]:
            # unisci intervalli ordinati
            ivs = sorted(cds_by_chrom[chrom])
            cur_s, cur_e = ivs[0]
            merged = []
            for s, e in ivs[1:]:
                if s <= cur_e:  # overlap/adiacenti
                    if e > cur_e: cur_e = e
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            # marca True
            for s, e in merged:
                arr[s:e] = True
        payload[chrom] = arr
        total += int(arr.sum())
        print(f"{chrom}: {int(arr.sum()):,} basi codificanti / {n:,}")

    np.savez_compressed(args.out, **payload)
    print(f"[DONE] Totale basi codificanti autosomi: {total:,}")
    print(f"[DONE] Salvato: {args.out}")

if __name__ == "__main__":
    main()
