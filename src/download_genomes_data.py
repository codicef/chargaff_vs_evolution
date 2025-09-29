#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import shutil
import urllib.request
import urllib.error
from tqdm import tqdm
import re
from html.parser import HTMLParser

# --------------------------
# Config
# --------------------------
BASE = "https://hgdownload.soe.ucsc.edu/goldenPath"
ROOT_OUT = os.path.join("data", "raw")

# Assemblies con phyloP >= 20 (dalla tua lista)
WAY = {
    "hg38": 470,
    "tarSyr2": 20,
    "rn6": 20,
    "mm10": 60,
    "dm6": 124,
    "ce11": 135,
    "wuhCor1": 119,
    "eboVir3": 160,
}

ASSEMBLIES = list(WAY.keys())

UA = "Mozilla/5.0 (compatible; UCSC-fetch/1.0; +https://hgdownload.soe.ucsc.edu)"

# --------------------------
# Helpers
# --------------------------
def human_bytes(n: int) -> str:
    if n is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def urlopen_simple(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)

def urlopen_with_range(url: str, start: int = 0, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if start > 0:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout=timeout)

def download(url: str, out_path: str, retries: int = 3, resume: bool = True) -> int:
    """
    Scarica url -> out_path con resume. Ritorna i byte finali del file.
    """
    ensure_dir(os.path.dirname(out_path))
    tmp_path = out_path + ".part"

    # size già presente (resume)
    existing = 0
    if resume and os.path.exists(tmp_path):
        existing = os.path.getsize(tmp_path)

    attempt = 0
    while True:
        attempt += 1
        try:
            # HEAD-like content-length (best effort)
            content_length = None
            try:
                head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
                with urllib.request.urlopen(head_req, timeout=15) as r:
                    cl = r.headers.get("Content-Length")
                    if cl is not None:
                        content_length = int(cl)
            except Exception:
                content_length = None

            # se esiste già il file completo, salta
            if os.path.exists(out_path):
                return os.path.getsize(out_path)

            # apri in append binario
            mode = "ab" if existing > 0 else "wb"
            with open(tmp_path, mode) as fh:
                start = existing if resume else 0
                with urlopen_with_range(url, start=start, timeout=60) as resp:
                    # Se server non supporta Range e noi avevamo partial, ricomincia da zero
                    if start > 0 and resp.getcode() not in (206, 200):
                        # riparti da zero
                        fh.close()
                        os.remove(tmp_path)
                        existing = 0
                        if attempt <= retries:
                            time.sleep(1.0 * attempt)
                            continue
                        else:
                            raise RuntimeError(f"Server non supporta resume per {url}")

                    # flusso di lettura
                    chunk = 1024 * 1024
                    downloaded = existing
                    t0 = time.time()
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
                        downloaded += len(buf)

                        # progress minimale su stdout
                        if content_length:
                            pct = 100.0 * downloaded / content_length
                            rate = downloaded / max(1e-9, (time.time() - t0))
                            sys.stdout.write(
                                f"\r  -> {os.path.basename(out_path)}: {human_bytes(downloaded)}/{human_bytes(content_length)} ({pct:.1f}%) @ {human_bytes(int(rate))}/s"
                            )
                        else:
                            sys.stdout.write(f"\r  -> {os.path.basename(out_path)}: {human_bytes(downloaded)}")
                        sys.stdout.flush()
                sys.stdout.write("\n")

            # rinomina atomica
            os.replace(tmp_path, out_path)
            return os.path.getsize(out_path)

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt <= retries:
                wait = 1.0 * attempt
                print(f"    Avviso: errore '{e}'. Retry {attempt}/{retries} tra {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                raise

def total_target_bytes(root: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".genome.fas.gz") or fn.endswith(".phylop.bw") or fn.endswith(".genes.gtf.gz"):
                fp = os.path.join(dirpath, fn)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total

def print_free_space(path: str):
    usage = shutil.disk_usage(path)
    print(f"Spazio totale: {human_bytes(usage.total)}")
    print(f"Spazio usato  : {human_bytes(usage.used)}")
    print(f"Spazio libero : {human_bytes(usage.free)}")

# --------------------------
# HTML index parsing (genes/)
# --------------------------
class SimpleIndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

def pick_best_gtf(asm: str) -> str | None:
    """
    Restituisce l'URL della migliore GTF per l'assembly, cercando in:
      BASE/<asm>/bigZips/genes/
    Ordine di preferenza: GENCODE (*.gtf.gz con 'gencode'), poi ensGene.gtf.gz,
    poi refGene.gtf.gz, poi knownGene.gtf.gz. Se più GENCODE, prende la versione
    con numero 'vN' più alto (best effort).
    """
    index_url = f"{BASE}/{asm}/bigZips/genes/"
    try:
        with urlopen_simple(index_url, timeout=20) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    parser = SimpleIndexParser()
    parser.feed(html)
    files = [h for h in parser.links if h.lower().endswith(".gtf.gz")]

    # 1) GENCODE
    gencode = [f for f in files if re.search(r"gencode.*\.gtf\.gz$", f, re.IGNORECASE)]
    if gencode:
        def extract_num(s):
            m = re.search(r'v(\d+)', s)
            return int(m.group(1)) if m else -1
        best = sorted(gencode, key=extract_num)[-1]
        return index_url + best

    # 2) Ensembl
    if "ensGene.gtf.gz" in files:
        return index_url + "ensGene.gtf.gz"
    # 3) RefSeq
    if "refGene.gtf.gz" in files:
        return index_url + "refGene.gtf.gz"
    # 4) knownGene
    if "knownGene.gtf.gz" in files:
        return index_url + "knownGene.gtf.gz"

    # Fallback: se c'è un unico .gtf.gz diverso, prendi quello
    if files:
        return index_url + files[-1]

    return None

# --------------------------
# Main
# --------------------------
def main():
    ensure_dir(ROOT_OUT)

    for asm in tqdm(ASSEMBLIES):
        way = WAY[asm]
        outdir = os.path.join(ROOT_OUT, asm)
        ensure_dir(outdir)

        fasta_url = f"{BASE}/{asm}/bigZips/{asm}.fa.gz"
        bw_url = f"{BASE}/{asm}/phyloP{way}way/{asm}.phyloP{way}way.bw"
        gtf_url = pick_best_gtf(asm)

        fasta_out = os.path.join(outdir, f"{asm}.genome.fas.gz")
        bw_out = os.path.join(outdir, f"{asm}.phylop.bw")
        gtf_out = os.path.join(outdir, f"{asm}.genes.gtf.gz") if gtf_url else None

        # FASTA
        if not os.path.exists(fasta_out):
            print(f"==> {asm} (phyloP {way}way)")
            print(f"    FASTA  : {fasta_url}")
            size_fa = download(fasta_url, fasta_out, retries=3, resume=True)
            print(f"    Scaricati: {human_bytes(size_fa)}")
        else:
            print(f"==> {asm} (phyloP {way}way) - FASTA già presente, salto download.")

        # phyloP
        if not os.path.exists(bw_out):
            print(f"    phyloP : {bw_url}")
            try:
                size_bw = download(bw_url, bw_out, retries=3, resume=True)
            except Exception as e:
                print(f"    Avviso: impossibile scaricare phyloP per {asm}: {e}")
                bw_url_alt = f"{BASE}/{asm}/phyloP{way}way/{asm}.{way}way.phyloP{way}way.bw"
                print(f"    Provo con URL alternativo: {bw_url_alt}")
                try:
                    size_bw = download(bw_url_alt, bw_out, retries=3, resume=True)
                except Exception as e2:
                    print(f"    Errore: impossibile scaricare phyloP per {asm}: {e2}")
        else:
            print(f"    phyloP già presente, salto download.")

        # GTF (annotazioni geni)
        if gtf_url and gtf_out:
            if not os.path.exists(gtf_out):
                print(f"    GTF    : {gtf_url}")
                try:
                    size_gtf = download(gtf_url, gtf_out, retries=3, resume=True)
                    print(f"    Scaricati: {human_bytes(size_gtf)} (GTF)")
                except Exception as e:
                    print(f"    Avviso: impossibile scaricare GTF per {asm}: {e}")
            else:
                print(f"    GTF    : già presente, salto download.")
        else:
            print(f"    GTF    : non trovato su UCSC (bigZips/genes/), salto.")

        print("    PhyloP/FASTA/GTF done")

    print("\nCalcolo spazio occupato dai file target (.fas.gz + .bw + .gtf.gz)...")
    tot_bytes = total_target_bytes(ROOT_OUT)
    print(f"Totale occupato: {human_bytes(tot_bytes)}")

    print(f"\nSpazio libero sul filesystem che contiene '{ROOT_OUT}':")
    print_free_space(ROOT_OUT)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nErrore: {e}", file=sys.stderr)
        sys.exit(1)
