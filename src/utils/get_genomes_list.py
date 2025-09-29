#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ftplib import FTP

def list_ftp_dirs(host, path=""):
    ftp = FTP(host)
    ftp.login()  # accesso anonimo
    ftp.cwd(path)

    entries = []
    ftp.retrlines("LIST", entries.append)

    dirs = []
    for entry in entries:
        parts = entry.split()
        # formato tipo: drwxr-xr-x   2 585      99          4096 Sep 27  2023 goldenPath
        if entry.startswith("d"):  # directory
            dirs.append(parts[-1])

    ftp.quit()
    return dirs

if __name__ == "__main__":
    host = "hgdownload.soe.ucsc.edu"
    # dirs /apache/htdocs/goldenPath
    dirs = list_ftp_dirs(host, "/apache/htdocs/goldenPath")
    print("Cartelle su", host, ":")
    for d in dirs:
        print(d)
