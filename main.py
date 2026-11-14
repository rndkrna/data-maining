"""
=============================================================
  SCRAPER BERITA POLITIK INDONESIA
  Untuk Keperluan Data Mining - Analisis Sentimen
=============================================================
  Sumber : Kompas, Detik, CNN Indonesia, Tribun, Antara
  Output : berita_politik.xlsx
=============================================================
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime

# Set encoding stdout ke UTF-8 untuk menghindari error Unicode di Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Tambahkan root direktori ke path agar import scrapers bekerja
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrapers import (
    KompasScraper,
    DetikScraper,
    CNNScraper,
    TribunScraper,
    AntaraScraper,
)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# Jumlah artikel yang ingin diambil per situs
ARTIKEL_PER_SITUS = 100

# Nama file output
OUTPUT_FILE = "berita_politik.xlsx"

# Pilih situs yang ingin dijalankan (True = aktif, False = skip)
SITUS_AKTIF = {
    "Kompas"       : True,
    "Detik"        : True,
    "CNN Indonesia": True,
    "Tribun News"  : True,
    "Antara News"  : True,
}

# ─────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────

def cetak_separator(karakter: str = "=", lebar: int = 60):
    print(karakter * lebar)


def jalankan_semua_scraper(max_per_situs: int) -> dict[str, list[dict]]:
    """
    Menjalankan semua scraper yang aktif dan mengembalikan hasilnya.

    Returns:
        Dict dengan key nama sumber dan value list data artikel
    """
    # Mapping nama ke kelas scraper
    semua_scraper = {
        "Kompas"       : KompasScraper,
        "Detik"        : DetikScraper,
        "CNN Indonesia": CNNScraper,
        "Tribun News"  : TribunScraper,
        "Antara News"  : AntaraScraper,
    }

    hasil_semua = {}

    for nama, KelaseScraper in semua_scraper.items():
        if not SITUS_AKTIF.get(nama, False):
            print(f"\n[SKIP] {nama} dinonaktifkan di konfigurasi.")
            continue

        cetak_separator("-")
        scraper = KelaseScraper()
        try:
            data = scraper.get_artikel(max_artikel=max_per_situs)
            hasil_semua[nama] = data
        except KeyboardInterrupt:
            print(f"\n[!] Scraping {nama} dibatalkan oleh pengguna.")
            if nama in hasil_semua:
                pass  # Tetap simpan yang sudah ada
            break
        except Exception as e:
            print(f"\n[ERROR] Scraper {nama} gagal: {e}")
            hasil_semua[nama] = []

    return hasil_semua


def simpan_ke_excel(hasil_semua: dict[str, list[dict]], nama_file: str):
    """
    Menggabungkan semua data dan menyimpan ke file Excel.
    - Sheet 'Semua_Data': gabungan semua sumber
    - Sheet per sumber: data per media
    """
    if not hasil_semua:
        print("\n[!] Tidak ada data untuk disimpan.")
        return

    # Gabungkan semua data
    semua_data = []
    for nama_sumber, data_list in hasil_semua.items():
        semua_data.extend(data_list)

    df_semua = pd.DataFrame(semua_data)

    # Bersihkan duplikasi data (URL atau judul sama)
    if not df_semua.empty:
        sebelum = len(df_semua)
        if "url" in df_semua.columns:
            df_semua.drop_duplicates(subset=["url"], keep="first", inplace=True)
        if "judul" in df_semua.columns:
            df_semua.drop_duplicates(subset=["judul"], keep="first", inplace=True)
        if len(df_semua) < sebelum:
            print(f"  [i] Ditemukan dan dibersihkan {sebelum - len(df_semua)} artikel berita duplikat.")
        df_semua.reset_index(drop=True, inplace=True)
        df_semua.insert(0, "no", range(1, len(df_semua) + 1))
        df_semua["sentimen"] = ""  # Diisi otomatis oleh InSet Lexicon via sentiment_modeling.py

    print(f"\n[SAVE] Menyimpan ke {nama_file}...")

    with pd.ExcelWriter(nama_file, engine="openpyxl") as writer:
        # Sheet utama: semua data
        if not df_semua.empty:
            df_semua.to_excel(writer, sheet_name="Semua_Data", index=False)
            print(f"  + Sheet 'Semua_Data': {len(df_semua)} baris")

        # Sheet per sumber
        for nama_sumber, data_list in hasil_semua.items():
            if not data_list:
                continue
            df_per_sumber = pd.DataFrame(data_list)
            df_per_sumber.insert(0, "no", range(1, len(df_per_sumber) + 1))
            df_per_sumber["sentimen"] = ""

            # Nama sheet max 31 karakter (batasan Excel)
            nama_sheet = nama_sumber[:31]
            df_per_sumber.to_excel(writer, sheet_name=nama_sheet, index=False)
            print(f"  + Sheet '{nama_sheet}': {len(df_per_sumber)} baris")

    print(f"\n[OK] Berhasil disimpan: {os.path.abspath(nama_file)}")


def cetak_ringkasan(hasil_semua: dict[str, list[dict]], waktu_mulai: float):
    """Mencetak ringkasan hasil scraping."""
    cetak_separator()
    print("RINGKASAN HASIL SCRAPING")
    cetak_separator()

    total = 0
    for nama, data in hasil_semua.items():
        jumlah = len(data)
        total += jumlah
        status = "+" if jumlah > 0 else "-"
        print(f"  [{status}] {nama:<20} : {jumlah} artikel")

    cetak_separator("-")
    waktu_selesai = time.time() - waktu_mulai
    menit = int(waktu_selesai // 60)
    detik = int(waktu_selesai % 60)
    print(f"  TOTAL               : {total} artikel")
    print(f"  Waktu               : {menit} menit {detik} detik")
    cetak_separator()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scraper Berita Politik Indonesia untuk Analisis Sentimen"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=ARTIKEL_PER_SITUS,
        help=f"Jumlah artikel per situs (default: {ARTIKEL_PER_SITUS})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help=f"Nama file output Excel (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--situs",
        type=str,
        nargs="+",
        choices=["kompas", "detik", "cnn", "tribun", "antara"],
        help="Pilih situs tertentu (contoh: --situs kompas detik)"
    )
    args = parser.parse_args()

    # Override situs aktif jika pengguna memilih secara spesifik
    if args.situs:
        peta_nama = {
            "kompas": "Kompas",
            "detik": "Detik",
            "cnn": "CNN Indonesia",
            "tribun": "Tribun News",
            "antara": "Antara News",
        }
        for key in SITUS_AKTIF:
            SITUS_AKTIF[key] = False
        for s in args.situs:
            nama_asli = peta_nama.get(s)
            if nama_asli:
                SITUS_AKTIF[nama_asli] = True

    # Header
    cetak_separator()
    print("SCRAPER BERITA POLITIK INDONESIA")
    print(f"   Waktu Mulai : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target      : {args.max} artikel per situs")
    print(f"   Output      : {args.output}")
    cetak_separator()

    waktu_mulai = time.time()

    # Jalankan scraping
    hasil = jalankan_semua_scraper(max_per_situs=args.max)

    # Simpan ke Excel
    simpan_ke_excel(hasil, args.output)

    # Ringkasan
    cetak_ringkasan(hasil, waktu_mulai)

    print("\n[INFO] Data scraping selesai disimpan ke", args.output)
    print("       Langkah berikutnya: Jalankan 'python sentiment_modeling.py' untuk pelabelan otomatis berbasis InSet Lexicon dan pelatihan model.\n")

