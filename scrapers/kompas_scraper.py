from datetime import datetime, timedelta
from .base_scraper import BaseScraper


class KompasScraper(BaseScraper):
    """
    Scraper untuk Kompas.com — Kategori Nasional/Politik.
    URL Indeks: https://indeks.kompas.com/?site=nasional
    """

    BASE_URL = "https://indeks.kompas.com/"
    INDEKS_URL = "https://indeks.kompas.com/?site=nasional&date={date}&page={page}"

    def __init__(self):
        super().__init__(nama_sumber="Kompas")

    def _ambil_daftar_url(self, max_artikel: int) -> list[str]:
        """Mengambil daftar URL artikel dari halaman indeks Kompas dengan navigasi tanggal."""
        urls = []
        current_date = datetime.now()
        max_mundur_hari = 60  # Batas maksimum mundur hari

        hari_terlewati = 0
        while len(urls) < max_artikel and hari_terlewati < max_mundur_hari:
            date_str = current_date.strftime("%Y-%m-%d")
            halaman = 1

            while len(urls) < max_artikel:
                page_url = self.INDEKS_URL.format(date=date_str, page=halaman)
                print(f"  [Kompas] Mengambil tanggal {date_str} (halaman {halaman})...")
                soup = self.get_page(page_url)
                if not soup:
                    break

                items = soup.select(".articleItem a")
                if not items:
                    items = [a for a in soup.find_all("a", href=True) if "/read/" in a.get("href", "")]

                ditemukan_baru = False
                for a in items:
                    href = a.get("href", "")
                    if "/read/" in href and "kompas.com" in href:
                        clean_url = href.split("?")[0]
                        # Abaikan tautan footer/penghargaan statis
                        if "kompascom-kembali-jadi-pemenang" in clean_url:
                            continue
                        if clean_url not in urls:
                            urls.append(clean_url)
                            ditemukan_baru = True
                        if len(urls) >= max_artikel:
                            break

                if not ditemukan_baru:
                    # Habis untuk tanggal ini, lanjut ke hari kemarin
                    break

                halaman += 1

            current_date -= timedelta(days=1)
            hari_terlewati += 1

        return urls[:max_artikel]

    def _parse_artikel(self, url: str) -> dict:
        """Mengambil dan mem-parsing satu halaman artikel Kompas."""
        data = self._buat_data_kosong(url)
        soup = self.get_page(url)
        if not soup:
            return data

        # Judul
        judul_tag = soup.select_one("h1.read__title") or soup.select_one("h1.article__title")
        if judul_tag:
            data["judul"] = judul_tag.get_text(strip=True)

        # Tanggal
        tanggal_tag = soup.select_one(".read__time") or soup.select_one("div.article__date")
        if tanggal_tag:
            data["tanggal"] = tanggal_tag.get_text(strip=True)

        # Penulis
        penulis_tag = soup.select_one(".read__credit a") or soup.select_one(".credit-title-name")
        if penulis_tag:
            data["penulis"] = penulis_tag.get_text(strip=True)

        # Kategori
        kategori_tag = soup.select_one(".read__breadcrumb a:last-child") or soup.select_one(".article__subtitle")
        if kategori_tag:
            data["kategori"] = kategori_tag.get_text(strip=True)

        # Tags
        tags_list = soup.select(".tag-link__list a") or soup.select(".article__tag a")
        if tags_list:
            data["tags"] = ", ".join(t.get_text(strip=True) for t in tags_list)

        # Isi Berita — gabungkan semua paragraf
        isi_div = soup.select_one("div.read__content") or soup.select_one("div.article__body")
        if isi_div:
            # Hapus elemen yang tidak diinginkan (iklan, rekomendasi, dll)
            for tag_buang in isi_div.select("script, style, .ber-artikel, .tag-link__list, .read__recc"):
                tag_buang.decompose()
            paragraf = isi_div.find_all("p")
            data["isi_berita"] = " ".join(p.get_text(strip=True) for p in paragraf if p.get_text(strip=True))

        return data

    def get_artikel(self, max_artikel: int = 100) -> list[dict]:
        print(f"\n[Kompas] Memulai scraping {max_artikel} artikel...")
        urls = self._ambil_daftar_url(max_artikel)
        print(f"[Kompas] Ditemukan {len(urls)} URL. Mulai parsing artikel...")

        hasil = []
        for i, url in enumerate(urls, 1):
            print(f"  [Kompas] ({i}/{len(urls)}) {url[:70]}...")
            data = self._parse_artikel(url)
            if data["judul"]:  # Hanya simpan jika berhasil dapat judul
                hasil.append(data)

        print(f"[Kompas] Selesai. Berhasil: {len(hasil)} artikel.")
        return hasil
