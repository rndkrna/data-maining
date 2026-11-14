from .base_scraper import BaseScraper


class CNNScraper(BaseScraper):
    """
    Scraper untuk CNN Indonesia — Kategori Nasional / Politik.
    URL: https://www.cnnindonesia.com/nasional/politik
    """

    INDEKS_URL = "https://www.cnnindonesia.com/nasional/politik?page={page}"

    def __init__(self):
        super().__init__(nama_sumber="CNN Indonesia")

    def _ambil_daftar_url(self, max_artikel: int) -> list[str]:
        """Mengambil daftar URL artikel dari halaman kanal politik CNN Indonesia."""
        urls = []
        halaman = 1

        while len(urls) < max_artikel:
            print(f"  [CNN Indonesia] Mengambil daftar halaman {halaman}...")
            soup = self.get_page(self.INDEKS_URL.format(page=halaman))
            if not soup:
                break

            artikel_links = soup.select("article a") or soup.select(".media__title a")
            if not artikel_links:
                print(f"  [CNN Indonesia] Tidak ada artikel di halaman {halaman}, berhenti.")
                break

            ditemukan_baru = False
            for a in artikel_links:
                href = a.get("href", "")
                if href and href.startswith("http") and href not in urls:
                    if "/nasional/" in href or "/politik/" in href:
                        urls.append(href)
                        ditemukan_baru = True
                if len(urls) >= max_artikel:
                    break

            if not ditemukan_baru:
                print(f"  [CNN Indonesia] Tidak ada artikel baru di halaman {halaman}, berhenti.")
                break

            halaman += 1

        return urls[:max_artikel]

    def _parse_artikel(self, url: str) -> dict:
        """Mengambil dan mem-parsing satu halaman artikel CNN Indonesia."""
        data = self._buat_data_kosong(url)
        soup = self.get_page(url)
        if not soup:
            return data

        # Judul
        judul_tag = soup.select_one("h1.title") or soup.select_one("h1")
        if judul_tag:
            data["judul"] = judul_tag.get_text(strip=True)

        # Tanggal
        tanggal_tag = soup.select_one(".date") or soup.select_one("div[class*='date']")
        if tanggal_tag:
            data["tanggal"] = tanggal_tag.get_text(strip=True)

        # Penulis
        penulis_tag = soup.select_one(".author") or soup.select_one("[class*='author']")
        if penulis_tag:
            data["penulis"] = penulis_tag.get_text(strip=True)

        # Kategori
        data["kategori"] = "Politik"

        # Tags
        tags_list = soup.select(".tag a") or soup.select("[class*='tag'] a")
        if tags_list:
            data["tags"] = ", ".join(t.get_text(strip=True) for t in tags_list)

        # Isi Berita
        isi_div = soup.select_one("div.detail-text") or soup.select_one("div[class*='body']")
        if isi_div:
            for tag_buang in isi_div.select("script, style, .ads, ins, .paralax"):
                tag_buang.decompose()
            paragraf = isi_div.find_all("p")
            data["isi_berita"] = " ".join(p.get_text(strip=True) for p in paragraf if p.get_text(strip=True))

        return data

    def get_artikel(self, max_artikel: int = 100) -> list[dict]:
        print(f"\n[CNN Indonesia] Memulai scraping {max_artikel} artikel...")
        urls = self._ambil_daftar_url(max_artikel)
        print(f"[CNN Indonesia] Ditemukan {len(urls)} URL. Mulai parsing artikel...")

        hasil = []
        for i, url in enumerate(urls, 1):
            print(f"  [CNN Indonesia] ({i}/{len(urls)}) {url[:70]}...")
            data = self._parse_artikel(url)
            if data["judul"]:
                hasil.append(data)

        print(f"[CNN Indonesia] Selesai. Berhasil: {len(hasil)} artikel.")
        return hasil
