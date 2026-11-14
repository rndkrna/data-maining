from .base_scraper import BaseScraper


class AntaraScraper(BaseScraper):
    """
    Scraper untuk Antara News — Kategori Politik.
    URL: https://www.antaranews.com/berita/politik
    """

    INDEKS_URL = "https://www.antaranews.com/politik/{page}"
    BASE_URL = "https://www.antaranews.com"

    def __init__(self):
        super().__init__(nama_sumber="Antara News")

    def _ambil_daftar_url(self, max_artikel: int) -> list[str]:
        """Mengambil daftar URL artikel dari halaman politik Antara."""
        urls = []
        halaman = 1

        while len(urls) < max_artikel:
            page_url = "https://www.antaranews.com/politik" if halaman == 1 else self.INDEKS_URL.format(page=halaman)
            print(f"  [Antara] Mengambil daftar halaman {halaman}...")
            soup = self.get_page(page_url)
            if not soup:
                break

            ditemukan_baru = False
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href.startswith("http"):
                    href = self.BASE_URL + href
                if "antaranews.com/berita/" in href:
                    clean_url = href.split("?")[0]
                    if clean_url not in urls:
                        urls.append(clean_url)
                        ditemukan_baru = True
                    if len(urls) >= max_artikel:
                        break

            if not ditemukan_baru:
                print(f"  [Antara] Tidak ada artikel baru di halaman {halaman}, berhenti.")
                break

            halaman += 1

        return urls[:max_artikel]

    def _parse_artikel(self, url: str) -> dict:
        """Mengambil dan mem-parsing satu halaman artikel Antara."""
        data = self._buat_data_kosong(url)
        soup = self.get_page(url)
        if not soup:
            return data

        # Judul
        judul_tag = soup.select_one("h1.post-title") or soup.select_one("h1[class*='title']") or soup.select_one("h1")
        if judul_tag:
            data["judul"] = judul_tag.get_text(strip=True)

        # Tanggal
        tanggal_tag = (
            soup.select_one("span.post-date") or
            soup.select_one("time") or
            soup.select_one("[class*='date']")
        )
        if tanggal_tag:
            data["tanggal"] = tanggal_tag.get("datetime", "") or tanggal_tag.get_text(strip=True)

        # Penulis
        penulis_tag = soup.select_one("span.reporter a") or soup.select_one("[class*='author']")
        if penulis_tag:
            data["penulis"] = penulis_tag.get_text(strip=True)

        # Kategori
        kategori_tag = soup.select_one(".breadcrumb a:last-of-type") or soup.select_one("[class*='category']")
        if kategori_tag:
            data["kategori"] = kategori_tag.get_text(strip=True)

        # Tags
        tags_list = soup.select("div.tags-content a") or soup.select("[class*='tag'] a")
        if tags_list:
            data["tags"] = ", ".join(t.get_text(strip=True) for t in tags_list)

        # Isi Berita
        isi_div = (
            soup.select_one("div.post-content") or
            soup.select_one("div[class*='content']") or
            soup.select_one("article div[itemprop='articleBody']")
        )
        if isi_div:
            for tag_buang in isi_div.select("script, style, .ads, figure, .related-article"):
                tag_buang.decompose()
            paragraf = isi_div.find_all("p")
            data["isi_berita"] = " ".join(p.get_text(strip=True) for p in paragraf if p.get_text(strip=True))

        return data

    def get_artikel(self, max_artikel: int = 100) -> list[dict]:
        print(f"\n[Antara News] Memulai scraping {max_artikel} artikel...")
        urls = self._ambil_daftar_url(max_artikel)
        print(f"[Antara News] Ditemukan {len(urls)} URL. Mulai parsing artikel...")

        hasil = []
        for i, url in enumerate(urls, 1):
            print(f"  [Antara] ({i}/{len(urls)}) {url[:70]}...")
            data = self._parse_artikel(url)
            if data["judul"]:
                hasil.append(data)

        print(f"[Antara News] Selesai. Berhasil: {len(hasil)} artikel.")
        return hasil
