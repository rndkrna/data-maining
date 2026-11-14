from .base_scraper import BaseScraper


class DetikScraper(BaseScraper):
    """
    Scraper untuk Detik.com — Kategori Berita Nasional.
    URL: https://news.detik.com/berita
    """

    INDEKS_URL = "https://news.detik.com/berita?page={page}"

    def __init__(self):
        super().__init__(nama_sumber="Detik")

    def _ambil_daftar_url(self, max_artikel: int) -> list[str]:
        """Mengambil daftar URL artikel dari halaman kategori Detik."""
        urls = []
        halaman = 1

        while len(urls) < max_artikel:
            print(f"  [Detik] Mengambil daftar halaman {halaman}...")
            soup = self.get_page(self.INDEKS_URL.format(page=halaman))
            if not soup:
                break

            # Artikel ada di dalam artikel dengan class "list-content__item"
            artikel_links = soup.select("article.list-content__item h3.media__title a")
            if not artikel_links:
                artikel_links = soup.select(".media__title a")

            if not artikel_links:
                print(f"  [Detik] Tidak ada artikel di halaman {halaman}, berhenti.")
                break

            ditemukan_baru = False
            for a in artikel_links:
                href = a.get("href", "")
                if href and href.startswith("http") and href not in urls:
                    # Filter hanya URL berita (bukan video, dll)
                    if "/berita/" in href or "/news/" in href:
                        urls.append(href)
                        ditemukan_baru = True
                if len(urls) >= max_artikel:
                    break

            if not ditemukan_baru:
                print(f"  [Detik] Tidak ada artikel baru di halaman {halaman}, berhenti.")
                break

            halaman += 1

        return urls[:max_artikel]

    def _parse_artikel(self, url: str) -> dict:
        """Mengambil dan mem-parsing satu halaman artikel Detik."""
        data = self._buat_data_kosong(url)
        soup = self.get_page(url)
        if not soup:
            return data

        # Judul
        judul_tag = soup.select_one("h1.detail__title") or soup.select_one("h1")
        if judul_tag:
            data["judul"] = judul_tag.get_text(strip=True)

        # Tanggal
        tanggal_tag = soup.select_one(".detail__date") or soup.select_one("div[class*='date']")
        if tanggal_tag:
            data["tanggal"] = tanggal_tag.get_text(strip=True)

        # Penulis
        penulis_tag = soup.select_one(".detail__author") or soup.select_one("[class*='author']")
        if penulis_tag:
            data["penulis"] = penulis_tag.get_text(strip=True)

        # Kategori — dari breadcrumb
        breadcrumb = soup.select(".nav-breadcrumb a")
        if len(breadcrumb) > 1:
            data["kategori"] = breadcrumb[-1].get_text(strip=True)

        # Tags
        tags_list = soup.select(".nav-tagging a") or soup.select("[class*='tag'] a")
        if tags_list:
            data["tags"] = ", ".join(t.get_text(strip=True) for t in tags_list)

        # Isi Berita
        isi_div = soup.select_one("div.detail__body-text") or soup.select_one("div[class*='body']")
        if isi_div:
            for tag_buang in isi_div.select("script, style, .ads, ins, .detail__cta-btn"):
                tag_buang.decompose()
            paragraf = isi_div.find_all("p")
            data["isi_berita"] = " ".join(p.get_text(strip=True) for p in paragraf if p.get_text(strip=True))

        return data

    def get_artikel(self, max_artikel: int = 100) -> list[dict]:
        print(f"\n[Detik] Memulai scraping {max_artikel} artikel...")
        urls = self._ambil_daftar_url(max_artikel)
        print(f"[Detik] Ditemukan {len(urls)} URL. Mulai parsing artikel...")

        hasil = []
        for i, url in enumerate(urls, 1):
            print(f"  [Detik] ({i}/{len(urls)}) {url[:70]}...")
            data = self._parse_artikel(url)
            if data["judul"]:
                hasil.append(data)

        print(f"[Detik] Selesai. Berhasil: {len(hasil)} artikel.")
        return hasil
