from .base_scraper import BaseScraper


class TribunScraper(BaseScraper):
    """
    Scraper untuk Tribun News — Kategori Nasional.
    URL: https://www.tribunnews.com/nasional
    """

    INDEKS_URL = "https://www.tribunnews.com/nasional?page={page}"
    BASE_URL = "https://www.tribunnews.com"

    def __init__(self):
        super().__init__(nama_sumber="Tribun News")

    def _ambil_daftar_url(self, max_artikel: int) -> list[str]:
        """Mengambil daftar URL artikel dari halaman nasional Tribun."""
        urls = []
        halaman = 1

        while len(urls) < max_artikel:
            print(f"  [Tribun] Mengambil daftar halaman {halaman}...")
            soup = self.get_page(self.INDEKS_URL.format(page=halaman))
            if not soup:
                break

            ditemukan_baru = False
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href.startswith("http"):
                    href = self.BASE_URL + href
                # Filter URL artikel Tribun nasional (URL berita biasanya memuat /nasional/ dan angka ID /d- atau ID di URL)
                if "tribunnews.com/nasional/" in href or "/nasional/20" in href:
                    clean_url = href.split("?")[0]
                    # Abaikan URL index/halaman utama kategori
                    if clean_url.endswith("/nasional") or clean_url.endswith("/nasional/"):
                        continue
                    if clean_url not in urls:
                        urls.append(clean_url)
                        ditemukan_baru = True
                    if len(urls) >= max_artikel:
                        break

            if not ditemukan_baru:
                print(f"  [Tribun] Tidak ada artikel baru di halaman {halaman}, berhenti.")
                break

            halaman += 1

        return urls[:max_artikel]

    def _parse_artikel(self, url: str) -> dict:
        """Mengambil dan mem-parsing satu halaman artikel Tribun."""
        data = self._buat_data_kosong(url)
        soup = self.get_page(url)
        if not soup:
            return data

        # Judul
        judul_tag = soup.select_one("h1#arttitle") or soup.select_one("h1")
        if judul_tag:
            data["judul"] = judul_tag.get_text(strip=True)

        # Tanggal
        tanggal_tag = (
            soup.select_one("time") or
            soup.select_one("div.time") or
            soup.select_one("[class*='date']")
        )
        if tanggal_tag:
            data["tanggal"] = tanggal_tag.get("datetime", "") or tanggal_tag.get_text(strip=True)

        # Penulis
        penulis_tag = soup.select_one("div.reporter a") or soup.select_one("[class*='reporter']")
        if penulis_tag:
            data["penulis"] = penulis_tag.get_text(strip=True)

        # Kategori
        breadcrumb = soup.select("nav.breadcrumb span") or soup.select(".breadcrumb li")
        if len(breadcrumb) > 1:
            data["kategori"] = breadcrumb[-1].get_text(strip=True)

        # Tags
        tags_list = soup.select("div.tags a") or soup.select("[class*='tag'] a")
        if tags_list:
            data["tags"] = ", ".join(t.get_text(strip=True) for t in tags_list)

        # Isi Berita
        isi_div = soup.select_one("div#article-2") or soup.select_one("div[id*='article']")
        if not isi_div:
            isi_div = soup.select_one("div.side-article") or soup.select_one("article .content")
        if isi_div:
            for tag_buang in isi_div.select("script, style, .ads, .related-article, [class*='baca-juga']"):
                tag_buang.decompose()
            paragraf = isi_div.find_all("p")
            data["isi_berita"] = " ".join(p.get_text(strip=True) for p in paragraf if p.get_text(strip=True))

        return data

    def get_artikel(self, max_artikel: int = 100) -> list[dict]:
        print(f"\n[Tribun News] Memulai scraping {max_artikel} artikel...")
        urls = self._ambil_daftar_url(max_artikel)
        print(f"[Tribun News] Ditemukan {len(urls)} URL. Mulai parsing artikel...")

        hasil = []
        for i, url in enumerate(urls, 1):
            print(f"  [Tribun] ({i}/{len(urls)}) {url[:70]}...")
            data = self._parse_artikel(url)
            if data["judul"]:
                hasil.append(data)

        print(f"[Tribun News] Selesai. Berhasil: {len(hasil)} artikel.")
        return hasil
