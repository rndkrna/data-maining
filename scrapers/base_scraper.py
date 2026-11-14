import requests
import time
import random
from bs4 import BeautifulSoup

class BaseScraper:
    """
    Kelas dasar untuk semua scraper berita.
    Menyediakan fungsi HTTP request dengan header management dan delay antar request.
    """

    HEADERS_LIST = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Accept-Language": "id,en-US;q=0.7,en;q=0.3",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
            "Accept-Language": "id-ID,id;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    ]

    def __init__(self, nama_sumber: str, delay_min: float = 0.2, delay_max: float = 0.5):
        """
        Args:
            nama_sumber: Nama media berita (contoh: "Kompas")
            delay_min: Jeda minimum antar request (detik)
            delay_max: Jeda maksimum antar request (detik)
        """
        self.nama_sumber = nama_sumber
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.session = requests.Session()

    def _get_headers(self) -> dict:
        """Mengembalikan header acak dari daftar."""
        return random.choice(self.HEADERS_LIST)

    def _delay(self):
        """Memberikan jeda acak agar tidak membebani server."""
        waktu_tunggu = random.uniform(self.delay_min, self.delay_max)
        time.sleep(waktu_tunggu)

    def get_page(self, url: str, timeout: int = 15) -> BeautifulSoup | None:
        """
        Mengambil halaman web dan mengembalikan objek BeautifulSoup.

        Args:
            url: URL yang ingin diambil
            timeout: Batas waktu request (detik)

        Returns:
            BeautifulSoup object atau None jika gagal
        """
        try:
            self._delay()
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return BeautifulSoup(response.text, "lxml")
        except requests.exceptions.HTTPError as e:
            print(f"  [!] HTTP Error {e.response.status_code} saat mengakses: {url}")
        except requests.exceptions.ConnectionError:
            print(f"  [!] Koneksi gagal: {url}")
        except requests.exceptions.Timeout:
            print(f"  [!] Timeout saat mengakses: {url}")
        except Exception as e:
            print(f"  [!] Error tidak terduga ({type(e).__name__}): {url}")
        return None

    def get_artikel(self, max_artikel: int = 100) -> list[dict]:
        """
        Method utama yang harus diimplementasikan oleh setiap scraper turunan.

        Returns:
            List of dict, setiap dict berisi data satu berita
        """
        raise NotImplementedError("Subclass harus mengimplementasikan method ini.")

    def _buat_data_kosong(self, url: str = "") -> dict:
        """Mengembalikan template data berita kosong."""
        return {
            "judul": "",
            "isi_berita": "",
            "tanggal": "",
            "penulis": "",
            "kategori": "Politik",
            "tags": "",
            "sumber": self.nama_sumber,
            "url": url,
        }
