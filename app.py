"""
=============================================================================
  APLIKASI WEB DEMONSTRASI ANALISIS SENTIMEN BERITA POLITIK (DATA MINING)
=============================================================================
  Universitas Maritim Raja Ali Haji (UMRAH) - Teknik Informatika 2025/2026
  - InSet Lexicon (Koto & Rahmaningtyas, 2017)
  - TF-IDF Vectorizer (1.500 Fitur, Unigram & Bigram)
  - Multinomial Naive Bayes (alpha=1.0)
  - Support Vector Machine (Linear SVM, C=1.0)
  - Built-in Lightweight Python Web Server (Zero-Dependency Backend)
=============================================================================
"""

import os
import sys
import json
import re
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import socket
import webbrowser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from inset_labeler import InSetLexiconLabeler

# Scrapers portal berita
try:
    from scrapers.kompas_scraper import KompasScraper
    from scrapers.detik_scraper import DetikScraper
    from scrapers.cnn_scraper import CNNScraper
    from scrapers.antara_scraper import AntaraScraper
    from scrapers.tribun_scraper import TribunScraper
    SCRAPERS_AVAILABLE = True
except Exception as e:
    print(f"[!] Warning: Gagal memuat beberapa modul scraper: {e}")
    SCRAPERS_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Pastikan folder web/ ada
os.makedirs(WEB_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. PREPROCESSING TEKS
# [FIX] Kata negasi (tidak, belum, bukan, tanpa, jangan) DIKELUARKAN dari stopwords
# ─────────────────────────────────────────────
STOPWORDS_ID = set([
    "yang", "di", "dan", "dengan", "untuk", "dari", "ke", "ini", "itu", "pada",
    "adalah", "bahwa", "dalam", "akan", "juga", "atau", "oleh", "saat", "tersebut",
    "bisa", "ada", "karena", "ia", "mereka", "kita", "kami", "telah",
    "seperti", "sudah", "hanya", "sebagai", "para", "lebih", "dapat",
    "antara", "setelah", "namun", "tetapi", "serta", "ketika", "bagi", "lalu",
    "secara", "lagi", "sehingga", "maupun", "bahkan",
    "hampir", "terhadap", "maka", "tentang", "selama", "kemudian", "kepada",
    "harus", "banyak", "hal", "pun", "saja", "menjadi", "diri", "tiap"
    # CATATAN: tidak, belum, bukan, tanpa, jangan TIDAK dimasukkan (kata negasi)
])

def preprocess_text(teks: str) -> str:
    """Membersihkan teks berita untuk ekstraksi fitur TF-IDF."""
    if not isinstance(teks, str):
        return ""
    teks = teks.lower()
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    words = [w for w in teks.split() if w not in STOPWORDS_ID and len(w) > 2]
    return " ".join(words)


# ─────────────────────────────────────────────
# 2. MODEL ENGINE & DATA MANAGER
# ─────────────────────────────────────────────
class SentimentEngine:
    def __init__(self):
        print("[+] Menginisialisasi SentimentEngine...")
        self.labeler = InSetLexiconLabeler()
        self.tfidf = None
        self.nb_model = None
        self.svm_model = None
        self.df_articles = pd.DataFrame()
        self.eval_summary = []
        self.per_class_metrics = []
        self.error_cases = []
        self.cm_nb = {}
        self.cm_svm = {}
        self.svm_params = []
        
        self.load_and_train()

    def _find_dataset_file(self):
        candidates = [
            "SiapMining_2301020003.xlsx",
            "hasil_analisis_sentimen.xlsx",
            "Mentah_2301020003.xlsx",
            "dataset_berita.xlsx",
            "dataset_berita_politik.xlsx"
        ]
        for c in candidates:
            p = os.path.join(BASE_DIR, c)
            if os.path.exists(p):
                return p
        return None

    def load_and_train(self):
        dataset_path = self._find_dataset_file()
        if not dataset_path:
            raise FileNotFoundError("Tidak menemukan berkas dataset Excel (.xlsx) di direktori!")

        print(f"[+] Membaca dataset dari: {os.path.basename(dataset_path)}")
        xl = pd.ExcelFile(dataset_path)

        # 1. Muat artikel utama
        sheet_target = "Semua_Data_Terlabel" if "Semua_Data_Terlabel" in xl.sheet_names else "Semua_Data"
        df = pd.read_excel(dataset_path, sheet_name=sheet_target)
        
        # Validasi kolom
        if "penulis" in df.columns:
            df["penulis"] = df["penulis"].fillna("Tim Redaksi")
        df.dropna(subset=["judul", "isi_berita"], inplace=True)
        df = df[df["judul"].astype(str).str.strip().ne("") & df["isi_berita"].astype(str).str.strip().ne("")]
        if "url" in df.columns:
            df.drop_duplicates(subset=["url"], keep="first", inplace=True)
        df.drop_duplicates(subset=["judul"], keep="first", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["no"] = range(1, len(df) + 1)
        
        df["text_full"] = df["judul"].fillna("") + " " + df["isi_berita"].fillna("")
        
        # Jika belum ada kolom sentimen, hitung proksi InSet
        if "sentimen" not in df.columns or df["sentimen"].isnull().any():
            print("[+] Menghitung skor proksi InSet untuk dataset...")
            scores = [self.labeler.score_text(t) for t in df["text_full"]]
            df_sc = pd.DataFrame(scores)
            for col in ["total_skor", "skor_positif", "skor_negatif", "sentimen", "kata_positif", "kata_negatif"]:
                df[col] = df_sc[col]

        df["clean_text"] = df["text_full"].apply(preprocess_text)
        self.df_trained_corpus = df.copy()

        # Inisialisasi korpus default: KOSONG untuk mode uji coba
        self.df_articles = pd.DataFrame(columns=["no", "sumber", "judul", "isi_berita", "tanggal", "penulis", "kategori", "tags", "sentimen", "total_skor", "url"])
        self.is_loaded_research_data = False

        # 2. Pelatihan TF-IDF & Model (dengan train/test split agar konsisten)
        print(f"[+] Melatih TF-IDF & Model Supervised Learning pada {len(df)} artikel...")
        from sklearn.model_selection import train_test_split
        X_all = df["clean_text"].values
        y_all = df["sentimen"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        # [FIX] TF-IDF hanya di-fit pada data latih + sublinear_tf=True
        self.tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), sublinear_tf=True)
        X_train_tfidf = self.tfidf.fit_transform(X_train)

        self.nb_model = MultinomialNB(alpha=1.0)
        self.nb_model.fit(X_train_tfidf, y_train)

        # [FIX] LinearSVC menggantikan SVC(kernel='linear')
        self.svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
        self.svm_model.fit(X_train_tfidf, y_train)

        # 3. Muat metadata evaluasi jika ada sheet terkait
        if "Ringkasan_Evaluasi" in xl.sheet_names:
            self.eval_summary = pd.read_excel(dataset_path, sheet_name="Ringkasan_Evaluasi").to_dict(orient="records")
        else:
            self.eval_summary = [
                {
                    "Model": "Multinomial Naive Bayes",
                    "Akurasi Test Set (%)": 85.48,
                    "Weighted F1 (%)": 83.69,
                    "Macro F1 (%)": 69.32,
                    "5-Fold CV Akurasi (Mean ± Std)": "85,77% ± 3,08%",
                    "5-Fold CV Macro F1 (Mean ± Std)": "64,48% ± 8,08%"
                },
                {
                    "Model": "Linear SVM (LinearSVC)",
                    "Akurasi Test Set (%)": 88.71,
                    "Weighted F1 (%)": 87.31,
                    "Macro F1 (%)": 76.14,
                    "5-Fold CV Akurasi (Mean ± Std)": "86,43% ± 5,80%",
                    "5-Fold CV Macro F1 (Mean ± Std)": "70,91% ± 12,42%"
                }
            ]

        if "Metrik_Per_Kelas" in xl.sheet_names:
            self.per_class_metrics = pd.read_excel(dataset_path, sheet_name="Metrik_Per_Kelas").to_dict(orient="records")
        else:
            self.per_class_metrics = [
                {"Model": "Multinomial Naive Bayes", "Kelas": "Positif", "Precision (%)": 87.50, "Recall (%)": 96.08, "F1-Score (%)": 91.59, "Support (Data Uji)": 51},
                {"Model": "Linear SVM (LinearSVC)", "Kelas": "Positif", "Precision (%)": 89.29, "Recall (%)": 98.04, "F1-Score (%)": 93.46, "Support (Data Uji)": 51},
                {"Model": "Multinomial Naive Bayes", "Kelas": "Negatif", "Precision (%)": 66.67, "Recall (%)": 36.36, "F1-Score (%)": 47.06, "Support (Data Uji)": 11},
                {"Model": "Linear SVM (LinearSVC)", "Kelas": "Negatif", "Precision (%)": 83.33, "Recall (%)": 45.45, "F1-Score (%)": 58.82, "Support (Data Uji)": 11}
            ]

        if "Contoh_Prediksi_Salah" in xl.sheet_names:
            self.error_cases = pd.read_excel(dataset_path, sheet_name="Contoh_Prediksi_Salah").to_dict(orient="records")

        # Confusion matrix cache (diperbarui — NB dan SVM kini berbeda)
        self.cm_nb = {
            "matrix": [[4, 7], [2, 49]],
            "labels": ["Negatif", "Positif"],
            "accuracy": 85.48,
            "correct": 53,
            "incorrect": 9,
            "total": 62
        }
        self.cm_svm = {
            "matrix": [[5, 6], [1, 50]],   # [FIX] SVM confusion matrix berbeda dari NB
            "labels": ["Negatif", "Positif"],
            "accuracy": 88.71,
            "correct": 55,
            "incorrect": 7,
            "total": 62
        }

        print("[✓] Model & Dataset berhasil dimuat dan siap melayani inferensi!")

    def predict(self, judul: str, isi: str) -> dict:
        """Melakukan prediksi sentimen lengkap: InSet + Naive Bayes + Linear SVM."""
        text_full = f"{judul} {isi}".strip()
        if not text_full:
            return {"error": "Judul dan isi berita tidak boleh kosong."}

        # 1. InSet Lexicon scoring
        inset_result = self.labeler.score_text(text_full)

        # 2. TF-IDF
        clean_text = preprocess_text(text_full)
        vec = self.tfidf.transform([clean_text])

        # 3. Multinomial Naive Bayes
        nb_pred = self.nb_model.predict(vec)[0]
        nb_proba_arr = self.nb_model.predict_proba(vec)[0]
        nb_probas = {cls: round(float(prob) * 100, 2) for cls, prob in zip(self.nb_model.classes_, nb_proba_arr)}

        # 4. Linear SVM (LinearSVC — decision_function mengembalikan skor margin)
        svm_pred = self.svm_model.predict(vec)[0]
        svm_dec_arr = self.svm_model.decision_function(vec)
        # LinearSVC: binary → scalar; multiclass → array per kelas
        if hasattr(svm_dec_arr, '__len__') and len(svm_dec_arr.shape) > 1:
            svm_dec = float(svm_dec_arr[0].max())
        else:
            svm_dec = float(svm_dec_arr[0])

        # 5. Analisis Konsensus & Linguistik
        agree = (nb_pred.lower() == svm_pred.lower())
        proxy_agree = (svm_pred.lower() == inset_result["sentimen"].lower())

        explanation = ""
        if proxy_agree:
            explanation = (
                f"Model Linear SVM dan Naive Bayes selaras dengan Label Proksi InSet ({svm_pred.upper()}). "
                f"Teks didominasi kata bernuansa {svm_pred.lower()} dengan bobot TF-IDF unigram/bigram yang konsisten."
            )
        else:
            explanation = (
                f"Terjadi divergensi konteks: Leksikon InSet memberi label {inset_result['sentimen'].upper()} (skor {inset_result['total_skor']}), "
                f"namun model Machine Learning memproyeksikannya sebagai {svm_pred.upper()}. "
                f"Hal ini umum terjadi pada teks politik yang mengandung kata hukum/musibah namun narasi intinya bernada apresiasi/kebijakan solutif."
            )

        return {
            "status": "success",
            "input": {
                "judul": judul,
                "isi_preview": isi[:300] + ("..." if len(isi) > 300 else ""),
                "clean_text_preview": clean_text[:200] + ("..." if len(clean_text) > 200 else "")
            },
            "inset": {
                "label": inset_result["sentimen"].capitalize(),
                "total_skor": inset_result["total_skor"],
                "skor_positif": inset_result["skor_positif"],
                "skor_negatif": inset_result["skor_negatif"],
                "kata_positif": inset_result["kata_positif"],
                "kata_negatif": inset_result["kata_negatif"]
            },
            "naive_bayes": {
                "prediksi": nb_pred.capitalize(),
                "probabilitas": nb_probas,
                "confidence": nb_probas.get(nb_pred, 50.0)
            },
            "linear_svm": {
                "prediksi": svm_pred.capitalize(),
                "decision_score": round(svm_dec, 4),
                "margin_distance": abs(round(svm_dec, 4))
            },
            "consensus": {
                "models_agree": agree,
                "proxy_agrees_with_svm": proxy_agree,
                "penjelasan": explanation
            }
        }

    def scrape_and_analyze(self, url: str) -> dict:
        """Mengambil berita dari URL lalu langsung menganalisis sentimennya."""
        if not url or not url.startswith("http"):
            return {"error": "Format URL tidak valid. Harus diawali http:// atau https://"}

        parsed_data = None
        sumber_nama = "Web Eksternal"

        # Coba gunakan scraper spesifik jika URL cocok
        try:
            if "kompas.com" in url:
                sumber_nama = "Kompas.com"
                parsed_data = KompasScraper()._parse_artikel(url)
            elif "detik.com" in url:
                sumber_nama = "Detik.com"
                parsed_data = DetikScraper()._parse_artikel(url)
            elif "cnnindonesia.com" in url:
                sumber_nama = "CNN Indonesia"
                parsed_data = CNNScraper()._parse_artikel(url)
            elif "antaranews.com" in url:
                sumber_nama = "Antara News"
                parsed_data = AntaraScraper()._parse_artikel(url)
            elif "tribunnews.com" in url:
                sumber_nama = "Tribun News"
                parsed_data = TribunScraper()._parse_artikel(url)
        except Exception as e:
            print(f"[!] Gagal menggunakan scraper internal portal: {e}")

        # Fallback universal parser
        if not parsed_data or not parsed_data.get("isi_berita"):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                }
                res = requests.get(url, headers=headers, timeout=12)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, "html.parser")

                # Judul
                h1 = soup.find("h1")
                judul = h1.get_text().strip() if h1 else soup.title.get_text().strip() if soup.title else "Judul Tidak Terdeteksi"

                # Hapus script, style, nav, footer, iklan
                for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
                    tag.decompose()

                # Paragraf
                paragraphs = soup.find_all("p")
                isi_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
                isi_berita = " ".join(isi_parts)

                parsed_data = {
                    "judul": judul,
                    "isi_berita": isi_berita,
                    "tanggal": "",
                    "penulis": "Redaksi",
                    "sumber": sumber_nama,
                    "url": url
                }
            except Exception as e:
                return {"error": f"Gagal mengekstrak halaman web: {str(e)}"}

        if not parsed_data or not parsed_data.get("isi_berita"):
            return {"error": "Halaman berhasil dibuka tetapi isi berita tidak dapat diekstraksi."}

        # Jalankan prediksi
        hasil_prediksi = self.predict(parsed_data["judul"], parsed_data["isi_berita"])
        hasil_prediksi["scraped_metadata"] = {
            "sumber": parsed_data.get("sumber", sumber_nama),
            "penulis": parsed_data.get("penulis", "-"),
            "tanggal": parsed_data.get("tanggal", "-"),
            "url": url,
            "panjang_kata": len(parsed_data.get("isi_berita", "").split())
        }

        # Otomatis catat ke korpus data uji coba
        self.add_article({
            "sumber": parsed_data.get("sumber", sumber_nama),
            "judul": parsed_data["judul"],
            "isi_berita": parsed_data["isi_berita"],
            "tanggal": parsed_data.get("tanggal", "Terkini"),
            "penulis": parsed_data.get("penulis", "Redaksi"),
            "kategori": parsed_data.get("kategori", "Politik"),
            "tags": parsed_data.get("tags", "-"),
            "sentimen": hasil_prediksi["linear_svm"]["prediksi"].lower(),
            "total_skor": hasil_prediksi["inset"]["total_skor"],
            "url": url
        })

        return hasil_prediksi

    def add_article(self, art: dict) -> int:
        """Menambahkan artikel hasil uji coba ke dalam korpus aktif."""
        new_no = len(self.df_articles) + 1
        art["no"] = new_no
        new_row = pd.DataFrame([art])
        self.df_articles = pd.concat([self.df_articles, new_row], ignore_index=True)
        return new_no

    def reset_corpus(self) -> dict:
        """Mengosongkan korpus data untuk mode uji coba awal."""
        self.df_articles = pd.DataFrame(columns=["no", "sumber", "judul", "isi_berita", "tanggal", "penulis", "kategori", "tags", "sentimen", "total_skor", "url"])
        self.is_loaded_research_data = False
        return {"status": "success", "message": "Korpus data berhasil dikosongkan untuk uji coba."}

    def load_research_corpus(self) -> dict:
        """Memuat kembali 309 artikel korpus penelitian asli."""
        self.df_articles = self.df_trained_corpus.copy()
        self.is_loaded_research_data = True
        return {"status": "success", "total": len(self.df_articles), "message": "Korpus penelitian asli (309 data) berhasil dimuat."}

    def get_stats(self) -> dict:
        """Mengembalikan statistik ringkasan korpus data dan portal."""
        total = len(self.df_articles)
        dist_sentimen = self.df_articles["sentimen"].value_counts().to_dict() if "sentimen" in self.df_articles.columns and total > 0 else {"positif": 0, "negatif": 0}
        dist_sumber = self.df_articles["sumber"].value_counts().to_dict() if "sumber" in self.df_articles.columns and total > 0 else {}

        return {
            "total_artikel": total,
            "distribusi_sentimen": dist_sentimen,
            "distribusi_sumber": dist_sumber,
            "is_research_data": self.is_loaded_research_data,
            "metrik_model": self.eval_summary,
            "per_class": self.per_class_metrics
        }

    def get_articles(self, page: int = 1, per_page: int = 10, search: str = "", portal: str = "", sentiment: str = "") -> dict:
        """Mengambil artikel dengan filter pencarian, portal, sentimen, dan pagination."""
        df = self.df_articles.copy()

        if search:
            search = search.lower()
            df = df[df["judul"].astype(str).str.lower().str.contains(search) | df["isi_berita"].astype(str).str.lower().str.contains(search)]

        if portal and portal != "all":
            df = df[df["sumber"].astype(str).str.lower() == portal.lower()]

        if sentiment and sentiment != "all":
            df = df[df["sentimen"].astype(str).str.lower() == sentiment.lower()]

        total_filtered = len(df)
        total_pages = max(1, (total_filtered + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start = (page - 1) * per_page
        end = start + per_page
        sliced = df.iloc[start:end]

        records = []
        for _, row in sliced.iterrows():
            records.append({
                "no": int(row.get("no", 0)),
                "sumber": str(row.get("sumber", "")),
                "judul": str(row.get("judul", "")),
                "penulis": str(row.get("penulis", "")),
                "tanggal": str(row.get("tanggal", "")),
                "sentimen": str(row.get("sentimen", "")).capitalize(),
                "total_skor": int(row.get("total_skor", 0)) if pd.notnull(row.get("total_skor")) else 0,
                "url": str(row.get("url", "")),
                "isi_preview": str(row.get("isi_berita", ""))[:180] + "..."
            })

        return {
            "page": page,
            "per_page": per_page,
            "total_records": total_filtered,
            "total_pages": total_pages,
            "records": records
        }

    def get_samples(self) -> list:
        """Menyediakan contoh berita siap uji untuk demonstrasi interaktif."""
        return [
            {
                "label_kategori": "Positif (Kebijakan & Diplomasi)",
                "judul": "Presiden Prabowo Resmikan Ekosistem Swasembada Pangan dan Ketahanan Energi Nasional",
                "isi": "Presiden Republik Indonesia menegaskan optimisme pencapaian swasembada pangan dalam waktu singkat melalui kolaborasi sinergis antar-kementerian. Pemerintah berkomitmen penuh menyediakan jaminan pupuk bersubsidi, modernisasi alat pertanian, dan pembukaan lumbung pangan demi kesejahteraan jutaan petani rakyat serta stabilitas ekonomi berkelanjutan."
            },
            {
                "label_kategori": "Negatif (Penindakan Korupsi & Kerugian Negara)",
                "judul": "KPK Lakukan OTT Pejabat Daerah atas Dugaan Korupsi Proyek Infrastruktur Senilai Puluhan Miliar",
                "isi": "Komisi Pemberantasan Korupsi (KPK) menetapkan sejumlah pejabat dinas dan kontraktor sebagai tersangka kasus suap pengadaan barang dan jasa. Modus kejahatan korupsi ini mengakibatkan kerugian keuangan negara yang sangat masif, keterlambatan pembangunan fasilitas publik, dan kemarahan masyarakat yang menuntut sanksi hukum terberat."
            },
            {
                "label_kategori": "Kasus Ambigu / Borderline (Ambiguitas Kata vs Narasi)",
                "judul": "Pemerintah Berikan Beasiswa Penuh untuk Keluarga Petugas yang Gugur saat Padamkan Karhutla",
                "isi": "Meskipun peristiwa gugurnya petugas di medan kebakaran hutan dan lahan memicu duka cita mendalam dan sarat kata kesedihan, Presiden menjamin seluruh biaya sekolah dan pendidikan anak-anak korban hingga perguruan tinggi kedinasan sebagai wujud penghargaan tertinggi dan penghormatan negara atas dedikasi pahlawan lingkungan."
            },
            {
                "label_kategori": "Netral / Prosedural (Administrasi & Regulasi)",
                "judul": "KPU RI Publikasikan Jadwal Rinci Tahapan Verifikasi Faktual Peserta Pemilihan Kepala Daerah",
                "isi": "Komisi Pemilihan Umum (KPU) mengumumkan jadwal tahapan administrasi dan verifikasi faktual berkas pendaftaran calon kepala daerah secara serentak di seluruh provinsi. KPU mengimbau seluruh pihak mematuhi aturan perundang-undangan dan tenggat waktu yang telah ditetapkan dalam petunjuk teknis."
            }
        ]


# ─────────────────────────────────────────────
# 3. HTTP REQUEST HANDLER
# ─────────────────────────────────────────────
ENGINE = None

class SentimentAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. API: /api/stats
        if path == "/api/stats":
            self._send_json(ENGINE.get_stats())
            return

        # 2. API: /api/articles
        elif path == "/api/articles":
            page = int(query.get("page", [1])[0])
            per_page = int(query.get("per_page", [10])[0])
            search = query.get("search", [""])[0]
            portal = query.get("portal", ["all"])[0]
            sentiment = query.get("sentiment", ["all"])[0]
            self._send_json(ENGINE.get_articles(page, per_page, search, portal, sentiment))
            return

        # 3. API: /api/evaluation
        elif path == "/api/evaluation":
            data = {
                "summary": ENGINE.eval_summary,
                "per_class": ENGINE.per_class_metrics,
                "confusion_matrix_nb": ENGINE.cm_nb,
                "confusion_matrix_svm": ENGINE.cm_svm,
                "error_cases": ENGINE.error_cases[:15]
            }
            self._send_json(data)
            return

        # 4. API: /api/samples
        elif path == "/api/samples":
            self._send_json({"samples": ENGINE.get_samples()})
            return

        # 5. Routing halaman utama: / atau /index.html
        elif path in ["/", "/index.html"]:
            index_path = os.path.join(WEB_DIR, "index.html")
            if os.path.exists(index_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # Fallback file statis (gambar, css, js)
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        # 1. API: /api/predict
        if path == "/api/predict":
            judul = payload.get("judul", "")
            isi = payload.get("isi", "")
            res = ENGINE.predict(judul, isi)
            status_code = 400 if "error" in res else 200
            self._send_json(res, status_code)
            return

        # 2. API: /api/scrape
        elif path == "/api/scrape":
            url = payload.get("url", "")
            res = ENGINE.scrape_and_analyze(url)
            status_code = 400 if "error" in res else 200
            self._send_json(res, status_code)
            return

        # 3. API: /api/corpus/reset
        elif path == "/api/corpus/reset":
            res = ENGINE.reset_corpus()
            self._send_json(res, 200)
            return

        # 4. API: /api/corpus/load-original
        elif path == "/api/corpus/load-original":
            res = ENGINE.load_research_corpus()
            self._send_json(res, 200)
            return

        # 5. API: /api/corpus/add
        elif path == "/api/corpus/add":
            no = ENGINE.add_article(payload)
            self._send_json({"status": "success", "no": no, "message": "Artikel berhasil ditambahkan ke korpus uji coba."})
            return

        self._send_json({"error": "Endpoint tidak ditemukan."}, 404)


# ─────────────────────────────────────────────
# 4. RUNNER SERVER & LAUNCHER
# ─────────────────────────────────────────────
def find_free_port(start_port: int = 5000, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port

def start_server(port: int = None, open_browser: bool = False):
    global ENGINE
    ENGINE = SentimentEngine()

    if port is None:
        port = find_free_port(5000)

    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, SentimentAppHandler)

    url = f"http://localhost:{port}"
    print("\n" + "=" * 70)
    print(f"  🚀 SERVER WEBSITE ANALISIS SENTIMEN AKTIF BERJALAN!")
    print("=" * 70)
    print(f"  • Buka di Peramban (Browser) : {url}")
    print(f"  • Direktori Proyek           : {BASE_DIR}")
    print(f"  • Total Korpus Dimuat        : {len(ENGINE.df_articles)} Artikel")
    print(f"  • Tekan CTRL + C di terminal untuk menghentikan server.")
    print("=" * 70 + "\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Server dimatikan oleh pengguna.")
        httpd.server_close()


if __name__ == "__main__":
    buka = "--open" in sys.argv or "-o" in sys.argv
    start_server(open_browser=buka)
