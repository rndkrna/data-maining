import os
import re
from typing import Dict, Tuple, List

LEXICON_DIR = os.path.join(os.path.dirname(__file__), "lexicon")
FILE_POS = os.path.join(LEXICON_DIR, "positive.tsv")
FILE_NEG = os.path.join(LEXICON_DIR, "negative.tsv")

class InSetLexiconLabeler:
    """
    Pelabelan Sentimen Berbasis InSet Lexicon (Indonesian Sentiment Lexicon)
    Referensi Ilmiah:
    Koto, F., & Rahmaningtyas, G. Y. (2017).
    InSet Lexicon: Evaluation of a word list for Indonesian sentiment analysis in microblogs.
    2017 International Conference on Asian Language Processing (IALP), 391-394.
    """
    def __init__(self, pos_file: str = FILE_POS, neg_file: str = FILE_NEG):
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.pos_words: Dict[str, int] = {}
        self.pos_phrases: Dict[str, int] = {}
        self.neg_words: Dict[str, int] = {}
        self.neg_phrases: Dict[str, int] = {}
        self._load_lexicon()

    def _load_lexicon(self):
        if not os.path.exists(self.pos_file) or not os.path.exists(self.neg_file):
            raise FileNotFoundError(
                f"File leksikon InSet belum lengkap. Pastikan {self.pos_file} dan {self.neg_file} tersedia."
            )

        # Kata tugas/tata bahasa netral dan nama media berita yang tidak boleh dihitung sebagai polaritas
        self.ignore_terms = set([
            "yang", "di", "dan", "dengan", "untuk", "dari", "ke", "ini", "itu", "pada",
            "adalah", "bahwa", "dalam", "akan", "juga", "atau", "oleh", "saat", "tersebut",
            "bisa", "tidak", "ada", "karena", "ia", "mereka", "kita", "kami", "telah",
            "seperti", "sudah", "hanya", "sebagai", "para", "lebih", "dapat", "antara",
            "setelah", "namun", "tetapi", "serta", "ketika", "bagi", "lalu", "masih",
            "belum", "secara", "lagi", "sehingga", "maupun", "bahkan", "tanpa", "hampir",
            "terhadap", "maka", "tentang", "selama", "kemudian", "kepada", "harus",
            "banyak", "hal", "pun", "saja", "menjadi", "diri", "tiap", "bila", "jika",
            "kompas", "detik", "cnn", "antara", "tribun", "news", "indonesia"
        ])

        # Load Positive
        with open(self.pos_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2 and parts[0] != "word":
                    term = parts[0].strip().lower()
                    if term in self.ignore_terms:
                        continue
                    try:
                        weight = int(parts[1].strip())
                    except ValueError:
                        continue
                    if " " in term:
                        self.pos_phrases[term] = weight
                    else:
                        self.pos_words[term] = weight

        # Load Negative
        with open(self.neg_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2 and parts[0] != "word":
                    term = parts[0].strip().lower()
                    if term in self.ignore_terms:
                        continue
                    try:
                        weight = abs(int(parts[1].strip()))
                    except ValueError:
                        continue
                    if " " in term:
                        self.neg_phrases[term] = weight
                    else:
                        self.neg_words[term] = weight

        print(f"[InSet] Leksikon berhasil dimuat:")
        print(f"  - Kata/Frasa Positif : {len(self.pos_words) + len(self.pos_phrases)} entri")
        print(f"  - Kata/Frasa Negatif : {len(self.neg_words) + len(self.neg_phrases)} entri")

    def score_text(self, text: str) -> Dict:
        """
        Menghitung skor sentimen berdasarkan InSet Lexicon.
        Mengembalikan skor positif, skor negatif, total polaritas, label, dan rincian kata.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "skor_positif": 0,
                "skor_negatif": 0,
                "total_skor": 0,
                "sentimen": "netral",
                "kata_positif": "",
                "kata_negatif": ""
            }

        text_lower = text.lower()

        matched_pos: List[str] = []
        matched_neg: List[str] = []
        pos_score = 0
        neg_score = 0

        # 1. Cek frasa (multi-word) terlebih dahulu
        for phrase, weight in self.pos_phrases.items():
            if phrase in text_lower:
                pos_score += weight
                matched_pos.append(f"{phrase}(+{weight})")

        for phrase, weight in self.neg_phrases.items():
            if phrase in text_lower:
                neg_score += weight
                matched_neg.append(f"{phrase}(-{weight})")

        # 2. Tokenisasi kata tunggal
        clean_words = re.findall(r"\b[a-z]{3,}\b", text_lower)
        for w in clean_words:
            if w in self.pos_words:
                weight = self.pos_words[w]
                pos_score += weight
                matched_pos.append(f"{w}(+{weight})")
            elif w in self.neg_words:
                weight = self.neg_words[w]
                neg_score += weight
                matched_neg.append(f"{w}(-{weight})")

        total_skor = pos_score - neg_score

        # Penentuan Label Objektif
        if total_skor > 0:
            label = "positif"
        elif total_skor < 0:
            label = "negatif"
        else:
            label = "netral"

        return {
            "skor_positif": pos_score,
            "skor_negatif": neg_score,
            "total_skor": total_skor,
            "sentimen": label,
            "kata_positif": ", ".join(matched_pos[:10]),  # Ambil hingga 10 kata teratas
            "kata_negatif": ", ".join(matched_neg[:10])
        }
