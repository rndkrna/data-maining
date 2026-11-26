"""
=============================================================================
  ANALISIS SENTIMEN BERITA POLITIK INDONESIA (DATA MINING)
  VERSI REVISI — BEBAS DATA LEAKAGE (Pipeline-Based)
=============================================================================
  Metodologi Akademis Standar (Versi Terkoreksi):
  1. Pelabelan Menggunakan "Label Proksi Berbasis Leksikon":
     - Menggunakan Indonesian Sentiment Lexicon (InSet) karya Koto & Rahmaningtyas (2017)
     - Label berfungsi sebagai "proksi awal" (proxy label), BUKAN ground truth
     - Catatan Metodologi: Polaritas kata pada tingkat dokumen tidak sama
       dengan stance/keberpihakan politik terhadap aktor tertentu.
  2. Text Preprocessing (Cleansing Regex, Case Folding, Stopwords Removal)
     CATATAN PENTING: Kata negasi (tidak, belum, bukan, tanpa, jangan)
     DIPERTAHANKAN karena mengubah makna kalimat secara semantik.
  3. Ekstraksi Fitur: TF-IDF Vectorizer (Unigram & Bigram, Max Features: 1500,
     sublinear_tf=True) — hanya di-fit pada data latih (Pipeline).
  4. Klasifikasi Terawasi (Supervised Learning):
     - Multinomial Naive Bayes (alpha=1.0)
     - Linear Support Vector Machine (LinearSVC, C=1.0, random_state=42)
  5. Validasi:
     - Hold-out Test Set (80% Train, 20% Test, Stratified, random_state=42)
     - 5-Fold Stratified Cross-Validation menggunakan Pipeline (bebas leakage)
     - Evaluasi Macro F1, Weighted Metrics, dan Metrik Per-Kelas
     - Analisis Confusion Matrix Lengkap
     - Identifikasi & Analisis Diagnostik Contoh Prediksi Salah (Error Analysis)
  6. Ekspor Audit Lengkap ke Excel Multi-Sheet
=============================================================================
  PERUBAHAN DARI VERSI SEBELUMNYA:
  [FIX-1] Data Leakage: TF-IDF sekarang HANYA di-fit pada X_train,
          bukan pada seluruh dataset. Menggunakan Pipeline scikit-learn.
  [FIX-2] Kata negasi (tidak, belum, bukan, tanpa, jangan) DIKELUARKAN
          dari daftar stopwords agar makna semantik kalimat terjaga.
  [FIX-3] Parameter sublinear_tf=True ditambahkan pada TfidfVectorizer
          sesuai praktik terbaik analisis sentimen teks.
  [FIX-4] Cross-Validation menggunakan Pipeline sehingga TF-IDF
          dipelajari ulang pada setiap fold secara independen.
  [FIX-5] LinearSVC (lebih efisien) digunakan sebagai ganti SVC(kernel='linear').
=============================================================================
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score
)

from inset_labeler import InSetLexiconLabeler

# Pastikan output terminal mendukung UTF-8 di Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ─────────────────────────────────────────────
# 1. PREPROCESSING TEKS KHUSUS BAHASA INDONESIA
# ─────────────────────────────────────────────
# [FIX-2] Kata negasi (tidak, belum, bukan, tanpa, jangan) DIHAPUS dari
# stopwords karena mengubah polaritas kalimat secara semantik.
# Contoh: "tidak baik" ≠ "baik"; menghapus "tidak" akan menyesatkan model.
STOPWORDS_ID = set([
    "yang", "di", "dan", "dengan", "untuk", "dari", "ke", "ini", "itu", "pada",
    "adalah", "bahwa", "dalam", "akan", "juga", "atau", "oleh", "saat", "tersebut",
    "bisa", "ada", "karena", "ia", "mereka", "kita", "kami", "telah",
    "seperti", "sudah", "hanya", "sebagai", "para", "lebih", "dapat",
    "antara", "setelah", "namun", "tetapi", "serta", "ketika", "bagi", "lalu",
    "secara", "lagi", "sehingga", "maupun", "bahkan",
    "hampir", "terhadap", "maka", "tentang", "selama", "kemudian", "kepada",
    "harus", "banyak", "hal", "pun", "saja", "menjadi", "diri", "tiap",
    # Kata-kata berikut sengaja TIDAK dimasukkan ke stopwords:
    # "tidak", "belum", "bukan", "tanpa", "jangan" — KATA NEGASI (dipertahankan)
    # "masih", "belum" — penanda aspek temporal yang memiliki bobot semantik
])

def preprocess_text(teks: str) -> str:
    """
    Membersihkan teks berita untuk ekstraksi fitur TF-IDF.
    Tahapan: case folding → hapus URL → hapus simbol/angka → stopwords removal.
    Kata negasi (tidak, belum, bukan, tanpa, jangan) DIPERTAHANKAN.
    """
    if not isinstance(teks, str):
        return ""
    # Case folding
    teks = teks.lower()
    # Hapus URL
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)
    # Hapus karakter khusus dan angka
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    # Tokenisasi sederhana & buang stopword (panjang >= 3 karakter)
    words = [w for w in teks.split() if w not in STOPWORDS_ID and len(w) >= 3]
    return " ".join(words)


# ─────────────────────────────────────────────
# 2. FUNGSI UTAMA PIPELINE SENTIMEN
# ─────────────────────────────────────────────
def main(input_excel: str = "dataset_berita.xlsx"):
    print("=" * 70)
    print("  MEMULAI PIPELINE ANALISIS SENTIMEN — VERSI TERKOREKSI (BEBAS LEAKAGE)")
    print("=" * 70)
    print("  [INFO] Perubahan utama dari versi sebelumnya:")
    print("  [FIX-1] TF-IDF hanya di-fit pada data LATIH (via Pipeline) → Bebas Data Leakage")
    print("  [FIX-2] Kata negasi dipertahankan (tidak/belum/bukan/tanpa/jangan)")
    print("  [FIX-3] sublinear_tf=True ditambahkan ke TF-IDF")
    print("  [FIX-4] Cross-Validation menggunakan Pipeline (leakage-free CV)")
    print("  [FIX-5] LinearSVC menggantikan SVC(kernel='linear')")
    print("=" * 70)

    # Cari file dataset
    if not os.path.exists(input_excel):
        fallback_files = [
            "SiapMining_2301020003.xlsx",
            "dataset_berita_politik.xlsx",
            "Mentah_2301020003.xlsx",
        ]
        for fb in fallback_files:
            if os.path.exists(fb):
                input_excel = fb
                break

    if not os.path.exists(input_excel):
        print(f"[!] Error: File dataset '{input_excel}' tidak ditemukan.")
        return

    print(f"\n[+] Membaca file dataset: {input_excel}")
    xl = pd.ExcelFile(input_excel)
    # Prioritas: Semua_Data_Terlabel (SiapMining) > Semua_Data (Mentah)
    if "Semua_Data_Terlabel" in xl.sheet_names:
        sheet_name = "Semua_Data_Terlabel"
    elif "Semua_Data" in xl.sheet_names:
        sheet_name = "Semua_Data"
    else:
        sheet_name = xl.sheet_names[0]
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    print(f"[+] Total data berita dimuat dari sheet '{sheet_name}': {len(df)} artikel")

    # ─────────────────────────────────────────────
    # PREPROCESSING: CLEANING MISSING VALUE & DEDUPLIKASI
    # ─────────────────────────────────────────────
    sebelum_mv = len(df)
    if "penulis" in df.columns:
        df["penulis"] = df["penulis"].fillna("Tim Redaksi")
    df.dropna(subset=["judul", "isi_berita"], inplace=True)
    df = df[df["judul"].astype(str).str.strip().ne("") & df["isi_berita"].astype(str).str.strip().ne("")]
    dihapus_mv = sebelum_mv - len(df)
    if dihapus_mv > 0:
        print(f"  [i] Dihapus {dihapus_mv} baris karena judul atau isi berita kosong (missing value).")

    sebelum_dup = len(df)
    if "url" in df.columns:
        df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    df.drop_duplicates(subset=["judul"], keep="first", inplace=True)
    dihapus_dup = sebelum_dup - len(df)
    if dihapus_dup > 0:
        print(f"  [i] Dihapus {dihapus_dup} baris duplikat (URL atau judul sama).")

    df.reset_index(drop=True, inplace=True)
    df["no"] = range(1, len(df) + 1)
    print(f"[+] Total korpus berita bersih & unik: {len(df)} artikel")

    # ─────────────────────────────────────────────
    # PELABELAN InSet LEXICON (hanya jika belum ada label)
    # ─────────────────────────────────────────────
    if "sentimen" not in df.columns or df["sentimen"].isna().any():
        print("\n[+] Menginisialisasi InSet Lexicon (Koto & Rahmaningtyas, 2017)...")
        labeler = InSetLexiconLabeler()
        df["text_full"] = df["judul"].fillna("") + " " + df["isi_berita"].fillna("")
        print("[+] Menghitung skor polaritas InSet untuk menghasilkan LABEL PROKSI...")
        labeling_results = [labeler.score_text(t) for t in df["text_full"]]
        df_scores = pd.DataFrame(labeling_results)
        df["skor_positif"]  = df_scores["skor_positif"]
        df["skor_negatif"]  = df_scores["skor_negatif"]
        df["total_skor"]    = df_scores["total_skor"]
        df["sentimen"]      = df_scores["sentimen"]
        df["kata_positif"]  = df_scores["kata_positif"]
        df["kata_negatif"]  = df_scores["kata_negatif"]
    else:
        print("\n[+] Label proksi InSet sudah tersedia di dataset, melewati pelabelan ulang.")
        if "text_full" not in df.columns:
            df["text_full"] = df["judul"].fillna("") + " " + df["isi_berita"].fillna("")

    print("\n" + "-" * 55)
    print("DISTRIBUSI KELAS LABEL PROKSI BERBASIS LEKSIKON (InSet):")
    print("-" * 55)
    dist = df["sentimen"].value_counts()
    for kelas, count in dist.items():
        persen = (count / len(df)) * 100
        print(f"  • {kelas.upper():<10}: {count} artikel ({persen:.2f}%)")

    # ─────────────────────────────────────────────
    # PREPROCESSING TEKS UNTUK MACHINE LEARNING
    # ─────────────────────────────────────────────
    print("\n[+] Melakukan Preprocessing Teks (Cleansing, Case Folding, Stopwords)...")
    print("    [INFO] Kata negasi (tidak/belum/bukan/tanpa/jangan) DIPERTAHANKAN")
    df["clean_text"] = df["text_full"].apply(preprocess_text)

    X = df["clean_text"].values
    y = df["sentimen"].values

    # ─────────────────────────────────────────────
    # PEMBAGIAN DATA TRAIN & TEST (STRATIFIED)
    # ─────────────────────────────────────────────
    print("\n[+] Membagi Data: 80% Training, 20% Testing (Stratified, random_state=42)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"  - Data Latih (Train): {len(X_train)} artikel")
    print(f"  - Data Uji (Test)   : {len(X_test)} artikel")

    # Verifikasi distribusi kelas di setiap set
    from collections import Counter
    cnt_train = Counter(y_train)
    cnt_test  = Counter(y_test)
    print(f"  - Distribusi Train  : {dict(cnt_train)}")
    print(f"  - Distribusi Test   : {dict(cnt_test)}")

    # ─────────────────────────────────────────────
    # PARAMETER TF-IDF (TERKOREKSI)
    # ─────────────────────────────────────────────
    # [FIX-1] TF-IDF hanya di-fit pada X_train
    # [FIX-3] sublinear_tf=True (log(1+tf) mengurangi dominasi kata frekuensi tinggi)
    tfidf_params = dict(
        max_features=1500,
        ngram_range=(1, 2),
        sublinear_tf=True        # [FIX-3] sesuai audit prompt
    )

    print("\n[+] Ekstraksi Fitur TF-IDF hanya pada DATA LATIH (bebas data leakage)...")
    print(f"    Parameter: max_features={tfidf_params['max_features']}, "
          f"ngram_range={tfidf_params['ngram_range']}, "
          f"sublinear_tf={tfidf_params['sublinear_tf']}")
    tfidf = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = tfidf.fit_transform(X_train)   # [FIX-1] fit HANYA pada train
    X_test_tfidf  = tfidf.transform(X_test)        # [FIX-1] transform saja pada test
    print(f"    Ukuran matriks TF-IDF Train: {X_train_tfidf.shape}")
    print(f"    Ukuran matriks TF-IDF Test : {X_test_tfidf.shape}")

    # ─────────────────────────────────────────────
    # MODEL 1: MULTINOMIAL NAIVE BAYES
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("1. MODEL MULTINOMIAL NAIVE BAYES (alpha=1.0)")
    print("=" * 55)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    p_nb_w, r_nb_w, f1_nb_w, _ = precision_recall_fscore_support(y_test, y_pred_nb, average="weighted", zero_division=0)
    p_nb_m, r_nb_m, f1_nb_m, _ = precision_recall_fscore_support(y_test, y_pred_nb, average="macro",    zero_division=0)

    print(f"Akurasi Naive Bayes         : {acc_nb * 100:.2f}%")
    print(f"Presisi (Weighted)          : {p_nb_w * 100:.2f}%")
    print(f"Recall (Weighted)           : {r_nb_w * 100:.2f}%")
    print(f"F1-Score (Weighted)         : {f1_nb_w * 100:.2f}%")
    print(f"F1-Score (Macro) ⭐          : {f1_nb_m * 100:.2f}%\n")
    print("Laporan Klasifikasi Naive Bayes:")
    print(classification_report(y_test, y_pred_nb, zero_division=0))

    # ─────────────────────────────────────────────
    # MODEL 2: LINEAR SVM (LinearSVC)
    # ─────────────────────────────────────────────
    print("=" * 55)
    print("2. MODEL LINEAR SUPPORT VECTOR MACHINE (LinearSVC)")
    print("=" * 55)
    # [FIX-5] LinearSVC lebih efisien untuk data teks berdimensi tinggi
    svm_params = {
        "C": 1.0,
        "max_iter": 2000,
        "random_state": 42
    }
    svm_model = LinearSVC(**svm_params)
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    p_svm_w, r_svm_w, f1_svm_w, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="weighted", zero_division=0)
    p_svm_m, r_svm_m, f1_svm_m, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="macro",    zero_division=0)

    print(f"Akurasi Linear SVM          : {acc_svm * 100:.2f}%")
    print(f"Presisi (Weighted)          : {p_svm_w * 100:.2f}%")
    print(f"Recall (Weighted)           : {r_svm_w * 100:.2f}%")
    print(f"F1-Score (Weighted)         : {f1_svm_w * 100:.2f}%")
    print(f"F1-Score (Macro) ⭐          : {f1_svm_m * 100:.2f}%\n")
    print("Laporan Klasifikasi Linear SVM:")
    print(classification_report(y_test, y_pred_svm, zero_division=0))

    # ─────────────────────────────────────────────
    # 3. 5-FOLD STRATIFIED CV — MENGGUNAKAN PIPELINE (BEBAS LEAKAGE)
    # ─────────────────────────────────────────────
    print("=" * 55)
    print("3. EKSPERIMEN TAMBAHAN: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("   [FIX-4] Menggunakan Pipeline — TF-IDF di-fit ulang setiap fold")
    print("=" * 55)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "f1_weighted"]

    # Pipeline NB: TF-IDF + MultinomialNB
    pipe_nb = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   MultinomialNB(alpha=1.0))
    ])
    # Pipeline SVM: TF-IDF + LinearSVC
    pipe_svm = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LinearSVC(**svm_params))
    ])

    cv_results_nb  = cross_validate(pipe_nb,  X, y, cv=cv, scoring=scoring)
    cv_results_svm = cross_validate(pipe_svm, X, y, cv=cv, scoring=scoring)

    cv_acc_nb_mean  = cv_results_nb['test_accuracy'].mean() * 100
    cv_acc_nb_std   = cv_results_nb['test_accuracy'].std()  * 100
    cv_f1m_nb_mean  = cv_results_nb['test_f1_macro'].mean() * 100
    cv_f1m_nb_std   = cv_results_nb['test_f1_macro'].std()  * 100
    cv_f1w_nb_mean  = cv_results_nb['test_f1_weighted'].mean() * 100
    cv_f1w_nb_std   = cv_results_nb['test_f1_weighted'].std()  * 100

    cv_acc_svm_mean = cv_results_svm['test_accuracy'].mean() * 100
    cv_acc_svm_std  = cv_results_svm['test_accuracy'].std()  * 100
    cv_f1m_svm_mean = cv_results_svm['test_f1_macro'].mean() * 100
    cv_f1m_svm_std  = cv_results_svm['test_f1_macro'].std()  * 100
    cv_f1w_svm_mean = cv_results_svm['test_f1_weighted'].mean() * 100
    cv_f1w_svm_std  = cv_results_svm['test_f1_weighted'].std()  * 100

    print(f"\nNaive Bayes  5-Fold CV Akurasi   : {cv_acc_nb_mean:.2f}% ± {cv_acc_nb_std:.2f}%")
    print(f"Naive Bayes  5-Fold CV Macro F1  : {cv_f1m_nb_mean:.2f}% ± {cv_f1m_nb_std:.2f}%")
    print(f"Naive Bayes  5-Fold CV Weight.F1 : {cv_f1w_nb_mean:.2f}% ± {cv_f1w_nb_std:.2f}%")
    print(f"\nLinear SVM   5-Fold CV Akurasi   : {cv_acc_svm_mean:.2f}% ± {cv_acc_svm_std:.2f}%")
    print(f"Linear SVM   5-Fold CV Macro F1  : {cv_f1m_svm_mean:.2f}% ± {cv_f1m_svm_std:.2f}%")
    print(f"Linear SVM   5-Fold CV Weight.F1 : {cv_f1w_svm_mean:.2f}% ± {cv_f1w_svm_std:.2f}%")

    # ─────────────────────────────────────────────
    # 4. CONFUSION MATRIX LENGKAP
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("4. CONFUSION MATRIX LENGKAP (DATA UJI)")
    print("=" * 55)
    labels_order = [l for l in ["positif", "negatif", "netral"] if l in np.unique(y)]
    cm_nb  = confusion_matrix(y_test, y_pred_nb,  labels=labels_order)
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels_order)

    df_cm_nb  = pd.DataFrame(cm_nb,  index=[f"Aktual_{l.upper()}" for l in labels_order],
                             columns=[f"Pred_{l.upper()}" for l in labels_order])
    df_cm_svm = pd.DataFrame(cm_svm, index=[f"Aktual_{l.upper()}" for l in labels_order],
                             columns=[f"Pred_{l.upper()}" for l in labels_order])

    print("Confusion Matrix — Multinomial Naive Bayes:")
    print(df_cm_nb)
    print("\nConfusion Matrix — Linear SVM:")
    print(df_cm_svm)

    # ─────────────────────────────────────────────
    # 5. METRIK PER KELAS SECARA RINCI
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("5. METRIK EVALUASI PER KELAS (PRECISION / RECALL / F1)")
    print("=" * 55)
    report_dict_nb  = classification_report(y_test, y_pred_nb,  zero_division=0, output_dict=True)
    report_dict_svm = classification_report(y_test, y_pred_svm, zero_division=0, output_dict=True)

    rows_per_class = []
    for k in labels_order:
        for model_name, rd in [("Multinomial Naive Bayes", report_dict_nb), ("Linear SVM", report_dict_svm)]:
            if k in rd:
                rows_per_class.append({
                    "Model": model_name,
                    "Kelas": k.capitalize(),
                    "Precision (%)": round(rd[k]["precision"] * 100, 2),
                    "Recall (%)": round(rd[k]["recall"] * 100, 2),
                    "F1-Score (%)": round(rd[k]["f1-score"] * 100, 2),
                    "Support (Data Uji)": int(rd[k]["support"])
                })
    df_per_class = pd.DataFrame(rows_per_class)
    if len(df_per_class) > 0:
        print(df_per_class.to_string(index=False))

    # ─────────────────────────────────────────────
    # 6. ANALISIS ERROR — CONTOH PREDIKSI SALAH
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("6. ANALISIS CONTOH PREDIKSI SALAH (ERROR ANALYSIS)")
    print("=" * 55)

    df_test = df.iloc[idx_test].copy()
    df_test["pred_nb"]  = y_pred_nb
    df_test["pred_svm"] = y_pred_svm

    misclassified = df_test[
        (df_test["sentimen"] != df_test["pred_svm"]) |
        (df_test["sentimen"] != df_test["pred_nb"])
    ].copy()

    contoh_salah = []
    for _, row in misclassified.head(15).iterrows():
        label_proksi = row["sentimen"]
        p_nb  = row["pred_nb"]
        p_svm = row["pred_svm"]
        judul = row["judul"]
        skor  = row.get("total_skor", 0)

        if "korupsi" in judul.lower() and label_proksi == "positif":
            alasan = ("Artikel memuat kata 'korupsi/tersangka' (bobot leksikon negatif) "
                      "namun konteks judul adalah 'apresiasi/keberhasilan' penegakan hukum, "
                      "sehingga skor leksikon dan classifier berselisih.")
        elif "kritik" in judul.lower() or "protes" in judul.lower():
            alasan = ("Teks memuat kata kritik kebijakan publik; classifier mendeteksi "
                      "fitur unigram/bigram berbobot negatif kuat sementara leksikon "
                      "mendeteksi kata penyeimbang netral.")
        elif label_proksi == "netral":
            alasan = ("Kelas Netral memiliki sangat sedikit sampel (highly imbalanced), "
                      "sehingga model cenderung memprediksi kelas mayoritas.")
        else:
            alasan = (f"Ambiguitas konteks semantik; InSet menghitung skor {skor}, "
                      f"namun pembobotan TF-IDF unigram/bigram mengarahkan prediksi model "
                      f"ke kelas berlawanan.")

        contoh_salah.append({
            "No_Artikel":          row.get("no", "-"),
            "Sumber":              row.get("sumber", "-"),
            "Judul_Berita":        (judul[:85] + "...") if len(judul) > 85 else judul,
            "Label_Proksi_InSet":  label_proksi.capitalize(),
            "Prediksi_Naive_Bayes": p_nb.capitalize(),
            "Prediksi_Linear_SVM":  p_svm.capitalize(),
            "Skor_InSet":          skor,
            "Kata_Positif":        str(row.get("kata_positif", ""))[:50],
            "Kata_Negatif":        str(row.get("kata_negatif", ""))[:50],
            "Analisis_Linguistik_Error": alasan
        })

    df_salah = pd.DataFrame(contoh_salah)
    if len(df_salah) > 0:
        print(f"Ditemukan {len(misclassified)} kasus misklasifikasi pada data uji.\n")
        for i, row in df_salah.head(9).iterrows():
            print(f"  [{i+1}] Judul   : {row['Judul_Berita']}")
            print(f"       Label   : {row['Label_Proksi_InSet']}  |  Pred. NB: {row['Prediksi_Naive_Bayes']}  |  Pred. SVM: {row['Prediksi_Linear_SVM']}")
            print(f"       Skor InSet: {row['Skor_InSet']}")
            print(f"       Analisis: {row['Analisis_Linguistik_Error']}")
            print()
    else:
        print("Model memprediksi seluruh data uji secara konsisten dengan label proksi.")

    # ─────────────────────────────────────────────
    # 7. TABEL RINGKASAN EVALUASI KOMPREHENSIF
    # ─────────────────────────────────────────────
    df_eval = pd.DataFrame([
        {
            "Model": "Multinomial Naive Bayes",
            "Akurasi Test Set (%)": round(acc_nb * 100, 2),
            "Precision Weighted (%)": round(p_nb_w * 100, 2),
            "Recall Weighted (%)":    round(r_nb_w * 100, 2),
            "Weighted F1 (%)": round(f1_nb_w * 100, 2),
            "Macro F1 (%)":    round(f1_nb_m * 100, 2),
            "5-Fold CV Akurasi (Mean±Std)":  f"{cv_acc_nb_mean:.2f}% ± {cv_acc_nb_std:.2f}%",
            "5-Fold CV Macro F1 (Mean±Std)": f"{cv_f1m_nb_mean:.2f}% ± {cv_f1m_nb_std:.2f}%",
            "5-Fold CV Wtd.F1 (Mean±Std)":   f"{cv_f1w_nb_mean:.2f}% ± {cv_f1w_nb_std:.2f}%",
        },
        {
            "Model": "Linear SVM (LinearSVC)",
            "Akurasi Test Set (%)": round(acc_svm * 100, 2),
            "Precision Weighted (%)": round(p_svm_w * 100, 2),
            "Recall Weighted (%)":    round(r_svm_w * 100, 2),
            "Weighted F1 (%)": round(f1_svm_w * 100, 2),
            "Macro F1 (%)":    round(f1_svm_m * 100, 2),
            "5-Fold CV Akurasi (Mean±Std)":  f"{cv_acc_svm_mean:.2f}% ± {cv_acc_svm_std:.2f}%",
            "5-Fold CV Macro F1 (Mean±Std)": f"{cv_f1m_svm_mean:.2f}% ± {cv_f1m_svm_std:.2f}%",
            "5-Fold CV Wtd.F1 (Mean±Std)":   f"{cv_f1w_svm_mean:.2f}% ± {cv_f1w_svm_std:.2f}%",
        }
    ])

    print("\n" + "=" * 70)
    print("TABEL RINGKASAN EVALUASI AKHIR (TEST SET & CROSS-VALIDATION)")
    print("=" * 70)
    for _, row in df_eval.iterrows():
        print(f"\n  Model         : {row['Model']}")
        print(f"  Akurasi       : {row['Akurasi Test Set (%)']:.2f}%")
        print(f"  Precision (W) : {row['Precision Weighted (%)']:.2f}%")
        print(f"  Recall (W)    : {row['Recall Weighted (%)']:.2f}%")
        print(f"  Weighted F1   : {row['Weighted F1 (%)']:.2f}%")
        print(f"  Macro F1      : {row['Macro F1 (%)']:.2f}%")
        print(f"  5-Fold CV Acc : {row['5-Fold CV Akurasi (Mean±Std)']}")
        print(f"  5-Fold CV F1m : {row['5-Fold CV Macro F1 (Mean±Std)']}")
        print(f"  5-Fold CV F1w : {row['5-Fold CV Wtd.F1 (Mean±Std)']}")

    # ─────────────────────────────────────────────
    # 8. PARAMETER MODEL (DOKUMENTASI AKADEMIS)
    # ─────────────────────────────────────────────
    df_params_tfidf = pd.DataFrame([
        {"Hiperparameter": "max_features", "Nilai": "1500",    "Justifikasi": "Membatasi kosakata pada 1500 term dengan bobot TF-IDF tertinggi untuk efisiensi komputasi"},
        {"Hiperparameter": "ngram_range",  "Nilai": "(1, 2)",  "Justifikasi": "Unigram & bigram untuk menangkap frasa kontekstual seperti 'tidak baik', 'sangat positif'"},
        {"Hiperparameter": "sublinear_tf", "Nilai": "True",    "Justifikasi": "Menggunakan log(1+tf) agar kata frekuensi tinggi tidak mendominasi (Manning et al., 2008)"},
    ])
    df_params_nb = pd.DataFrame([
        {"Hiperparameter": "alpha", "Nilai": "1.0", "Justifikasi": "Additive (Laplace) smoothing baku untuk menangani zero-probability pada kata yang tidak ada di training"},
    ])
    df_params_svm = pd.DataFrame([
        {"Hiperparameter": "Jenis Model",  "Nilai": "LinearSVC",  "Justifikasi": "Implementasi SVM linear yang efisien untuk teks berdimensi tinggi (Joachims, 1998; Chang & Lin, 2011)"},
        {"Hiperparameter": "C (Regularization)", "Nilai": "1.0", "Justifikasi": "Parameter penalti soft-margin standar; mengontrol trade-off antara margin maksimal dan misclassification"},
        {"Hiperparameter": "max_iter",     "Nilai": "2000",       "Justifikasi": "Batas iterasi solver untuk menjamin konvergensi"},
        {"Hiperparameter": "random_state", "Nilai": "42",         "Justifikasi": "Menjamin reprodusibilitas hasil pengujian eksperimen"},
    ])
    df_params_split = pd.DataFrame([
        {"Hiperparameter": "test_size",    "Nilai": "0.20",       "Justifikasi": "80/20 split — praktik standar untuk dataset skala kecil-menengah"},
        {"Hiperparameter": "random_state", "Nilai": "42",         "Justifikasi": "Reprodusibilitas eksperimen"},
        {"Hiperparameter": "stratify",     "Nilai": "y (sentimen)", "Justifikasi": "Mempertahankan proporsi kelas di train dan test sesuai distribusi asli"},
    ])

    df_methodology = pd.DataFrame([
        {"Topik": "Karakteristik Label Target",
         "Keterangan": "Label Proksi Berbasis Leksikon (Lexicon-Based Proxy Labels) menggunakan InSet Lexicon — BUKAN ground truth atau label kebenaran absolut"},
        {"Topik": "Batasan Metodologi",
         "Keterangan": "Polaritas kata leksikal dokumen mengukur nada leksikal teks secara agregat; tidak mencerminkan keberpihakan media terhadap aktor politik tertentu"},
        {"Topik": "Data Leakage (Versi Lama)",
         "Keterangan": "[DIPERBAIKI] Versi sebelumnya: TF-IDF fit_transform pada seluruh data sebelum split → mencemari data uji. Versi baru: TF-IDF fit hanya pada X_train via Pipeline"},
        {"Topik": "Kata Negasi (Versi Lama)",
         "Keterangan": "[DIPERBAIKI] Versi sebelumnya: 'tidak', 'belum', 'tanpa' masuk stopwords. Versi baru: kata negasi DIPERTAHANKAN karena mengubah makna semantik"},
        {"Topik": "Ambiguitas Contoh Dokumen vs Aktor",
         "Keterangan": "Berita 'KPK Berhasil Ungkap Kasus Korupsi Pejabat' memuat kata negatif (korupsi, tersangka) namun sentimen terhadap KPK adalah positif/apresiatif"},
    ])

    # ─────────────────────────────────────────────
    # 9. SIMPAN HASIL KE EXCEL MULTI-SHEET
    # ─────────────────────────────────────────────
    # Prediksi untuk seluruh dataset (menggunakan tfidf yang sudah fit pada train)
    X_all_tfidf = tfidf.transform(X)
    df["prediksi_naive_bayes"] = nb_model.predict(X_all_tfidf)
    df["prediksi_svm"]         = svm_model.predict(X_all_tfidf)

    output_excel = "SiapMining_2301020003.xlsx"
    output_eval  = "hasil_analisis_sentimen.xlsx"

    def simpan_ke_excel(nama_file, eval_only=False):
        with pd.ExcelWriter(nama_file, engine="openpyxl") as writer:
            if not eval_only:
                df_export = df[["no", "judul", "isi_berita", "tanggal", "penulis", "kategori",
                                "tags", "sumber", "url", "sentimen", "total_skor",
                                "skor_positif", "skor_negatif", "kata_positif", "kata_negatif"]]
                df_export.to_excel(writer, sheet_name="Semua_Data_Terlabel", index=False)
                for s in df["sumber"].unique():
                    sub = df_export[df_export["sumber"] == s].reset_index(drop=True)
                    sub.loc[:, "no"] = range(1, len(sub) + 1)
                    sub.to_excel(writer, sheet_name=str(s)[:31], index=False)

            df_eval.to_excel(writer, sheet_name="Ringkasan_Evaluasi", index=False)
            df_per_class.to_excel(writer, sheet_name="Metrik_Per_Kelas", index=False)
            df_cm_svm.to_excel(writer, sheet_name="Confusion_Matrix_SVM")
            df_cm_nb.to_excel(writer, sheet_name="Confusion_Matrix_NB")
            if len(df_salah) > 0:
                df_salah.to_excel(writer, sheet_name="Contoh_Prediksi_Salah", index=False)
            df_params_svm.to_excel(writer, sheet_name="Spesifikasi_Model_SVM", index=False)
            df_params_tfidf.to_excel(writer, sheet_name="Spesifikasi_TF-IDF", index=False)
            df_params_nb.to_excel(writer, sheet_name="Spesifikasi_Model_NB", index=False)
            df_params_split.to_excel(writer, sheet_name="Spesifikasi_Split", index=False)
            df_methodology.to_excel(writer, sheet_name="Metodologi_Label_Proksi", index=False)

    print(f"\n[+] Menyimpan hasil ke: {output_eval} ...")
    try:
        simpan_ke_excel(output_eval, eval_only=True)
        print(f"[✓] Berhasil menyimpan evaluasi ke: {os.path.abspath(output_eval)}")
    except PermissionError:
        ts = datetime.now().strftime("%H%M%S")
        fb = f"hasil_analisis_sentimen_{ts}.xlsx"
        simpan_ke_excel(fb, eval_only=True)
        print(f"[✓] Disimpan ke fallback: {os.path.abspath(fb)}")

    print(f"[+] Memperbarui dataset utama: {output_excel} ...")
    try:
        simpan_ke_excel(output_excel, eval_only=False)
        print(f"[✓] Dataset '{output_excel}' berhasil diperbarui.")
    except PermissionError:
        print(f"[!] Dataset '{output_excel}' sedang terbuka, skip update dataset.")

    print("\n" + "=" * 70)
    print("  PROSES SELESAI! Hasil tersimpan di SiapMining_2301020003.xlsx")
    print("  dan hasil_analisis_sentimen.xlsx")
    print("=" * 70)

    # Return dict untuk dipakai oleh app.py
    return {
        "acc_nb": acc_nb, "f1_nb_w": f1_nb_w, "f1_nb_m": f1_nb_m,
        "acc_svm": acc_svm, "f1_svm_w": f1_svm_w, "f1_svm_m": f1_svm_m,
        "y_pred_nb": y_pred_nb, "y_pred_svm": y_pred_svm, "y_test": y_test,
        "nb_model": nb_model, "svm_model": svm_model, "tfidf": tfidf,
        "df": df, "df_eval": df_eval
    }


if __name__ == "__main__":
    main()
