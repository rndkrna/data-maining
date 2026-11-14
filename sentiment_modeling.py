"""
=============================================================================
  ANALISIS SENTIMEN BERITA POLITIK INDONESIA (DATA MINING)
=============================================================================
  Metodologi Akademis Standar:
  1. Pelabelan Menggunakan "Label Proksi Berbasis Leksikon":
     - Menggunakan Indonesian Sentiment Lexicon (InSet) karya Koto & Rahmaningtyas (2017)
     - Label berfungsi sebagai "proksi awal" (proxy ground truth)
     - Catatan Metodologi: Polaritas kata pada tingkat dokumen tidak sama 
       dengan stance/keberpihakan politik terhadap aktor tertentu.
  2. Text Preprocessing (Cleansing Regex, Case Folding, Stopwords Removal)
  3. Ekstraksi Fitur: TF-IDF Vectorizer (Unigram & Bigram, Max Features: 1500)
  4. Klasifikasi Terawasi (Supervised Learning):
     - Multinomial Naive Bayes (alpha=1.0)
     - Support Vector Machine (Kernel Linear, C=1.0, tol=1e-3, random_state=42)
  5. Validasi:
     - Hold-out Test Set (80% Train, 20% Test, Stratified)
     - 5-Fold Stratified Cross-Validation (Mean & Std Dev)
     - Evaluasi Macro F1, Weighted Metrics, dan Metrik Per-Kelas
     - Analisis Confusion Matrix Lengkap
     - Identifikasi & Analisis Diagnostik Contoh Prediksi Salah (Error Analysis)
  6. Ekspor Audit Lengkap ke Excel Multi-Sheet
=============================================================================
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
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
STOPWORDS_ID = set([
    "yang", "di", "dan", "dengan", "untuk", "dari", "ke", "ini", "itu", "pada",
    "adalah", "bahwa", "dalam", "akan", "juga", "atau", "oleh", "saat", "tersebut",
    "bisa", "tidak", "ada", "karena", "ia", "mereka", "kita", "kami", "telah",
    "seperti", "sudah", "hanya", "oleh", "sebagai", "para", "lebih", "dapat",
    "antara", "setelah", "namun", "tetapi", "serta", "ketika", "bagi", "lalu",
    "masih", "belum", "secara", "lagi", "sehingga", "maupun", "bahkan", "tanpa",
    "hampir", "terhadap", "maka", "tentang", "selama", "kemudian", "kepada",
    "harus", "banyak", "hal", "pun", "saja", "menjadi", "diri", "seperti", "tiap"
])

def preprocess_text(teks: str) -> str:
    """Membersihkan teks berita untuk ekstraksi fitur TF-IDF."""
    if not isinstance(teks, str):
        return ""

    # Case folding
    teks = teks.lower()
    # Hapus URL
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)
    # Hapus karakter khusus dan angka
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    # Tokenisasi sederhana & buang stopword
    words = [w for w in teks.split() if w not in STOPWORDS_ID and len(w) > 2]

    return " ".join(words)


# ─────────────────────────────────────────────
# 2. FUNGSI UTAMA PIPELINE SENTIMEN
# ─────────────────────────────────────────────
def main(input_excel: str = "dataset_berita.xlsx"):
    print("=" * 70)
    print("  MEMULAI PIPELINE ANALISIS SENTIMEN DENGAN LABEL PROKSI INSET")
    print("=" * 70)

    if not os.path.exists(input_excel):
        fallback_files = ["dataset_berita_politik.xlsx", "test_run_all.xlsx"]
        for fb in fallback_files:
            if os.path.exists(fb):
                input_excel = fb
                break

    if not os.path.exists(input_excel):
        print(f"[!] Error: File dataset '{input_excel}' tidak ditemukan.")
        return

    print(f"[+] Membaca file dataset: {input_excel}")
    xl = pd.ExcelFile(input_excel)
    sheet_name = "Semua_Data" if "Semua_Data" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    print(f"[+] Total data berita dimuat: {len(df)} artikel")

    # ─────────────────────────────────────────────
    # PREPROCESSING: CLEANING MISSING VALUE & DEDUPLIKASI
    # ─────────────────────────────────────────────
    # 1. Missing Value Handling
    sebelum_mv = len(df)
    if "penulis" in df.columns:
        df["penulis"] = df["penulis"].fillna("Tim Redaksi")
    df.dropna(subset=["judul", "isi_berita"], inplace=True)
    df = df[df["judul"].astype(str).str.strip().ne("") & df["isi_berita"].astype(str).str.strip().ne("")]
    if len(df) < sebelum_mv:
        print(f"  [i] Dihapus {sebelum_mv - len(df)} baris karena judul atau isi berita kosong.")

    # 2. Deduplikasi Data (Menghapus berita kembar)
    sebelum_dup = len(df)
    if "url" in df.columns:
        df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    df.drop_duplicates(subset=["judul"], keep="first", inplace=True)
    if len(df) < sebelum_dup:
        print(f"  [i] Dihapus {sebelum_dup - len(df)} baris duplikat (URL atau judul sama).")

    df.reset_index(drop=True, inplace=True)
    df["no"] = range(1, len(df) + 1)
    print(f"[+] Total korpus berita bersih & unik: {len(df)} artikel")

    # Inisialisasi InSet Lexicon Labeler
    print("\n[+] Menginisialisasi InSet Lexicon (Koto & Rahmaningtyas, 2017)...")
    labeler = InSetLexiconLabeler()

    # Gabungkan teks judul + isi berita untuk cakupan representatif
    df["text_full"] = df["judul"].fillna("") + " " + df["isi_berita"].fillna("")

    # Pelabelan Proksi Ilmiah menggunakan InSet Lexicon
    print("[+] Menghitung skor polaritas InSet untuk menghasilkan LABEL PROKSI BERBASIS LEKSIKON...")
    labeling_results = [labeler.score_text(t) for t in df["text_full"]]
    df_scores = pd.DataFrame(labeling_results)

    df["skor_positif"] = df_scores["skor_positif"]
    df["skor_negatif"] = df_scores["skor_negatif"]
    df["total_skor"] = df_scores["total_skor"]
    df["sentimen"] = df_scores["sentimen"]
    df["kata_positif"] = df_scores["kata_positif"]
    df["kata_negatif"] = df_scores["kata_negatif"]

    print("\n" + "-" * 55)
    print("DISTRIBUSI KELAS LABEL PROKSI BERBASIS LEKSIKON (INSET):")
    print("-" * 55)
    dist = df["sentimen"].value_counts()
    for kelas, count in dist.items():
        persen = (count / len(df)) * 100
        print(f"  • {kelas.upper():<10}: {count} artikel ({persen:.2f}%)")

    # Preprocessing teks untuk machine learning
    print("\n[+] Melakukan Preprocessing Teks (Cleansing, Case Folding, Stopwords)...")
    df["clean_text"] = df["text_full"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["sentimen"]

    # Ekstraksi Fitur TF-IDF (Unigram & Bigram)
    print("[+] Ekstraksi Fitur TF-IDF (N-gram 1-2, Max Features: 1500)...")
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)

    # Pembagian Data Train & Test (80% Train, 20% Test) dengan Stratifikasi
    print("[+] Membagi Data: 80% Training, 20% Testing (Stratified Split)...")
    train_idx, test_idx, y_train, y_test = train_test_split(
        np.arange(len(df)), y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_tfidf[train_idx]
    X_test = X_tfidf[test_idx]

    print(f"  - Data Latih (Train) : {X_train.shape[0]} artikel")
    print(f"  - Data Uji (Test)    : {X_test.shape[0]} artikel")

    # ─────────────────────────────────────────────
    # MODEL 1: MULTINOMIAL NAIVE BAYES
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("1. MODEL MULTINOMIAL NAIVE BAYES")
    print("=" * 55)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    p_nb_w, r_nb_w, f1_nb_w, _ = precision_recall_fscore_support(y_test, y_pred_nb, average="weighted", zero_division=0)
    p_nb_m, r_nb_m, f1_nb_m, _ = precision_recall_fscore_support(y_test, y_pred_nb, average="macro", zero_division=0)

    print(f"Akurasi Naive Bayes       : {acc_nb * 100:.2f}%")
    print(f"Presisi (Weighted)        : {p_nb_w * 100:.2f}%")
    print(f"Recall (Weighted)         : {r_nb_w * 100:.2f}%")
    print(f"F1-Score (Weighted)       : {f1_nb_w * 100:.2f}%")
    print(f"F1-Score (Macro) ⭐        : {f1_nb_m * 100:.2f}%\n")
    print("Laporan Klasifikasi Naive Bayes:")
    print(classification_report(y_test, y_pred_nb, zero_division=0))

    # ─────────────────────────────────────────────
    # MODEL 2: SUPPORT VECTOR MACHINE (LINEAR SVM)
    # ─────────────────────────────────────────────
    print("=" * 55)
    print("2. MODEL SUPPORT VECTOR MACHINE (LINEAR SVM)")
    print("=" * 55)
    # Catat parameter SVM secara transparan
    svm_params = {
        "kernel": "linear",
        "C": 1.0,
        "tol": 1e-3,
        "max_iter": -1,
        "decision_function_shape": "ovr",
        "random_state": 42
    }
    svm_model = SVC(**svm_params)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    p_svm_w, r_svm_w, f1_svm_w, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="weighted", zero_division=0)
    p_svm_m, r_svm_m, f1_svm_m, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="macro", zero_division=0)

    print(f"Akurasi Linear SVM        : {acc_svm * 100:.2f}%")
    print(f"Presisi (Weighted)        : {p_svm_w * 100:.2f}%")
    print(f"Recall (Weighted)         : {r_svm_w * 100:.2f}%")
    print(f"F1-Score (Weighted)       : {f1_svm_w * 100:.2f}%")
    print(f"F1-Score (Macro) ⭐        : {f1_svm_m * 100:.2f}%\n")
    print("Laporan Klasifikasi Linear SVM:")
    print(classification_report(y_test, y_pred_svm, zero_division=0))

    # ─────────────────────────────────────────────
    # 3. EXPERIMENT TAMBAHAN: 5-FOLD STRATIFIED CV
    # ─────────────────────────────────────────────
    print("=" * 55)
    print("3. EKSPERIMEN TAMBAHAN: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 55)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "f1_weighted"]

    cv_results_nb = cross_validate(MultinomialNB(alpha=1.0), X_tfidf, y, cv=cv, scoring=scoring)
    cv_results_svm = cross_validate(SVC(**svm_params), X_tfidf, y, cv=cv, scoring=scoring)

    cv_acc_nb_mean = cv_results_nb['test_accuracy'].mean() * 100
    cv_acc_nb_std  = cv_results_nb['test_accuracy'].std() * 100
    cv_f1m_nb_mean = cv_results_nb['test_f1_macro'].mean() * 100
    cv_f1m_nb_std  = cv_results_nb['test_f1_macro'].std() * 100

    cv_acc_svm_mean = cv_results_svm['test_accuracy'].mean() * 100
    cv_acc_svm_std  = cv_results_svm['test_accuracy'].std() * 100
    cv_f1m_svm_mean = cv_results_svm['test_f1_macro'].mean() * 100
    cv_f1m_svm_std  = cv_results_svm['test_f1_macro'].std() * 100

    print(f"Naive Bayes 5-Fold CV Akurasi    : {cv_acc_nb_mean:.2f}% ± {cv_acc_nb_std:.2f}%")
    print(f"Naive Bayes 5-Fold CV Macro F1   : {cv_f1m_nb_mean:.2f}% ± {cv_f1m_nb_std:.2f}%")
    print(f"Linear SVM 5-Fold CV Akurasi     : {cv_acc_svm_mean:.2f}% ± {cv_acc_svm_std:.2f}%")
    print(f"Linear SVM 5-Fold CV Macro F1    : {cv_f1m_svm_mean:.2f}% ± {cv_f1m_svm_std:.2f}%\n")

    # ─────────────────────────────────────────────
    # 4. CONFUSION MATRIX LENGKAP
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("4. CONFUSION MATRIX LENGKAP (DATA UJI)")
    print("=" * 55)
    labels_order = [l for l in ["positif", "negatif", "netral"] if l in y.unique()]
    cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels_order)
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels_order)

    df_cm_nb = pd.DataFrame(cm_nb, index=[f"Aktual_{l.upper()}" for l in labels_order], columns=[f"Pred_{l.upper()}" for l in labels_order])
    df_cm_svm = pd.DataFrame(cm_svm, index=[f"Aktual_{l.upper()}" for l in labels_order], columns=[f"Pred_{l.upper()}" for l in labels_order])

    print("Confusion Matrix Multinomial Naive Bayes:")
    print(df_cm_nb)
    print("\nConfusion Matrix Linear SVM:")
    print(df_cm_svm)

    # ─────────────────────────────────────────────
    # 5. METRIK PER KELAS SECARA RINCI
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("5. METRIK EVALUASI PER KELAS (PRECISION / RECALL / F1)")
    print("=" * 55)
    report_dict_nb = classification_report(y_test, y_pred_nb, zero_division=0, output_dict=True)
    report_dict_svm = classification_report(y_test, y_pred_svm, zero_division=0, output_dict=True)

    rows_per_class = []
    for k in labels_order:
        if k in report_dict_nb:
            rows_per_class.append({
                "Model": "Multinomial Naive Bayes",
                "Kelas": k.capitalize(),
                "Precision (%)": round(report_dict_nb[k]["precision"] * 100, 2),
                "Recall (%)": round(report_dict_nb[k]["recall"] * 100, 2),
                "F1-Score (%)": round(report_dict_nb[k]["f1-score"] * 100, 2),
                "Support (Data Uji)": report_dict_nb[k]["support"]
            })
        if k in report_dict_svm:
            rows_per_class.append({
                "Model": "Linear SVM",
                "Kelas": k.capitalize(),
                "Precision (%)": round(report_dict_svm[k]["precision"] * 100, 2),
                "Recall (%)": round(report_dict_svm[k]["recall"] * 100, 2),
                "F1-Score (%)": round(report_dict_svm[k]["f1-score"] * 100, 2),
                "Support (Data Uji)": report_dict_svm[k]["support"]
            })
    df_per_class = pd.DataFrame(rows_per_class)
    if len(df_per_class) > 0:
        print(df_per_class.to_string(index=False))

    # ─────────────────────────────────────────────
    # 6. ANALISIS ERROR & CONTOH PREDIKSI SALAH
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("6. ANALISIS CONTOH PREDIKSI SALAH (ERROR ANALYSIS)")
    print("=" * 55)

    df_test = df.iloc[test_idx].copy()
    df_test["pred_nb"] = y_pred_nb
    df_test["pred_svm"] = y_pred_svm

    misclassified = df_test[(df_test["sentimen"] != df_test["pred_svm"]) | (df_test["sentimen"] != df_test["pred_nb"])].copy()

    # Buat tabel analisis kesalahan
    contoh_salah = []
    for idx, row in misclassified.head(15).iterrows():
        label_proksi = row["sentimen"]
        p_nb = row["pred_nb"]
        p_svm = row["pred_svm"]
        judul = row["judul"]
        skor = row["total_skor"]

        # Analisis linguistik mengapa terjadi perbedaan
        if "korupsi" in judul.lower() and label_proksi == "positif":
            alasan = "Artikel memuat kata 'korupsi/tersangka' (bobot leksikon negatif) namun konteks judul adalah 'apresiasi/keberhasilan' penegakan hukum sehingga skor leksikon dan classifier berselisih."
        elif "kritik" in judul.lower() or "protes" in judul.lower():
            alasan = "Teks memuat kata kritik kebijakan publik; classifier mendeteksi fitur unigram/bigram dengan bobot negatif kuat sementara leksikon mendeteksi kata penyeimbang netral."
        elif label_proksi == "netral":
            alasan = "Kelas Netral memiliki sampel sangat sedikit (imbalanced), sehingga model cenderung memprediksi kelas mayoritas (positif/negatif)."
        else:
            alasan = f"Ambiguitas konteks kata polaritas politik; InSet menghitung skor {skor}, namun pembobotan TF-IDF unigram/bigram mengarahkan prediksi model ke kelas berlawanan."

        contoh_salah.append({
            "No_Artikel": row["no"],
            "Sumber": row["sumber"],
            "Judul_Berita": judul[:85] + ("..." if len(judul) > 85 else ""),
            "Label_Proksi_InSet": label_proksi.capitalize(),
            "Prediksi_Naive_Bayes": p_nb.capitalize(),
            "Prediksi_Linear_SVM": p_svm.capitalize(),
            "Skor_InSet": skor,
            "Kata_Positif": str(row["kata_positif"])[:40],
            "Kata_Negatif": str(row["kata_negatif"])[:40],
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
            "Weighted F1 (%)": round(f1_nb_w * 100, 2),
            "Macro F1 (%)": round(f1_nb_m * 100, 2),
            "5-Fold CV Akurasi (Mean ± Std)": f"{cv_acc_nb_mean:.2f}% ± {cv_acc_nb_std:.2f}%",
            "5-Fold CV Macro F1 (Mean ± Std)": f"{cv_f1m_nb_mean:.2f}% ± {cv_f1m_nb_std:.2f}%"
        },
        {
            "Model": "Support Vector Machine (Linear SVM)",
            "Akurasi Test Set (%)": round(acc_svm * 100, 2),
            "Weighted F1 (%)": round(f1_svm_w * 100, 2),
            "Macro F1 (%)": round(f1_svm_m * 100, 2),
            "5-Fold CV Akurasi (Mean ± Std)": f"{cv_acc_svm_mean:.2f}% ± {cv_acc_svm_std:.2f}%",
            "5-Fold CV Macro F1 (Mean ± Std)": f"{cv_f1m_svm_mean:.2f}% ± {cv_f1m_svm_std:.2f}%"
        }
    ])

    print("\n" + "=" * 55)
    print("TABEL RINGKASAN EVALUASI AKHIR (TEST SET & CROSS-VALIDATION)")
    print("=" * 55)
    print(df_eval.to_string(index=False))

    # ─────────────────────────────────────────────
    # 8. CATATAN SPESIFIKASI PARAMETER SVM
    # ─────────────────────────────────────────────
    df_params_svm = pd.DataFrame([
        {"Hiperparameter": "Kernel", "Nilai": "linear", "Justifikasi": "Sesuai teori Joachims (1998), text mining dengan TF-IDF berdimensi tinggi terpisah secara linier"},
        {"Hiperparameter": "C (Regularization)", "Nilai": "1.0", "Justifikasi": "Parameter penalti standar soft-margin untuk mengontrol trade-off margin dan misclassification"},
        {"Hiperparameter": "Tolerance (tol)", "Nilai": "0.001 (1e-3)", "Justifikasi": "Kriteria konvergensi algoritma optimasi Sequential Minimal Optimization (SMO)"},
        {"Hiperparameter": "Max Iterations", "Nilai": "-1 (tanpa batas)", "Justifikasi": "Memastikan solver mencapai titik konvergensi optimum global"},
        {"Hiperparameter": "Decision Function", "Nilai": "ovr (One-vs-Rest)", "Justifikasi": "Skema baku klasifikasi multi-kelas untuk kelas Positif, Negatif, Netral"},
        {"Hiperparameter": "Random State", "Nilai": "42", "Justifikasi": "Menjamin reprodusibilitas hasil pengujian eksperimen"}
    ])

    # ─────────────────────────────────────────────
    # 9. METODOLOGI & BATASAN ANALISIS (STANCE VS POLARITY)
    # ─────────────────────────────────────────────
    df_methodology = pd.DataFrame([
        {"Topik Metodologis": "Karakteristik Label Target", "Keterangan": "Label Proksi Berbasis Leksikon (Lexicon-Based Proxy Labels) menggunakan InSet Lexicon"},
        {"Topik Metodologis": "Batasan Metodologi Ilmiah", "Keterangan": "Polaritas kata leksikal dokumen HANYA mengukur nada leksikal teks secara agregat, TIDAK mencerminkan keberpihakan/sentimen media terhadap aktor politik tertentu."},
        {"Topik Metodologis": "Contoh Ambiguitas Dokumen vs Aktor", "Keterangan": "Berita 'KPK Berhasil Ungkap Kasus Korupsi Pejabat' memuat banyak kata negatif (korupsi, tersangka, suap), namun sentimen terhadap lembaga KPK adalah SANGAT POSITIF/APRESIATIF."},
        {"Topik Metodologis": "Unit Analisis yang Direkomendasikan", "Keterangan": "Target-Based Sentiment Analysis (TBSA) / Aspect-Based Sentiment Analysis (ABSA) & Stance Detection"},
        {"Topik Metodologis": "Skema Anotasi Lanjutan", "Keterangan": "Memetakan Tuple [Entitas Target, Sumber Pernyataan (Kutipan/Redaksi), Polaritas Sikap (Mendukung/Menentang/Netral)] dengan validasi Cohen/Fleiss Kappa >= 0.70"}
    ])

    # ─────────────────────────────────────────────
    # 10. SIMPAN SELURUH HASIL KE EXCEL MULTI-SHEET
    # ─────────────────────────────────────────────
    output_eval_excel = "hasil_analisis_sentimen.xlsx"

    # Jika file sedang terbuka di Excel, simpan ke nama alternatif otomatis
    from datetime import datetime
    def simpan_excel(nama_file):
        df["prediksi_naive_bayes"] = nb_model.predict(X_tfidf)
        df["prediksi_svm"] = svm_model.predict(X_tfidf)
        with pd.ExcelWriter(nama_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Semua_Data_Terlabel", index=False)
            df_eval.to_excel(writer, sheet_name="Ringkasan_Evaluasi", index=False)
            df_per_class.to_excel(writer, sheet_name="Metrik_Per_Kelas", index=False)
            df_cm_svm.to_excel(writer, sheet_name="Confusion_Matrix_SVM")
            df_cm_nb.to_excel(writer, sheet_name="Confusion_Matrix_NB")
            if len(df_salah) > 0:
                df_salah.to_excel(writer, sheet_name="Contoh_Prediksi_Salah", index=False)
            df_params_svm.to_excel(writer, sheet_name="Spesifikasi_Model_SVM", index=False)
            df_methodology.to_excel(writer, sheet_name="Metodologi_Label_Proksi", index=False)

    # Prediksi untuk seluruh dataset
    df["prediksi_naive_bayes"] = nb_model.predict(X_tfidf)
    df["prediksi_svm"] = svm_model.predict(X_tfidf)

    print(f"\n[+] Menyimpan hasil analisis komparatif ke: {output_eval_excel} ...")
    try:
        simpan_excel(output_eval_excel)
        print(f"[✓] Berhasil menyimpan evaluasi lengkap ke: {os.path.abspath(output_eval_excel)}")
    except PermissionError:
        # File sedang terbuka di Excel — simpan ke nama sementara
        ts = datetime.now().strftime("%H%M%S")
        fallback = f"hasil_analisis_sentimen_{ts}.xlsx"
        print(f"[!] File '{output_eval_excel}' sedang terbuka di Excel.")
        print(f"[+] Menyimpan ke file alternatif: {fallback} ...")
        simpan_excel(fallback)
        print(f"[✓] Berhasil disimpan ke: {os.path.abspath(fallback)}")
        print(f"[i] Tutup '{output_eval_excel}' di Excel lalu rename '{fallback}' jika diperlukan.")

    # Update file dataset utama dengan label proksi InSet
    print(f"[+] Memperbarui dataset utama '{input_excel}' dengan label proksi InSet...")
    df_export = df[["no", "judul", "isi_berita", "tanggal", "penulis", "kategori", "tags", "sumber", "url", "sentimen", "total_skor", "skor_positif", "skor_negatif", "kata_positif", "kata_negatif"]]
    try:
        with pd.ExcelWriter(input_excel, engine="openpyxl") as writer:
            df_export.to_excel(writer, sheet_name="Semua_Data", index=False)
            for s in df["sumber"].unique():
                sub = df_export[df_export["sumber"] == s].reset_index(drop=True)
                sub["no"] = range(1, len(sub) + 1)
                sheet_title = str(s)[:31]
                sub.to_excel(writer, sheet_name=sheet_title, index=False)
        if os.path.exists("dataset_berita_politik.xlsx"):
            df_export.to_excel("dataset_berita_politik.xlsx", index=False)
        print(f"[✓] Dataset '{input_excel}' berhasil diperbarui dengan label proksi sentimen!")
    except PermissionError:
        print(f"[!] Dataset '{input_excel}' sedang terbuka di Excel, skip update dataset.")
    print("\n" + "=" * 70)
    print("  PROSES SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
