import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import re

STOPWORDS_ID = set([
    'yang', 'di', 'dan', 'dengan', 'untuk', 'dari', 'ke', 'ini', 'itu', 'pada',
    'adalah', 'bahwa', 'dalam', 'akan', 'juga', 'atau', 'oleh', 'saat', 'tersebut',
    'bisa', 'ada', 'karena', 'ia', 'mereka', 'kita', 'kami', 'telah',
    'seperti', 'sudah', 'hanya', 'sebagai', 'para', 'lebih', 'dapat',
    'antara', 'setelah', 'namun', 'tetapi', 'serta', 'ketika', 'bagi', 'lalu',
    'secara', 'lagi', 'sehingga', 'maupun', 'bahkan',
    'hampir', 'terhadap', 'maka', 'tentang', 'selama', 'kemudian', 'kepada',
    'harus', 'banyak', 'hal', 'pun', 'saja', 'menjadi', 'diri', 'tiap'
])

def preprocess_text(teks):
    if not isinstance(teks, str): return ''
    teks = teks.lower()
    teks = re.sub(r'https?://\S+|www\.\S+', ' ', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\d+', ' ', teks)
    words = [w for w in teks.split() if w not in STOPWORDS_ID and len(w) >= 3]
    return ' '.join(words)

df = pd.read_excel('SiapMining_2301020003.xlsx', sheet_name='Semua_Data_Terlabel')
print('Total data:', len(df))
print('Distribusi sentimen:\n', df['sentimen'].value_counts())

df['text_full'] = df['judul'].fillna('') + ' ' + df['isi_berita'].fillna('')
df['clean_text'] = df['text_full'].apply(preprocess_text)

X = df['clean_text'].values
y = df['sentimen'].values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)
print(f'Train: {len(X_train)} (Pos: {(y_train=="positif").sum()}, Neg: {(y_train=="negatif").sum()})')
print(f'Test: {len(X_test)} (Pos: {(y_test=="positif").sum()}, Neg: {(y_test=="negatif").sum()})')

tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# NB
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_tfidf, y_train)
pred_nb = nb.predict(X_test_tfidf)

# SVM
svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm.fit(X_train_tfidf, y_train)
pred_svm = svm.predict(X_test_tfidf)

labels = ['negatif', 'positif']

print('\n=== METRIK MULTINOMIAL NAIVE BAYES ===')
print('Akurasi:', accuracy_score(y_test, pred_nb))
p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, pred_nb, average='weighted')
p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, pred_nb, average='macro')
print(f'Weighted: P={p_w*100:.2f}%, R={r_w*100:.2f}%, F1={f1_w*100:.2f}%')
print(f'Macro: P={p_m*100:.2f}%, R={r_m*100:.2f}%, F1={f1_m*100:.2f}%')
print('Classification Report:\n', classification_report(y_test, pred_nb, digits=4))
print('Confusion Matrix (neg, pos):\n', confusion_matrix(y_test, pred_nb, labels=labels))

print('\n=== METRIK LINEAR SVM ===')
print('Akurasi:', accuracy_score(y_test, pred_svm))
p_w_s, r_w_s, f1_w_s, _ = precision_recall_fscore_support(y_test, pred_svm, average='weighted')
p_m_s, r_m_s, f1_m_s, _ = precision_recall_fscore_support(y_test, pred_svm, average='macro')
print(f'Weighted: P={p_w_s*100:.2f}%, R={r_w_s*100:.2f}%, F1={f1_w_s*100:.2f}%')
print(f'Macro: P={p_m_s*100:.2f}%, R={r_m_s*100:.2f}%, F1={f1_m_s*100:.2f}%')
print('Classification Report:\n', classification_report(y_test, pred_svm, digits=4))
print('Confusion Matrix (neg, pos):\n', confusion_matrix(y_test, pred_svm, labels=labels))

print('\n=== 5-FOLD STRATIFIED CV DENGAN PIPELINE ===')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe_nb = Pipeline([('tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1,2), sublinear_tf=True)), ('clf', MultinomialNB(alpha=1.0))])
pipe_svm = Pipeline([('tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1,2), sublinear_tf=True)), ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42))])

cv_nb = cross_validate(pipe_nb, X, y, cv=cv, scoring=['accuracy', 'f1_macro', 'f1_weighted'])
cv_svm = cross_validate(pipe_svm, X, y, cv=cv, scoring=['accuracy', 'f1_macro', 'f1_weighted'])

print('CV NB Akurasi per fold:', [round(x*100, 2) for x in cv_nb['test_accuracy']])
print('CV NB Macro F1 per fold:', [round(x*100, 2) for x in cv_nb['test_f1_macro']])
print(f'CV NB Mean Acc: {cv_nb["test_accuracy"].mean()*100:.2f}% ± {cv_nb["test_accuracy"].std()*100:.2f}%')
print(f'CV NB Mean Macro F1: {cv_nb["test_f1_macro"].mean()*100:.2f}% ± {cv_nb["test_f1_macro"].std()*100:.2f}%')
print(f'CV NB Mean Weighted F1: {cv_nb["test_f1_weighted"].mean()*100:.2f}% ± {cv_nb["test_f1_weighted"].std()*100:.2f}%')

print('\nCV SVM Akurasi per fold:', [round(x*100, 2) for x in cv_svm['test_accuracy']])
print('CV SVM Macro F1 per fold:', [round(x*100, 2) for x in cv_svm['test_f1_macro']])
print(f'CV SVM Mean Acc: {cv_svm["test_accuracy"].mean()*100:.2f}% ± {cv_svm["test_accuracy"].std()*100:.2f}%')
print(f'CV SVM Mean Macro F1: {cv_svm["test_f1_macro"].mean()*100:.2f}% ± {cv_svm["test_f1_macro"].std()*100:.2f}%')
print(f'CV SVM Mean Weighted F1: {cv_svm["test_f1_weighted"].mean()*100:.2f}% ± {cv_svm["test_f1_weighted"].std()*100:.2f}%')

# Misclassifications
df_test = df.iloc[idx_test].copy()
df_test['pred_nb'] = pred_nb
df_test['pred_svm'] = pred_svm

misclassified = df_test[(df_test['sentimen'] != df_test['pred_nb']) | (df_test['sentimen'] != df_test['pred_svm'])]
print(f'\nTotal misclassified pada data uji: {len(misclassified)}')
for idx, r in misclassified.iterrows():
    print(f"- No: {r['no']}, Sumber: {r['sumber']}, Label: {r['sentimen']}, NB: {r['pred_nb']}, SVM: {r['pred_svm']}, Skor: {r['total_skor']}")
    print(f"  Judul: {r['judul'][:80]}...")
