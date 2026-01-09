import re
import math
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import ufal.morphodita as morphodita



# =========================
# CONFIG
# =========================

MORPHODITA_MODEL = "czech-morfflex2.1-pdtc2.0-250909.tagger"
TRANSCRIPT_FILE = "prepis.md"

SEGMENT_LEN = 300      # 5 min
SEGMENT_OVERLAP = 120  # 2 min
MERGE_GAP = 5          # seconds

MIN_TOPIC_SIZE = 2

STOP_POS = {
    "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "NUM"
}


# =========================
# HELPERS
# =========================

def parse_time(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def chunk_text(text, max_chars=800):
    chunks = []
    buf = []

    for word in text.split():
        buf.append(word)
        if sum(len(w) for w in buf) > max_chars:
            chunks.append(" ".join(buf))
            buf = []

    if buf:
        chunks.append(" ".join(buf))

    return chunks

# =========================
# STEP 1: LOAD TRANSCRIPT
# =========================

def load_transcript(path) -> pd.DataFrame:
    rows = []

    time_rx = re.compile(r"\[(\d+:\d+:\d+)\]\s+(\w+):")

    current_t = None
    current_speaker = None

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()

        m = time_rx.match(line)
        if m:
            current_t = parse_time(m.group(1))
            current_speaker = m.group(2)
            continue

        if current_t is not None and line:
            rows.append({
                "t": current_t,
                "speaker": current_speaker,
                "text": line
            })

    df = pd.DataFrame(rows)
    return df


# =========================
# STEP 2: merge items said by the same speaker inside the MERGE_GAP
# =========================

def merge_utterances(df):
    merged = []

    current = None

    for row in df.itertuples():
        if (
            current
            and row.speaker == current["speaker"]
            and row.t - current["t_end"] <= MERGE_GAP
        ):
            current["text"] += " " + row.text
            current["t_end"] = row.t
        else:
            if current:
                merged.append(current)
            current = {
                "speaker": row.speaker,
                "t_start": row.t,
                "t_end": row.t,
                "text": row.text
            }

    if current:
        merged.append(current)

    return pd.DataFrame(merged)


# =========================
# STEP 3: divide to segments SEGMENT_LEN long with SEGMENT_OVERLAP
# =========================

def build_segments(df) -> pd.DataFrame:
    t_min = df["t_start"].min()
    t_max = df["t_end"].max()

    segments = []
    t = t_min

    index = 1
    while t < t_max:
        t_end = t + SEGMENT_LEN

        mask = (df["t_end"] >= t) & (df["t_start"] <= t_end)
        chunk = df[mask]

        if not chunk.empty:
            segments.append({
                "t_start": int(t),
                "t_end": int(t_end),
                "index": index ,
                "text": " ".join(chunk["text"].tolist())
            })
            index = index + 1

        t += SEGMENT_LEN - SEGMENT_OVERLAP

    Path("segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return pd.DataFrame(segments)


# =========================
# STEP 4: MorphoDiTa LEMMATIZATION
# =========================

class Lemmatizer:
    def __init__(self, model_path):
        self.tagger = morphodita.Tagger.load(model_path)
        if not self.tagger:
            raise RuntimeError("Cannot load MorphoDiTa model")

        self.tokenizer = self.tagger.newTokenizer()
        self.forms = morphodita.Forms()
        self.lemmas = morphodita.TaggedLemmas()
        self.tokens = morphodita.TokenRanges()

    def lemmatize(self, text):
        self.tokenizer.setText(text)
        lemmas_out = []

        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.tagger.tag(self.forms, self.lemmas)

            for lemma in self.lemmas:
                base = lemma.lemma.split("_")[0]
                tag = lemma.tag

                # POS filtrace (první znak tagu)
                if tag[0] in {"N", "V", "A"} and base.isalpha():
                    lemmas_out.append(base.lower())

        return lemmas_out


# =========================
# STEP 5: TF-IDF
# =========================

def build_tfidf(segments: pd.DataFrame, lemmatizer: Lemmatizer):
    docs = []
    lemma_maps = []

    for seg in tqdm(segments.itertuples(), total=len(segments)):
        lemmas = lemmatizer.lemmatize(seg.text)
        lemma_maps.append(Counter(lemmas))
        docs.append(" ".join(lemmas))

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(docs)
    return X, vectorizer, lemma_maps


# =========================
# STEP 6: CLUSTERING
# =========================

def cluster_segments(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(X.toarray())
    print("cluster labels")
    print(labels)
    return labels


# =========================
# STEP 7: AGGREGATE TOPICS
# =========================

def summarize_topics(segments, labels, lemma_maps):
    topics = defaultdict(list)
    print("topics")
    print(topics)

    for i, label in enumerate(labels):
        if label >= 0:
            print(topics[label])
            topics[label].append(i)
    
    print("topics")
    print(topics)

    summaries = []

    for label, idxs in topics.items():
        t_start = min(segments.iloc[i].t_start for i in idxs)
        t_end = max(segments.iloc[i].t_end for i in idxs)

        lemmas = Counter()
        for i in idxs:
            lemmas.update(lemma_maps[i])

        summaries.append({
            "topic_id": int(label),
            "segments_ids": idxs,
            "segments_count": len(idxs),
            "segments": [segments.iloc[i].text for i in idxs],
            "time_minutes": (t_end - t_start) / 60,
            "top_lemmas": [w for w, _ in lemmas.most_common(15)]
        })

    return sorted(summaries, key=lambda x: -x["time_minutes"])


# =========================
# MAIN
# =========================

def main():
    print("Loading transcript...")
    df = load_transcript(TRANSCRIPT_FILE)

    print("Merging utterances...")
    df = merge_utterances(df)

    print("Building segments...")
    segments = build_segments(df)

    print("Lemmatizing + TF-IDF...")
    lemmatizer = Lemmatizer(MORPHODITA_MODEL)
    X, vectorizer, lemma_maps = build_tfidf(segments, lemmatizer)

    print("Clustering...")
    labels = cluster_segments(X)
    print(labels)

    print("Summarizing topics...")
    topics = summarize_topics(segments, labels, lemma_maps)
    print(topics)

    print("\n=== TOPICS ===")
    for t in topics:
        print(
            f"\nTopic {t['topic_id']} | {t['time_minutes']:.1f} min"
        )
        print(", ".join(t["top_lemmas"]))

    Path("topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
