import argparse
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import ufal.morphodita as morphodita



# =========================
# CONFIG
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
MORPHODITA_MODEL = str(SCRIPT_DIR / "czech-morfflex2.1-pdtc2.0-250909.tagger")

SEGMENT_LEN = 300      # 5 min
SEGMENT_OVERLAP = 120  # 2 min
MERGE_GAP = 5          # seconds

MIN_TOPIC_SIZE = 2

STOP_POS = {
    "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "NUM"
}

UNWANTED_KEYWORDS = {
    "být", "mít", "říci", "říkat", "chtít", "moci",
    "dělat", "vědět", "myslit", "jít", "prosit", 
    "udělat", "řešit", "mluvit", "věc"
}

COMMON_MISTAKES = {
    "písavný": "písemné",
    "Litovla": "Litovle",
    "Litového": "Litovel",
    "Litovl": "Litovel",
    "Litovélo": "Litovel",
    "Stavěnový": "Stavební",
    "Stavěvní": "Stavební",
    "navědomý": "na vědomí",
    "zápisě": "zápise",
    "krátkrobým": "krátkodobým",
    "Ritovel": "Litovel",
    "rozpoštením": "rozpočtovým",
    "Litovilsko": "Litovelsko",
    "na Sovburgách": "v Nasobůrkách",
    "psířiště": "psí hřiště",
    " krum": " korun",
    "zasadeny": "zasazeny",
    "litovaské": "litovelské",
    "po zemku": "pozemku",
    "dobudové": "důvodové",
    "Litovelezero": "Litovel s.r.o.",
    "Žejrenko": "", #TODO
    "řezové": "Březové",
    "Alomouckem": "Olomouckém",
    "Alomoucko": "Olomouckou",
    "Alomouckej": "Olomoucké",
    " toveláci": " litoveláci"
}

DOMAIN_HINTS = {
    "stavba": "průběh stavby",
    "silnice": "místní komunikace",
    "výkop": "stavební práce",
    "vodovod": "vodovodní infrastruktura",
    "kanalizace": "kanalizace",
    "dotace": "dotace a financování",
    "obyvatel": "dopad na obyvatele",
    "komunikace": "komunikace města s občany",
    "kontrola": "kontrola a dohled",
    "usnesení": "postup orgánů města",
    "pozemek": "majetek města",
    "škola": "školství",
    "mikroregion": "meziobecní spolupráce",
}

HINT_KEYWORDS = {
    "školství": ["škola", "školy", "školní", "žák", "učitel"],
    "vodovodní infrastruktura": ["vodovod", "přípojka", "voda"],
    "kanalizace": ["kanalizace", "kanál"],
    "místní komunikace": ["silnice", "chodník", "cesta"],
    "průběh stavby": ["stavba", "výkop", "projekt"],
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

def segment_to_json(row):
    return {
        "id": int(row.name),
        "start": float(row.t_start),
        "end": float(row.t_end),
        "duration": float(row.t_end - row.t_start),
        "speakers": list(row.speakers),
        "speaker_count": len(row.speakers),
        "speaker_texts": {
            speaker: text
            for speaker, text in row.speaker_texts.items()
        },
        "word_count": int(
            sum(len(txt.split()) for txt in row.speaker_texts.values())
        )
    }

def deduplicate_sentences(sentences):
    # zachová pořadí, odstraní doslovné duplicity
    return list(dict.fromkeys(
        s.strip() for s in sentences if len(s.split()) > 8
    ))


def jaccard_similarity(a, b):
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def find_hint_sentence(sentences, hint):
    if not hint or hint not in HINT_KEYWORDS:
        return None

    keywords = HINT_KEYWORDS[hint]

    for s in sentences:
        s_l = s.lower()
        if any(k in s_l for k in keywords):
            return s

    return None

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

        for wrong in sorted(COMMON_MISTAKES, key=len, reverse=True):
            line = line.replace(wrong, COMMON_MISTAKES[wrong])

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

def merge_utterances(df) -> pd.DataFrame:
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
                "text": row.text,
                "word_count": len(row.text.split())
            }

    if current:
        merged.append(current)

    return pd.DataFrame(merged)


# =========================
# STEP 3: divide to segments SEGMENT_LEN long with SEGMENT_OVERLAP
# =========================

def build_segments(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
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
                "index": index,
                "speakers": list(set(chunk["speaker"])),
                "speaker_texts": (
                    chunk.groupby("speaker")["text"]
                    .apply(lambda x: " ".join(x))
                    .to_dict()
                ),
                "word_count": int(chunk["text"].str.split().str.len().sum())
            })
            index = index + 1

        t += SEGMENT_LEN - SEGMENT_OVERLAP

    Path(outdir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return pd.DataFrame(segments)


# =========================
# STEP 4: MorphoDiTa LEMMATIZATION - NLP technique to find word base forms
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
# STEP 5: TF-IDF - use statistics to determine which words are the most important ones
# =========================

def build_tfidf(segments: pd.DataFrame, lemmatizer: Lemmatizer):
    docs = []
    lemma_maps = []

    for seg in tqdm(segments.itertuples(), total=len(segments)):
        full_text = " ".join(seg.speaker_texts.values())

        lemmas = lemmatizer.lemmatize(full_text)

        lemma_maps.append(Counter(lemmas))
        docs.append(" ".join(lemmas))

    vectorizer = TfidfVectorizer(
        min_df=1,        # viz předchozí ladění
        max_df=0.95,
        ngram_range=(1, 1)
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
    return labels


# =========================
# STEP 7: AGGREGATE TOPICS
# =========================

def summarize_topics(segments: pd.DataFrame, labels, lemma_maps):
    # --- 1️⃣ seskupení segmentů podle clusterů ---
    topics = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            topics[label].append(i)

    summaries = []

    # --- 2️⃣ zpracování každého tématu ---
    for label, idxs in topics.items():
        segs = [segments.iloc[i] for i in idxs]

        # --- čas tématu ---
        t_start = min(seg.t_start for seg in segs)
        t_end = max(seg.t_end for seg in segs)
        time_minutes = round((t_end - t_start) / 60, 1)

        # --- top lemmata ---
        lemma_counter = Counter()
        for i in idxs:
            lemma_counter.update(lemma_maps[i])

        top_lemmas = [
            lemma for lemma, _ in lemma_counter.most_common(20)
            if lemma not in UNWANTED_KEYWORDS
        ][:15]

        # --- speaker statistiky (forma) ---
        speaker_words = Counter()
        for seg in segs:
            for speaker, text in seg.speaker_texts.items():
                speaker_words[speaker] += len(text.split())

        total_words = sum(speaker_words.values())
        dominant_ratio = max(speaker_words.values()) / total_words
        speaker_count = len(speaker_words)

        # --- topic_type ---
        if dominant_ratio > 0.75 and speaker_count <= 3:
            topic_type = "monologue"
        elif speaker_count >= 3:
            topic_type = "discussion"
        else:
            topic_type = "procedural"

        # --- topic_hint ---
        topic_hint = sorted({
            DOMAIN_HINTS[l] for l in top_lemmas if l in DOMAIN_HINTS
        })
        topic_hint = ", ".join(topic_hint)

        # --- representative_text (OPRAVENÁ LOGIKA) ---
        dominant_speaker = speaker_words.most_common(1)[0][0]

        # spoj texty dominantního řečníka
        all_text = []

        for seg in segs:
            for text in seg.speaker_texts.values():
                all_text.append(text)

        all_text = " ".join(all_text)

        # rozdělení na věty
        sentences = re.split(r'(?<=[.!?])\s+', all_text)

        # 1️⃣ deduplikace
        sentences = list(dict.fromkeys(
            s.strip() for s in sentences if len(s.split()) > 8
        ))

        # 2️⃣ skórování s penalizací krátkých vět
        def score_sentence(s):
            base = sum(1 for l in top_lemmas if l in s.lower())
            length_bonus = min(len(s.split()) / 20, 1.0)
            return base * length_bonus

        def jaccard_similarity(a, b):
            A = set(a.lower().split())
            B = set(b.lower().split())
            if not A or not B:
                return 0.0
            return len(A & B) / len(A | B)

        scored = [
            (score_sentence(s), s)
            for s in sentences
            if score_sentence(s) > 0
        ]
        scored.sort(reverse=True)

        # 3️⃣ výběr s diverzitou (MMR-lite)
        representative_text = []

        # 1️⃣ povinná věta pokrývající topic_hint
        hint_sentence = find_hint_sentence(sentences, topic_hint)
        if hint_sentence:
            representative_text.append(hint_sentence)

        # 2️⃣ doplnění zbytku přes scoring + diverzitu
        for _, s in scored:
            if len(representative_text) >= 3:
                break
            if s in representative_text:
                continue
            if all(jaccard_similarity(s, prev) < 0.5 for prev in representative_text):
                representative_text.append(s)

        # --- finální objekt ---
        summaries.append({
            "topic_id": int(label),
            "segments_ids": idxs,
            "segments_count": len(idxs),
            "segments": [segment_to_json(segments.iloc[i]) for i in idxs],
            "speakers": sorted(speaker_words.keys()),
            "time_minutes": time_minutes,
            "topic_type": topic_type,
            "topic_hint": topic_hint,
            "top_lemmas": top_lemmas,
            "representative_text": representative_text
        })

    # --- seřazení dle délky ---
    return sorted(summaries, key=lambda x: -x["time_minutes"])    


def build_llm_query_payload(
    topics,
    min_minutes=2.0,
    max_topics=12,
    max_evidence_per_topic=3):
    """
    Připraví strukturovaný vstup pro LLM z výstupu summarize_topics().

    - vyřadí krátká / nevýznamná témata
    - seřadí témata podle času (významu)
    - omezí počet témat i důkazních vět
    """

    filtered = [
        t for t in topics
        if t.get("time_minutes", 0) >= min_minutes
    ]

    filtered.sort(key=lambda t: -t["time_minutes"])

    filtered = filtered[:max_topics]

    llm_topics = []

    for i, t in enumerate(filtered):
        llm_topics.append({
            "order": i + 1,
            "time_minutes": round(t["time_minutes"], 1),
            "topic_type": t["topic_type"],
            "topic_hint": t["topic_hint"],
            "evidence": t["representative_text"][:max_evidence_per_topic]
        })

    return llm_topics


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", 
                        type=Path,
                        help="Transcription which should be used. Required.")
    parser.add_argument("--outdir",
                        "-o",
                        type=Path,
                        default=Path("out"),
                        help="Output directory (default: ./out)"
                    )

    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"{args.file} is not a valid file")

    try:
        args.outdir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError("Cannot create output directory") from e
    
    print("Loading transcript...")
    df = load_transcript(args.file)

    print("Merging utterances...")
    df = merge_utterances(df)

    print("Building segments...")
    segments = build_segments(df, args.outdir)

    print("Lemmatizing + TF-IDF...")
    lemmatizer = Lemmatizer(MORPHODITA_MODEL)
    X, vectorizer, lemma_maps = build_tfidf(segments, lemmatizer)

    print("Clustering...")
    labels = cluster_segments(X)

    print("Summarizing topics...")
    topics = summarize_topics(segments, labels, lemma_maps)

    print("\n=== TOPICS ===")
    for t in topics:
        print(
            f"\nTopic {t['topic_id']} | {t['time_minutes']:.1f} min"
        )
        print(", ".join(t["top_lemmas"]))

    Path(args.outdir / "topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    llm_payload = build_llm_query_payload(
        topics,
        min_minutes=3.0,
        max_topics=10,
        max_evidence_per_topic=3
    )

    Path(args.outdir / "llm_input.json").write_text(
        json.dumps(llm_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
