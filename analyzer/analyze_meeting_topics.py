"""
Deterministic municipal meeting analyzer.

Analyzes transcripts of municipal council meetings to extract topics and
generate structured summaries. Uses a deterministic NLP pipeline rather than
relying solely on LLM interpretation.

Pipeline:
    1. Load transcript → parse timestamped speaker text
    2. Merge utterances → combine consecutive statements by same speaker
    3. Build segments → create overlapping time windows
    4. Lemmatize → reduce words to base forms (Czech language)
    5. TF-IDF → identify important words per segment
    6. Cluster → group segments into topics using HDBSCAN
    7. Summarize → extract topic metadata and representative sentences
    8. Format → prepare structured input for LLM article generation

The deterministic approach ensures the LLM receives focused, structured data
rather than making subjective decisions about what's important.
"""

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

# Model and paths
SCRIPT_DIR = Path(__file__).resolve().parent
MORPHODITA_MODEL = str(SCRIPT_DIR / "czech-morfflex2.1-pdtc2.0-250909.tagger")

# Segmentation parameters
SEGMENT_LEN = 300      # Segment length in seconds (5 minutes)
SEGMENT_OVERLAP = 120  # Overlap between segments in seconds (2 minutes)
MERGE_GAP = 5          # Maximum gap in seconds to merge consecutive utterances from same speaker

# Clustering parameters
MIN_TOPIC_SIZE = 2     # Minimum number of segments required to form a topic cluster

# Keyword filtering - common verbs and generic words to exclude from topics
UNWANTED_KEYWORDS = {
    "být", "mít", "říci", "říkat", "chtít", "moci",
    "dělat", "vědět", "myslit", "jít", "prosit", 
    "udělat", "řešit", "mluvit", "věc"
}

# Transcription error corrections - common speech-to-text mistakes specific to this locale
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

# Domain hints - map lemmas to topic categories for better topic labeling
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

# Hint keywords - used to find representative sentences that match topic hints
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
    """
    Convert timestamp string to seconds.

    Args:
        t: Time string in format "HH:MM:SS"

    Returns:
        int: Total seconds

    Example:
        parse_time("1:30:45") -> 5445
    """
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def _segment_to_json(row):
    """
    Convert a segment DataFrame row to a JSON-serializable dictionary.

    Args:
        row: pandas Series with segment data

    Returns:
        dict: Segment data with id, timing, speakers, and text information
    """
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

def _jaccard_similarity(a, b):
    """
    Calculate Jaccard similarity between two sentences.

    Jaccard similarity is the size of intersection divided by the size of union.
    Measures how similar two sentences are based on shared words.

    Args:
        a: First sentence string
        b: Second sentence string

    Returns:
        float: Similarity score between 0.0 (no overlap) and 1.0 (identical)

    Example:
        _jaccard_similarity("hello world", "hello there") -> 0.333...
        # intersection: {hello} (1 word)
        # union: {hello, world, there} (3 words)
        # score: 1/3
    """
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _find_hint_sentence(sentences, hint):
    """
    Find the first sentence that contains keywords related to a topic hint.

    Args:
        sentences: List of sentence strings to search
        hint: Topic hint string (must be a key in HINT_KEYWORDS)

    Returns:
        str or None: First matching sentence, or None if no match found
    """
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
    """
    Load and parse a transcript file into a DataFrame.

    Parses transcript with format:
        [HH:MM:SS] SPEAKER_NAME:
        Text spoken by speaker...

    Also applies common transcription error corrections from COMMON_MISTAKES.

    Args:
        path: Path to transcript file

    Returns:
        pd.DataFrame: DataFrame with columns:
            - t: timestamp in seconds (int)
            - speaker: speaker name (str)
            - text: utterance text (str)
    """
    rows = []

    time_rx = re.compile(r"\[(\d+:\d+:\d+)\]\s+(\w+):")

    current_t = None
    current_speaker = None

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # Apply common transcription error corrections
        for wrong in sorted(COMMON_MISTAKES, key=len, reverse=True):
            line = line.replace(wrong, COMMON_MISTAKES[wrong])

        # Check if line is a timestamp header
        m = time_rx.match(line)
        if m:
            current_t = parse_time(m.group(1))
            current_speaker = m.group(2)
            continue

        # Add text line to current speaker's utterance
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
    """
    Merge consecutive utterances from the same speaker.

    Combines utterances from the same speaker if they occur within MERGE_GAP seconds.
    This reduces fragmentation in the transcript.

    Args:
        df: DataFrame from load_transcript() with columns: t, speaker, text

    Returns:
        pd.DataFrame: Merged utterances with columns:
            - speaker: speaker name (str)
            - t_start: start timestamp in seconds (int)
            - t_end: end timestamp in seconds (int)
            - text: combined utterance text (str)
            - word_count: number of words (int)
    """
    merged = []

    current = None

    for row in df.itertuples():
        # Merge if same speaker and within MERGE_GAP seconds
        if (
            current
            and row.speaker == current["speaker"]
            and row.t - current["t_end"] <= MERGE_GAP
        ):
            current["text"] += " " + row.text
            current["t_end"] = row.t
        else:
            # Start new utterance
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
    """
    Divide transcript into overlapping time-based segments.

    Creates segments of SEGMENT_LEN seconds with SEGMENT_OVERLAP seconds overlap.
    This allows the clustering algorithm to better identify topics that span time windows.

    Args:
        df: DataFrame from merge_utterances() with t_start, t_end, speaker, text
        outdir: Output directory where segments.json will be saved

    Returns:
        pd.DataFrame: Segments with columns:
            - t_start: segment start time in seconds (int)
            - t_end: segment end time in seconds (int)
            - index: segment number (int)
            - speakers: list of speaker names in segment (list[str])
            - speaker_texts: dict mapping speaker to combined text (dict)
            - word_count: total words in segment (int)

    Side effects:
        Writes segments.json to outdir
    """
    t_min = df["t_start"].min()
    t_max = df["t_end"].max()

    segments = []
    t = t_min

    index = 1
    while t < t_max:
        t_end = t + SEGMENT_LEN

        # Select utterances that overlap with this time window
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

        # Advance with overlap
        t += SEGMENT_LEN - SEGMENT_OVERLAP

    # Save segments to file
    Path(outdir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return pd.DataFrame(segments)


# =========================
# STEP 4: MorphoDiTa LEMMATIZATION - NLP technique to find word base forms
# =========================

class Lemmatizer:
    """
    Czech language lemmatizer using MorphoDiTa.

    Lemmatization converts words to their base forms (e.g., "školy" -> "škola").
    This reduces noise and helps identify topics more accurately.

    Only extracts nouns (N), verbs (V), and adjectives (A) as they carry
    the most semantic meaning for topic detection.
    """

    def __init__(self, model_path):
        """
        Initialize the lemmatizer with a MorphoDiTa model.

        Args:
            model_path: Path to MorphoDiTa .tagger model file

        Raises:
            RuntimeError: If model cannot be loaded
        """
        self.tagger = morphodita.Tagger.load(model_path)
        if not self.tagger:
            raise RuntimeError("Cannot load MorphoDiTa model")

        self.tokenizer = self.tagger.newTokenizer()
        self.forms = morphodita.Forms()
        self.lemmas = morphodita.TaggedLemmas()
        self.tokens = morphodita.TokenRanges()

    def lemmatize(self, text):
        """
        Convert text to list of base-form lemmas.

        Filters to only nouns, verbs, and adjectives (based on POS tags).

        Args:
            text: Czech text to lemmatize

        Returns:
            list[str]: List of lowercase lemmas (base forms of words)

        Example:
            lemmatize("Ve školách jsou žáci") -> ["škola", "být", "žák"]
        """
        self.tokenizer.setText(text)
        lemmas_out = []

        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.tagger.tag(self.forms, self.lemmas)

            for lemma in self.lemmas:
                base = lemma.lemma.split("_")[0]
                tag = lemma.tag

                # Filter by POS (part-of-speech): N=noun, V=verb, A=adjective
                if tag[0] in {"N", "V", "A"} and base.isalpha():
                    lemmas_out.append(base.lower())

        return lemmas_out


# =========================
# STEP 5: TF-IDF - use statistics to determine which words are the most important ones
# =========================

def build_tfidf(segments: pd.DataFrame, lemmatizer: Lemmatizer):
    """
    Build TF-IDF vectors from segments for clustering.

    TF-IDF (Term Frequency-Inverse Document Frequency) identifies words that
    are important to a segment but not common across all segments.

    Process:
    1. Lemmatize all segment text
    2. Build TF-IDF matrix where each row is a segment vector
    3. Words appearing in >95% of segments are filtered (too common)

    Args:
        segments: DataFrame from build_segments()
        lemmatizer: Lemmatizer instance for text processing

    Returns:
        tuple: (X, vectorizer, lemma_maps) where:
            - X: scipy sparse matrix of TF-IDF vectors (n_segments × n_features)
            - vectorizer: fitted TfidfVectorizer for feature inspection
            - lemma_maps: list of Counter objects with lemma frequencies per segment
    """
    docs = []
    lemma_maps = []

    # Lemmatize each segment
    for seg in tqdm(segments.itertuples(), total=len(segments)):
        full_text = " ".join(seg.speaker_texts.values())

        lemmas = lemmatizer.lemmatize(full_text)

        lemma_maps.append(Counter(lemmas))
        docs.append(" ".join(lemmas))

    # Build TF-IDF vectors
    vectorizer = TfidfVectorizer(
        min_df=1,        # minimum document frequency
        max_df=0.95,     # filter words appearing in >95% of segments
        ngram_range=(1, 1)  # use single words only
    )

    X = vectorizer.fit_transform(docs)
    return X, vectorizer, lemma_maps

# =========================
# STEP 6: CLUSTERING
# =========================

def cluster_segments(X):
    """
    Cluster segments into topics using HDBSCAN.

    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
    automatically finds topics without requiring a predetermined number of clusters.
    Segments that don't fit any cluster are marked as noise (label = -1).

    Args:
        X: TF-IDF matrix from build_tfidf() (sparse matrix)

    Returns:
        np.array: Cluster labels for each segment
            - labels >= 0: topic/cluster ID
            - label = -1: noise (doesn't belong to any topic)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(X.toarray())
    return labels


# =========================
# STEP 7: AGGREGATE TOPICS
# =========================

def _extract_time_range(segs):
    """Calculate time range for a topic."""
    t_start = min(seg.t_start for seg in segs)
    t_end = max(seg.t_end for seg in segs)
    time_minutes = round((t_end - t_start) / 60, 1)
    return t_start, t_end, time_minutes


def _compute_top_lemmas(idxs, lemma_maps):
    """Extract and filter top lemmas from segments."""
    lemma_counter = Counter()
    for i in idxs:
        lemma_counter.update(lemma_maps[i])

    top_lemmas = [
        lemma for lemma, _ in lemma_counter.most_common(20)
        if lemma not in UNWANTED_KEYWORDS
    ][:15]

    return top_lemmas


def _analyze_speakers(segs):
    """Calculate speaker statistics and determine dominant speaker."""
    speaker_words = Counter()
    for seg in segs:
        for speaker, text in seg.speaker_texts.items():
            speaker_words[speaker] += len(text.split())

    total_words = sum(speaker_words.values())
    dominant_ratio = max(speaker_words.values()) / total_words if total_words > 0 else 0
    speaker_count = len(speaker_words)
    dominant_speaker = speaker_words.most_common(1)[0][0] if speaker_words else None

    return {
        "speaker_words": speaker_words,
        "total_words": total_words,
        "dominant_ratio": dominant_ratio,
        "speaker_count": speaker_count,
        "dominant_speaker": dominant_speaker
    }


def _determine_topic_type(speaker_stats):
    """Determine if topic is monologue, discussion, or procedural."""
    dominant_ratio = speaker_stats["dominant_ratio"]
    speaker_count = speaker_stats["speaker_count"]

    if dominant_ratio > 0.75 and speaker_count <= 3:
        return "monologue"
    elif speaker_count >= 3:
        return "discussion"
    else:
        return "procedural"


def _generate_topic_hint(top_lemmas):
    """Generate topic hint from top lemmas using domain hints."""
    topic_hint = sorted({
        DOMAIN_HINTS[lemma] for lemma in top_lemmas if lemma in DOMAIN_HINTS
    })
    return ", ".join(topic_hint)


def _score_sentence(sentence, top_lemmas):
    """
    Score a sentence based on topic relevance and length quality.

    The score combines two factors:
    1. Relevance: How many topic keywords (lemmas) appear in the sentence
    2. Quality: Length penalty to avoid very short sentences (< 20 words get penalized)

    Formula: relevance_score * length_quality_factor
    - relevance_score: count of top_lemmas found in sentence
    - length_quality_factor: min(word_count / 20, 1.0)
      - sentences with 20+ words get factor of 1.0 (no penalty)
      - shorter sentences get proportional penalty (e.g., 10 words → 0.5 factor)
    """
    # Count how many topic keywords appear in this sentence
    lemma_match_count = sum(1 for lemma in top_lemmas if lemma in sentence.lower())

    # Calculate length quality factor (penalize sentences shorter than 20 words)
    word_count = len(sentence.split())
    length_quality_factor = min(word_count / 20.0, 1.0)

    # Final score: relevance weighted by length quality
    score = lemma_match_count * length_quality_factor

    return score


def _extract_representative_sentences(segs, top_lemmas, topic_hint):
    """
    Extract representative sentences with scoring and diversity selection.

    Uses MMR-lite (Maximal Marginal Relevance) approach:
    1. Scores sentences by topic keyword presence and length quality
    2. Selects sentences that are both relevant and diverse
    3. Prioritizes hint sentence if available

    Ensures selected sentences aren't too similar (Jaccard < 0.5).

    Args:
        segs: List of segment rows from DataFrame
        top_lemmas: Most important lemmas for this topic
        topic_hint: Domain hint string (or empty)

    Returns:
        list[str]: Up to 3 representative sentences
    """
    # Collect all text
    all_text = []
    for seg in segs:
        for text in seg.speaker_texts.values():
            all_text.append(text)
    all_text = " ".join(all_text)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', all_text)

    # Deduplicate
    sentences = list(dict.fromkeys(
        s.strip() for s in sentences if len(s.split()) > 8
    ))

    # Score sentences
    scored = [
        (_score_sentence(s, top_lemmas), s)
        for s in sentences
        if _score_sentence(s, top_lemmas) > 0
    ]
    scored.sort(reverse=True)

    # Select with diversity (MMR-lite)
    representative_text = []

    # Add mandatory hint sentence if available
    hint_sentence = _find_hint_sentence(sentences, topic_hint)
    if hint_sentence:
        representative_text.append(hint_sentence)

    # Add remaining sentences with diversity
    for _, s in scored:
        if len(representative_text) >= 3:
            break
        if s in representative_text:
            continue
        if all(_jaccard_similarity(s, prev) < 0.5 for prev in representative_text):
            representative_text.append(s)

    return representative_text


def summarize_topics(segments: pd.DataFrame, labels, lemma_maps):
    """
    Aggregate segments into topic summaries with metadata and representative text.

    Returns a list of topic dictionaries sorted by time spent on each topic.
    """
    # Group segments by cluster labels
    topics = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            topics[label].append(i)

    summaries = []

    # Process each topic
    for label, idxs in topics.items():
        segs = [segments.iloc[i] for i in idxs]

        # Extract topic features using helper functions
        t_start, t_end, time_minutes = _extract_time_range(segs)
        top_lemmas = _compute_top_lemmas(idxs, lemma_maps)
        speaker_stats = _analyze_speakers(segs)
        topic_type = _determine_topic_type(speaker_stats)
        topic_hint = _generate_topic_hint(top_lemmas)
        representative_text = _extract_representative_sentences(segs, top_lemmas, topic_hint)

        # Build final summary object
        summaries.append({
            "topic_id": int(label),
            "segments_ids": idxs,
            "segments_count": len(idxs),
            "segments": [_segment_to_json(segments.iloc[i]) for i in idxs],
            "speakers": sorted(speaker_stats["speaker_words"].keys()),
            "time_minutes": time_minutes,
            "topic_type": topic_type,
            "topic_hint": topic_hint,
            "top_lemmas": top_lemmas,
            "representative_text": representative_text
        })

    # Sort by time spent (most to least)
    return sorted(summaries, key=lambda x: -x["time_minutes"])    


def build_llm_query_payload(
    topics,
    min_minutes=2.0,
    max_topics=12,
    max_evidence_per_topic=3):
    """
    Prepare structured input for LLM from topic summaries.

    Filters and formats topics to create concise, focused LLM input:
    1. Removes short/insignificant topics (< min_minutes)
    2. Sorts topics by time spent (importance indicator)
    3. Limits number of topics and evidence sentences

    This creates a dense, structured prompt that gives the LLM just enough
    information to write a good article without overwhelming it.

    Args:
        topics: List of topic dicts from summarize_topics()
        min_minutes: Minimum topic duration to include (default: 2.0)
        max_topics: Maximum number of topics to include (default: 12)
        max_evidence_per_topic: Max representative sentences per topic (default: 3)

    Returns:
        list[dict]: Filtered topics with structure:
            - order: ranking by importance (int)
            - time_minutes: duration spent on topic (float)
            - topic_type: "monologue", "discussion", or "procedural" (str)
            - topic_hint: domain hint for topic (str)
            - evidence: list of representative sentences (list[str])
    """

    # Filter out short topics
    filtered = [
        t for t in topics
        if t.get("time_minutes", 0) >= min_minutes
    ]

    # Sort by time spent (longer = more important)
    filtered.sort(key=lambda t: -t["time_minutes"])

    # Limit to top N topics
    filtered = filtered[:max_topics]

    # Format for LLM
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
    """
    Main entry point for the meeting analyzer.

    Runs the complete deterministic pipeline:
    1. Loads and parses transcript file
    2. Merges consecutive utterances from same speaker
    3. Divides into overlapping time segments
    4. Lemmatizes text and builds TF-IDF vectors
    5. Clusters segments into topics using HDBSCAN
    6. Extracts topic summaries with representative sentences
    7. Prepares structured LLM input

    Outputs:
        - segments.json: All time-based segments with speaker info
        - topics.json: Full topic analysis with metadata
        - llm_input.json: Filtered and formatted topics for LLM

    Command line arguments:
        --file: Path to transcript file (required)
        --outdir, -o: Output directory (default: ./out)
    """
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
