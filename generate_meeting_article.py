import json
import unicodedata
from pathlib import Path
import yaml


# =========================
# CONFIG
# =========================

TOPICS_FILE = "llm_input.json"   # vstup: seznam témat
OUTPUT_MD = "meeting.md"

MEETING_DATE = "2024-09-12"
MEETING_NUMBER = 12
LAYOUT = "meeting"


# =========================
# TEXT NORMALIZATION
# =========================

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


# =========================
# METADATA
# =========================

def build_meeting_metadata(topics):
    """
    Metadata NIKDY negeneruje LLM.
    """
    total_minutes = int(sum(t["time_minutes"] for t in topics))
    hours = total_minutes // 60
    minutes = total_minutes % 60

    return {
        "layout": LAYOUT,
        "title": f"Jednání zastupitelstva – {MEETING_DATE}",
        "meeting_date": MEETING_DATE,
        "meeting_number": MEETING_NUMBER,
        "meeting_duration": f"{hours} h {minutes} min",
        "meeting_duration_minutes": total_minutes,
    }


# =========================
# TOPIC PREPARATION
# =========================

def prepare_topics_for_llm(topics):
    """
    Připraví JEDINÝ vstup pro LLM:
    - seřazený
    - omezený
    - deterministický
    """
    prepared = []

    for t in sorted(topics, key=lambda x: x["order"]):
        prepared.append({
            "order": t["order"],
            "time_minutes": round(t["time_minutes"], 1),
            "topic_type": t["topic_type"],
            "topic_hint": t.get("topic_hint"),
            "evidence": t.get("evidence", [])[:3],  # pojistka proti token overflow
        })

    return prepared


# =========================
# LLM PROMPT
# =========================

def build_llm_prompt(prepared_topics):
    """
    LLM:
    - NEVÍ o Jekyllu
    - NEVÍ o sekcích
    - NEVÍ o metadatech
    """
    return f"""
Jsi redaktor regionálního zpravodajství.
Píšeš věcný a neutrální článek o průběhu jednání zastupitelstva.

Pravidla:
- piš SOUVISLÝ TEXT, bez nadpisů a sekcí
- postupuj chronologicky podle pořadí témat
- zohledni, kolik času bylo jednotlivým tématům věnováno
- u každého tématu stručně vysvětli, čeho se týkalo
- zmiň, zda šlo o diskuzi, procedurální bod nebo vystoupení jednotlivce
- nepřidávej žádná fakta, jména ani čísla, která nejsou v podkladech
- nic nehodnoť, pouze popisuj

Podklady (seřazeno podle významu):

{json.dumps(prepared_topics, ensure_ascii=False, indent=2)}
""".strip()


# =========================
# JEKYLL OUTPUT
# =========================

def build_jekyll_page(metadata, summary_text, article_text):
    """
    Složí VALIDNÍ Jekyll page.
    """
    yaml_lines = [
        "---",
        f"layout: {metadata['layout']}",
        f"title: {metadata['title']}",
        f"meeting_date: {metadata['meeting_date']}",
        f"meeting_number: {metadata['meeting_number']}",
        f"meeting_duration: \"{metadata['meeting_duration']}\"",
        f"meeting_duration_minutes: {metadata['meeting_duration_minutes']}",
        "summary: >",
    ]

    for line in summary_text.splitlines():
        yaml_lines.append("  " + line)

    yaml_lines.append("---\n")

    return "\n".join(yaml_lines) + article_text.strip() + "\n"


def write_jekyll_draft(metadata: dict, path: str):
    header = dict(metadata)
    header["summary"] = "<<< VLOŽ SHRUTÍ (3–4 VĚTY) >>>"

    yaml_text = yaml.safe_dump(
        header,
        allow_unicode=True,
        sort_keys=False,
        width=80
    )

    content = (
        f"---\n{yaml_text}---\n\n"
        "<<< VLOŽ TEXT ČLÁNKU ZDE >>>\n"
    )

    Path(path).write_text(content, encoding="utf-8")

# =========================
# MAIN
# =========================

def main():
    topics = json.loads(Path(TOPICS_FILE).read_text(encoding="utf-8"))

    metadata = build_meeting_metadata(topics)
    prepared_topics = prepare_topics_for_llm(topics)
    llm_prompt = build_llm_prompt(prepared_topics)

    # Uložíme prompt – LLM voláš TY (lokálně / API)
    Path("llm_prompt.txt").write_text(llm_prompt, encoding="utf-8")

    print("✔ LLM prompt uložen do llm_prompt.txt")
    print("→ Pošli jej do LLM a vezmi výstup:")
    print("  - první odstavec = summary")
    print("  - celý text = article body")

    print("\n--- METADATA ---")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    write_jekyll_draft(metadata, "jekyll_draft.md")

    print("\n--- HOTOVO ---")


if __name__ == "__main__":
    main()
