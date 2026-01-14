# Template Files

This directory contains legacy template files from earlier versions of the article generator.

## Current Status

**These files are NO LONGER USED** by `generate_meeting_article.py`.

The templates have been integrated directly into the Python code:

- `build_llm_prompt()` - Generates LLM prompts
- `build_jekyll_draft()` - Generates Jekyll templates
- `build_meeting_metadata()` - Generates metadata

## Legacy Files

For reference only:

- `llm-prompt.md` - Old LLM instruction template
- `meeting_template.md` - Old Jekyll structure template
- `meeting_template_metadata.md` - Old metadata template
- `meeting_template_text.md` - Old text template
- `jekyll_draft.md` - Example output (regenerated on each run)

## Safe to Delete

You can safely delete these files:

```bash
cd article-generator/
rm llm-prompt.md meeting_template.md meeting_template_*.md
```

The `jekyll_draft.md` file will be regenerated when you run the generator.

## Current Usage

To generate article materials, use:

```bash
python generate_meeting_article.py \
  --topics output/llm_input.json \
  --date 2025-01-15 \
  --number 23
```

See `README.adoc` for full documentation.
