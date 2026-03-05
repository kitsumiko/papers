# Add Paper to Collection

Add a research paper to this repo's learning index and chronological list. The user provides a paper URL (e.g. an arXiv link) after the command; anything typed after `/add-paper` is the input.

## Steps

### 1. Parse the paper URL
Extract the paper URL from the user's message. If no URL is given, ask for one. Support at least:
- arXiv abstract pages: `https://arxiv.org/abs/XXXX.XXXXX`
- arXiv PDF links (normalize to abstract when editing)
- Other direct links (conference, author page) if needed

### 2. Fetch paper metadata
Use the URL to obtain:
- **Title**
- **Authors** (short form for citation, e.g. "FirstAuthor et al.")
- **Publication date** (year and month if available; for arXiv use submission date)
- **Abstract** (to decide placement)

If the URL is arXiv, you can fetch the abstract page to get this. Derive `YYYY.MM` for the chronological index.

### 3. Discover learning areas and choose placement
**Do not hardcode topic names.** Instead:

- List all `.md` files in the `learning/` directory (e.g. with a file search or glob). Exclude `glossary.md`.
- For each topic file, read the **Overview** (and optionally the first few section headers) to understand what that area covers.
- Using the paper's title and abstract, choose the **single best-fit topic file** and the **best-fit section** within it (e.g. "LLM Foundations", "Training at Scale").
- Present to the user in one short block:
  - **Paper:** [title]
  - **Proposed area:** `learning/<filename>.md`
  - **Proposed section:** [section name from that file]
  - **Date for index:** YYYY.MM

Ask the user to confirm or request a different area/section before making any edits.

### 4. Add to the topic file
Once confirmed:

- Open the chosen `learning/<topic>.md`.
- In the chosen section, find the last numbered entry (e.g. `9. [Title](url)...`).
- Append a new entry with the next number in the same format as existing entries:
  - `N. [Full Paper Title](canonical_url) (Authors, Year)`
  - `   - *Why*: One or two concise sentences on why the paper matters and what the reader learns.`
- Use the repo style: no emoji for research papers (📄 is for policy docs only). Prefer arXiv abstract URL. Follow existing "Why" style (specific, 1–2 sentences). See `CONTRIBUTING.md` and existing entries in that file for examples.

### 5. Add to chronological index
Edit `by-date.md`:

- Ensure there is a `## YYYY` heading for the paper's year if needed.
- Ensure there is a `### YYYY.MM` heading for the paper's month (e.g. `### 2026.02`).
- Insert a new bullet at the top of that month's list: `- [Full Paper Title](canonical_url) (Authors, Year)`.
- Keep months in reverse chronological order (newest first). Create a new year section at the top of the file if the paper is from a future year.

### 6. Commit
Commit the two changed files with a clear message, e.g. `Add paper: Title (YYYY)`.

## Conventions (from CONTRIBUTING.md)
- Use `###` for paper titles only where CONTRIBUTING specifies; in topic files use numbered list entries.
- Links: prefer arXiv abstract; then PDF; then proceedings.
- "Why": 1–2 sentences, specific and contextual.
- Emoji: 📄 policy docs only in topic files; research papers have no emoji there.
