# PLOS ONE submission — copy/paste reference

Everything you need to enter into the Editorial Manager submission system for
*PLOS ONE*, organized in submission-form order. Items marked **✅ READY** can be
pasted as-is; items marked **⚠️ CONFIRM** need a decision or a value only you/your
co-authors can supply.

Source of requirements: <https://journals.plos.org/plosone/s/submission-guidelines>
(verified 2026-06-17). PLOS ONE accepts/rejects on **technical soundness, valid
conclusions, reproducibility, and clear writing — not novelty or impact.**

---

## 1. Article type
**✅ READY** — `Research Article`

## 2. Title

PLOS requires **sentence case** (only the first word, proper nouns, and genus
names capitalized), max 250 characters.

- **Full title (✅ READY, paste this — sentence-cased):**
  `A stylometric application of large language models`
  - ⚠️ CONFIRM: the manuscript currently uses title case ("A Stylometric
    Application of Large Language Models"). Either is intelligible, but PLOS
    house style is sentence case — update the manuscript title page to match.
- **Short title (✅ READY, ≤100 chars):**
  `Stylometric application of large language models`

## 3. Authors & affiliations (in byline order)

From the manuscript byline. ⚠️ CONFIRM each author's **department** — PLOS
requires department + institution + city + state + country for every author. The
manuscript lists only "Dartmouth College, Hanover, NH, USA" for all authors;
only Manning's department is known from the cover-letter address.

| # | Name (First M. Last) | Email | Affiliation |
|-|-|-|-|
| 1 | Harrison F. Stropkay | harrison.f.stropkay.25@dartmouth.edu | ⚠️ [dept], Dartmouth College, Hanover, NH, USA |
| 2 | Jiayi Chen | jiayi.chen.gr@dartmouth.edu | ⚠️ [dept], Dartmouth College, Hanover, NH, USA |
| 3 | Mohammad J. Latifi | mohammad.javad.latifi.jebelli@dartmouth.edu | ⚠️ [dept], Dartmouth College, Hanover, NH, USA |
| 4 | Daniel N. Rockmore | daniel.n.rockmore@dartmouth.edu | ⚠️ [dept], Dartmouth College, Hanover, NH, USA |
| 5 | Jeremy R. Manning | jeremy.r.manning@dartmouth.edu | Department of Psychological & Brain Sciences, Dartmouth College, Hanover, NH 03755, USA |

- ⚠️ CONFIRM: byline shows "Mohammad J. Latifi" but the email implies a fuller
  name ("Mohammad Javad Latifi Jebelli"). Enter the name exactly as it should
  appear in print.

## 4. Corresponding author
**✅ READY** — Jeremy R. Manning, jeremy.r.manning@dartmouth.edu
- ⚠️ CONFIRM: corresponding author **must** supply an **ORCID iD** in their
  submission-system profile at submission time. → `[your ORCID iD]`

## 5. Author contributions (CRediT taxonomy)
**⚠️ CONFIRM** — at minimum one CRediT role per author, entered in the form.
Pick from: Conceptualization · Data curation · Formal analysis · Funding
acquisition · Investigation · Methodology · Project administration · Resources ·
Software · Supervision · Validation · Visualization · Writing – original draft ·
Writing – review & editing.

Suggested starting point (edit to reality):
- Harrison F. Stropkay — Software, Investigation, Formal analysis, Visualization, Writing – review & editing
- Jiayi Chen — Software, Investigation, Formal analysis, Writing – review & editing
- Mohammad J. Latifi — Software, Investigation, Writing – review & editing
- Daniel N. Rockmore — Conceptualization, Methodology, Supervision, Writing – review & editing
- Jeremy R. Manning — Conceptualization, Methodology, Supervision, Project administration, Writing – original draft

## 6. Abstract
**✅ READY** — 106 words (limit 300), no citations, no abbreviations beyond LLM/GPT-2. Paste:

We show that large language models (LLMs) can be used to distinguish the writings of different authors. Specifically, an individual GPT-2 model, trained from scratch on the works of one author, will predict held-out text from that author more accurately than held-out text from other authors. We suggest that, in this way, a model trained on one author's works embodies the unique writing style of that author. We first demonstrate our approach on books written by eight different (known) authors. We also use this approach to confirm R. P. Thompson's authorship of the well-studied 15th book of the *Oz* series, originally attributed to L. F. Baum.

## 7. Keywords / subject areas
**⚠️ CONFIRM** — at submission you select PLOS "Subject Areas" from their fixed
taxonomy (type-ahead). Likely matches to search for: *Natural language
processing*, *Machine learning*, *Language*, *Computational linguistics*,
*Forecasting*, *Literature*. Free-text keyword suggestions:

`stylometry; authorship attribution; large language models; GPT-2; cross-entropy;
natural language processing; digital humanities; computational linguistics`

## 8. Financial Disclosure / funding statement
Entered in the **Financial Disclosure** field (NOT in the manuscript).
- **⚠️ CONFIRM** whether any grant funded this. If none, paste the PLOS default:

  `The authors received no specific funding for this work.`

- If funded: include funder name, full grant number(s), the recipient author's
  initials, and a sentence on the funder's role (or "The funders had no role in
  study design, data collection and analysis, decision to publish, or
  preparation of the manuscript.").

## 9. Competing Interests statement
Entered in the form (NOT in the manuscript). **⚠️ CONFIRM**, default:

`The authors have declared that no competing interests exist.`

## 10. Data Availability statement
Entered in the **Additional Information** section. PLOS prefers a repository with
a **DOI / accession number**.
- **✅ READY (paste this):**

  `All code and data needed to reproduce the results in this paper are publicly
  available in the GitHub repository https://github.com/ContextLab/llm-stylometry.`

- **⚠️ STRONGLY RECOMMENDED:** mint a permanent DOI by archiving a release of the
  repo on Zenodo (GitHub → Zenodo integration), then add: "...and archived at
  Zenodo (DOI: 10.5281/zenodo.XXXXXXX)." Reviewers/editors weight a DOI snapshot
  more than a live GitHub link.

## 11. Ethics statement
**✅ READY** — no human or animal subjects (analysis of public-domain literary
texts), so no IRB/IACUC approval is required. If a form field demands it:

`This study did not involve human participants, animal subjects, or personally
identifiable data; it analyzes published, public-domain literary texts. No ethics
approval was required.`

## 12. Related-manuscript / prior-publication declaration
**✅ READY** — at submission you must confirm the work is not under consideration
elsewhere. It is not. Disclose the preprint (PLOS permits this):

`This manuscript is not under consideration for publication elsewhere. An earlier
version is posted as a preprint on arXiv (arXiv:2510.21958), consistent with PLOS
ONE's preprint policy.`

## 13. Cover letter
**✅ READY** — upload `cover_letter_plosone.pdf` (this directory; 1 page, within
the PLOS limit). It already covers: contribution summary, relation to prior work,
article type, reproducibility/data availability, and the preprint disclosure.
- **⚠️ CONFIRM (optional, go in the cover letter or the form):**
  - Suggested Academic Editors: `[2–3 names with expertise in NLP / stylometry / digital humanities]`
  - Opposed reviewers, if any: `[names + brief reason]`

## 14. Files to upload
All bundled in `plosone_submission.zip` (manuscript + figures + supplement).
| File | Role | Notes |
|-|-|-|
| `main_plos.pdf` | Manuscript | **Embedded figures removed per PLOS editorial request** — only captions remain; PLOS auto-inserts the separately-uploaded figure files into the reviewer PDF. Figureless build of `main.tex` (regenerate by compiling `main_plos.tex`). Provide `.tex` source on acceptance. |
| `cover_letter_plosone.pdf` | Cover letter | Separate upload, 1 page. |
| `Fig1.eps`–`Fig7.eps` | Figures | One file per figure, **citation order** (Fig1=loss curves, Fig2=t-stats, Fig3=confusion matrix, Fig4=MDS, Fig5=Oz losses, Fig6=tokens, Fig7=embedding). EPS vector, RGB, fonts embedded, ≤7.5″ wide, <20 MB. Cited in text as "Fig 1"…"Fig 7". Run through PACE (https://pacev2.apexcovantage.com) before upload. |
| `supplement.pdf` | Supporting Information | Self-contained; keeps its own figures (the figure-removal rule applies only to the main manuscript). ⚠️ Label items `S1 File`, `S1 Fig`, etc. (each needs "S" + number); cite each (e.g. "S1 File") in the manuscript. |

## 15. Manuscript-prep checklist before upload
- [ ] Title page in **sentence case**; short title ≤100 chars.
- [ ] Abstract ≤300 words, no citations (✅ currently 106).
- [x] Embedded figures removed from the manuscript file (done — `main_plos.pdf`); figures uploaded separately as `Fig1.eps`–`Fig7.eps`.
- [ ] Supporting Information items labeled `S1 …`, `S2 …` and cited in text.
- [ ] Data Availability statement entered in the form (and DOI minted if doing Zenodo).
- [ ] Financial Disclosure + Competing Interests entered in the form, not the manuscript.
- [ ] Corresponding author ORCID added to submission-system profile.
- [ ] CRediT contribution entered for every author.
