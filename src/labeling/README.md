# Labeling (Phase B)

Turns raw ingested events (`data/events.db`'s `raw_events`, from Phase A ingestion) into
structured, labeled events in a new `classified_events` table, per the taxonomy design in
`progress/event_taxonomy.md`. Bootstrap labeling only -- rule-based and GDELT-derived labelers exist
to produce a first labeled corpus fast, and their quality is honestly spot-checked below, not
presented as ground truth. Training a real from-scratch classifier on this corpus is Phase C.

## Schema

`classified_events` (added to the existing `data/events.db`, never a separate DB file):

```sql
CREATE TABLE IF NOT EXISTS classified_events (
    event_id        TEXT NOT NULL,
    label_source    TEXT NOT NULL,   -- gdelt_derived | rule_based | llm_assisted | manual
    event_type      TEXT NOT NULL,
    severity        INTEGER NOT NULL,
    severity_score  REAL NOT NULL,
    polarity        REAL,
    countries       TEXT NOT NULL DEFAULT '[]',
    sector          TEXT,
    companies       TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL,
    reasoning       TEXT,
    labeler_version TEXT,
    labeled_at      TEXT NOT NULL,
    PRIMARY KEY (event_id, label_source),
    FOREIGN KEY (event_id) REFERENCES raw_events(id)
);
```

Primary key `(event_id, label_source)` -- the same underlying `raw_events` row can carry
independent labels from multiple labelers without collision. Full field-by-field rationale
(closed taxonomy, severity bands, entity schema) is in `progress/event_taxonomy.md`; this file is
ops-focused.

## Running it

```
./.venv/bin/python -m labeling.run_label
```

For each labeler, queries for not-yet-labeled-by-that-labeler rows first, so re-running is
idempotent -- both in storage (`INSERT OR IGNORE` on the primary key) and in *computation* (a
labeler is never even invoked on a row it already labeled). Prints a per-labeler run summary
(candidates/labeled/inserted) and a whole-corpus breakdown by `event_type`/`severity`, plus a
count of `confidence < 0.3` rows.

**Relabeling after editing a keyword dictionary or the taxonomy** requires deleting that
`label_source`'s rows first -- deliberately not automated, to avoid silently discarding prior
labels:

```sql
DELETE FROM classified_events WHERE label_source = 'rule_based';
```

## Labelers

- **`gdelt_derived`** (`labeling/gdelt_labeler.py`) -- deterministic, live, no key needed. Maps
  GDELT's QuadClass/GoldsteinScale/NumMentions onto `event_type`/`severity`/`polarity` per
  `progress/event_taxonomy.md` §3. **Hard limitation, not an implementation gap:** GDELT rows can only
  ever be labeled `geopolitical_military_tension` or `unclassified` -- GDELT's CAMEO codes
  describe actor interactions, not economic/sectoral content, so there's no principled way to
  derive the other 5 categories from them. Sector/companies are always `null`/`[]` for
  `gdelt_derived` rows (GDELT rows carry no article text), always with a `reasoning` string
  explaining why.
- **`rule_based`** (`labeling/rule_labeler.py`) -- deterministic keyword-dictionary labeler, live,
  no key needed, applied to `rss`/`newsapi` rows (title+text present). See spot-check below for
  honest quality.
- **`llm_assisted`** (`labeling/llm_labeler.py`) -- key-gated on `ANTHROPIC_API_KEY` (read via
  `ingestion.env.load_env_file()` then `os.environ`). **Not verified live in this environment** --
  no key is configured here, so the only path actually exercised is the missing-key skip (one
  warning log line, `[]` returned, zero network calls -- confirmed in both orchestrator runs
  below, `llm_labeler[rss]`/`llm_labeler[newsapi]` both show `skipped (ANTHROPIC_API_KEY not
  set)`). Implementation notes:
  - No `claude-api` skill was found installed in this environment (checked `~/.claude/skills` and
    the installed plugin marketplaces). Instead, current model IDs and the structured-output
    pattern were confirmed two ways: (1) fetching Anthropic's public
    `anthropics/skills` repo (`skills/claude-api/shared/models.md`,
    `skills/claude-api/SKILL.md`) directly from GitHub, and (2) inspecting the actually-installed
    `anthropic` Python SDK (v0.117.0, added to `requirements.txt`) in this venv to confirm the API
    shape (`client.messages.parse(..., output_format=<pydantic model>)`,
    `response.parsed_output`).
  - **Deliberate, documented deviation from the fetched skill's default model guidance:** the
    skill's general guidance defaults to `claude-opus-4-8` for assistant use. This labeler instead
    defaults `MODEL_ID` to `claude-haiku-4-5`, because it's a cost-sensitive, high-volume *batch*
    classification job (batch size explicitly capped at `MAX_EVENTS_PER_RUN = 50` per run for
    cost/rate control), not an interactive session. `MODEL_ID` is overridable via the
    `ANTHROPIC_LABELER_MODEL` env var if a reviewer disagrees with this tradeoff.
  - Structured output is enforced via `output_format=<pydantic model>` (the closed taxonomy's
    fields, with `Literal[EVENT_TYPES]` and numeric bounds), not manual JSON-string parsing.
  - Since this path is untested against the live API, no retry/backoff logic is implemented --
    flagged as a known gap for whenever a key becomes available and this is actually exercised.
- **`ollama_assisted`** (`labeling/ollama_labeler.py`) -- added 2026-08-15 for the GDELT
  `geopolitical_military_tension` content re-check (`progress/event_taxonomy.md` "Flagged ambiguity
  8"). Same shape/prompt discipline as `llm_assisted`, but talks to a **local Ollama server**
  (`qwen2.5:3b-instruct` by default) over plain HTTP instead of the Anthropic API -- free, no key,
  per explicit user direction for this pass. Not part of `labeling/run_label.py`'s normal
  orchestration; only invoked by `labeling/run_gdelt_reclassify.py` as Stage 2, for whatever Stage
  1 (`labeling/gdelt_content_filter.py`, a cited keyword filter reusing `rule_labeler.py`'s own
  `geopolitical_military_tension` keyword list) can't confidently confirm/reject. Gated on the
  Ollama server actually being reachable and the model actually being pulled
  (`ollama_labeler.is_available()`) -- skips cleanly (`[]`, logged warning, no network calls) if
  not, same graceful-degradation discipline as the `ANTHROPIC_API_KEY` gate above.
  - **Local install used in this environment** (no root available -- `sudo` requires a password
    that can't be supplied non-interactively here): downloaded Ollama's official Linux release
    tarball directly (`https://github.com/ollama/ollama/releases/download/v0.32.13/ollama-linux-amd64.tar.zst`)
    and extracted it to `~/.local/ollama` rather than using the installer script's
    system-wide/`sudo`-requiring path. Run as a plain user process: `~/.local/ollama/bin/ollama
    serve` with `OLLAMA_MODELS=~/.local/ollama/models`, `OLLAMA_NUM_PARALLEL=1`. Model pulled via
    `ollama pull qwen2.5:3b-instruct`.
  - **Why `qwen2.5:3b-instruct`, not a smaller/faster model:** benchmarked `qwen2.5:1.5b-instruct`
    first (~2-3s/call vs. 3b's ~5.7s/call) but it was unreliable enough to reject outright --
    misclassified a plain car-crash headline as `geopolitical_military_tension`, and returned
    `confidence` on an inconsistent 0-1-vs-0-100 scale across calls (10, 0.8, 0.97, 0.05, 100 seen
    across 5 near-identical requests). The 3b model was consistent and correct across the same
    spot checks. `ollama_labeler.py::_clip()` still defensively rescales an out-of-range
    confidence/severity_score rather than trusting the model's stated scale, since even the 3b
    model isn't guaranteed immune to this.
  - **Why sequential, not concurrent, requests:** benchmarked 6 concurrent `/api/chat` calls
    (`OLLAMA_NUM_PARALLEL=6`) against 6 sequential ones on this machine's 16 CPU cores (no GPU) --
    concurrent finished in ~28s vs. sequential's ~34s, a small gain, not a multiplier, because
    inference here is CPU-compute-bound, not I/O-bound: concurrent requests contend for the same
    cores rather than overlapping idle time. Run sequentially (`OLLAMA_NUM_PARALLEL=1`) instead,
    accepting a real multi-hour runtime for the full uncertain-bucket batch (~3,200 rows at
    ~6s/call ≈ 5+ hours) rather than adding parallelism complexity for a ~20% gain.

## Verification performed (2026-07-19)

Ran `./.venv/bin/python -m labeling.run_label` against the live `data/events.db` (89 `gdelt` rows,
450 `rss` rows, 0 `newsapi` rows):

| run | gdelt_labeler | rule_labeler[rss] | rule_labeler[newsapi] | llm_labeler |
|---|---|---|---|---|
| 1st | 89 candidates / 89 labeled / 89 inserted | 450 / 450 / 450 | 0 / 0 / 0 | skipped, no key |
| 2nd (immediately after) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | skipped, no key |

Second run confirms idempotency: zero candidates found by the anti-join query (no
computation performed), zero rows inserted, for every labeler -- since no new `raw_events` rows
were ingested between the two runs (ingestion wasn't re-run here), this is the expected
zero-delta case, same caveat Phase A's own idempotency check documents for RSS (a genuinely new
article arriving between runs would show up as a nonzero delta, not a bug).

**Note on these two runs:** they were run *after* a mid-implementation bug found by the spot-check
below was fixed (see "Bug found and fixed" section) -- the numbers above are from the corrected
`rule_labeler`, not the first, buggier version. This is disclosed rather than silently rerun and
overwritten without a trace.

Final corpus state after both runs: 539 total `classified_events` rows (89 `gdelt_derived` + 450
`rule_based`; 0 `llm_assisted`, key-gated).

By `event_type` (whole corpus): `unclassified` 435, `corporate_strategic` 44,
`geopolitical_military_tension` 40, `macroeconomic_conditions` 8, `regulatory_subsidy_action` 4,
`trade_export_control` 4, `natural_disaster` 3, `supply_chain_fab_disruption` 1.

By `severity`: 1 (negligible) 394, 2 (minor) 105, 3 (moderate) 16, 4 (major) 11, 5 (severe) 13.

`confidence < 0.3`: 435 rows (all of them the `unclassified`-with-confidence-0 rows -- i.e. every
row the rule-labeler declined to guess on is, correctly, also flagged low-confidence; there is no
row where the labeler was overconfident about an `unclassified` call).

## Honest spot-check (rule_based, rss/newsapi rows)

Per the persona rule that bootstrap labels are a labeling aid, not ground truth, this section
reports an actual hand assessment, not an estimate.

### Bug found and fixed during the spot-check

The **first** spot-check sample (20 random `rule_based` rows, before any fix) surfaced a genuine
implementation bug, not a taxonomy ambiguity: the keyword matcher used plain Python substring
checks (`keyword in text`), which match *inside* unrelated words. Two concrete examples actually
observed:

- `"Linear Pumps Market Forecast Points Higher **Toward** 2035, Driven by Semiconductor and
  Automation Capex Cycles"` -- the severity-5 keyword `"war"` matched inside `"To**war**d"`,
  producing `event_type=geopolitical_military_tension, severity=5` for a routine industrial market
  forecast. Wrong on both event_type and severity.
- `"Democratic socialists top MAGA candidates... following the 2025 election of New York City
  Mayor Zohran **Mamdani**"` -- the `AMD` ticker alias matched inside `"Mam**dan**i"` wait --
  inside `"M**amd**ani"`, incorrectly tagging `companies=["AMD"]` on an article with no connection
  to AMD or semiconductors at all.

**Fix applied:** `labeling/rule_labeler.py`'s keyword matching was changed from bare substring
checks to boundary-aware regex matching (`(?<!\w)keyword(?!\w)`, not a plain `\bkeyword\b` --
`\b` alone breaks on punctuation-containing aliases like `"u.s."`, verified before committing to
the fix). All `rule_based` rows were deleted and relabeled with the fixed matcher before the
final orchestrator runs and the final spot-check below. This is disclosed as a real
mid-implementation correction, not silently absorbed into the "final" numbers with no trace.

### Final spot-check (25 random rows, post-fix)

Concrete side-by-side examples (title vs. assigned label), hand-assessed:

**Good:**

| Title | Assigned |
|---|---|
| "Trump threatens higher Canada tariffs over fires" | `trade_export_control`, severity 3 (proposed/threatened, not enacted) -- correct category and correctly non-enacted severity |
| "John G Ullman & Associates Inc. Sells 6,289 Shares of Taiwan Semiconductor Manufacturing Company Ltd. $TSM" | `unclassified`, but entities correctly extracted: `sector=semiconductors, companies=[TSM], countries=[Taiwan]` -- correctly recognized as a routine holdings disclosure, not a real event |
| "These World Cup tweets show how AI will unite **us** all" | `countries=[]` (correctly did *not* fire on the lowercase pronoun "us" -- confirms the case-sensitive bare-`US` alias design works as intended) |
| "Should You Invest in the VanEck Semiconductor ETF (SMH)?" | `unclassified`, `sector=semiconductors` -- correct: sector-relevant but no concrete event, matches the "negligible/routine commentary" anchor |

**Bad / genuinely wrong:**

| Title | Assigned | Problem |
|---|---|---|
| "Bank of America revamps Tesla forecast before earnings" | `countries=["United States"]` | `"America"` matched as a literal whole word *inside the company name* "Bank of America" -- not a substring bug (word-boundary regex is working as designed here), but a real precision limitation: naive keyword aliasing can't tell "America" the country from "America" inside a proper noun. Same pattern recurred on a second row ("Bank of America sends strong verdict on Microsoft stock"). |
| "Forest City's Network School says **US**$122m investment plan on hold amid Malaysia's probe..." | `countries=["United States"]` | The `US` alias matched the currency prefix "US$", not an actual mention of the United States as a geopolitical actor -- article is about Malaysia/Israel, not the US. |

**Recall gaps (defensible, not fabrications):**

| Title | Assigned | Note |
|---|---|---|
| "Chip worker shortage puts U.S. semiconductor boom on the brink" | `unclassified` | Arguably `supply_chain_fab_disruption`-relevant, but the dictionary only has the exact phrase `"chip shortage"`, not `"chip worker shortage"` -- a real coverage gap from exact-phrase matching, not a wrong guess. |
| "Taipei reassesses ties with PNG after office closure" | `unclassified` (though `countries=[Taiwan]` correctly detected) | A genuine diplomatic-relations story that arguably belongs in `geopolitical_military_tension`, but that category's keyword list skews toward hard military terms (missile/troops/invasion), not soft-diplomacy language. |

**Weak-default-band-2 in evidence** (the ambiguity flagged in `progress/event_taxonomy.md` §4, now
concretely observed, not just predicted): several rows with a real event_type match but no
severity keyword hit got the default severity 2, even where band 1 ("analyst commentary") would
arguably fit better -- e.g. `"Analysis: Fed Chairman Warsh faces an inflation credibility test..."`
→ `macroeconomic_conditions`, severity 2 (the leading `"Analysis:"` is exactly the kind of
band-1 signal the rubric describes, but isn't in the band-1 keyword list).

**Macroeconomic over-absorption** (ambiguity 3, also concretely observed): `"Mortgage and
refinance interest rates today, Friday, July 17, 2026: Rates are mixed today"` matched
`macroeconomic_conditions` on `"interest rates"` -- technically on-topic, but this is a recurring
daily rates almanac post, not really a discrete "event" in the sense the taxonomy intends. A real,
predicted-in-advance failure mode, now confirmed in practice.

### Estimated quality (rule_based, post-fix)

Of 25 hand-checked rows: **18/25 (72%) fully correct** across event_type, severity, and entities;
**3/25 (12%)** had a correct (or defensibly-unclassified) event_type/severity but one incorrect
entity tag (the "America"/"US$" false-positive pattern above); **2/25 (8%)** were defensible
recall misses (`unclassified`, no fabrication, just missed real signal); **2/25 (8%)** showed the
documented weak-default-band-2 or macro-over-absorption pattern on an otherwise-correct category
call. **Zero fabricated categories** were observed in the post-fix sample (every wrong call was
either an entity-extraction imprecision or a conservative `unclassified`, never a confidently
wrong event_type on a truly irrelevant article). This is consistent with a first-pass bootstrap
labeler: good enough to seed a corpus and prioritize `unclassified`/low-confidence rows for
review, not good enough to treat as ground truth -- exactly the caveat the persona rules require.

## GDELT `geopolitical_military_tension` content re-check (2026-08-15)

Per `progress/event_taxonomy.md` "Flagged ambiguity 8": once real headlines became available for
GDELT-derived cluster-representative events (`ingestion/article_titles.py`'s scrape enrichment),
two live examples surfaced that got labeled `geopolitical_military_tension` with severity 0.9257
purely from `QuadClass=4` -- a domestic murder-suicide story and a press release about a legal
appeal to protect an endangered lizard species. Neither has anything to do with geopolitics or the
military. `labeling/gdelt_content_filter.py` (Stage 1) + `labeling/ollama_labeler.py` (Stage 2,
local model) + `labeling/run_gdelt_reclassify.py` (orchestration) were built to re-check every
`gdelt_derived geopolitical_military_tension` cluster-representative event against its real
headline. See the "Labelers" section above for the local-Ollama setup and model choice.

### Bug found and fixed during Stage 1 development

Read-only prototyping against the live corpus's 4,378 titled candidates (before any code was
committed) validated the two-stage split looked reasonable, but the **first committed version** of
`gdelt_content_filter.py::classify_title` used plain Python substring checks (`keyword in
title.lower()`) -- the exact same bug class `rule_labeler.py` already found and fixed once (see
"Bug found and fixed" above: `"war"` matching inside `"Toward"`). `GEO_KEYWORDS` inherits `"war"`
directly from `rule_labeler.py`'s own list, so the same collision reappeared here: `"war"` matches
inside `"toward"`, `"software"`, `"warning"`, `"warehouse"`, `"warplanes"`, `"award"`, and similar
words with no geopolitical content. This was caught by an already-run interactive test batch (10
rows through Stage 2, 888 more resolved by Stage 1 alone) *before* the full multi-hour background
run was launched -- but the full run had already been started against the buggy version by the
time the bug was noticed.

**Fix applied:** switched to `labeling.rule_labeler._contains_keyword` (the same word-boundary-aware
`(?<!\w)keyword(?!\w)` regex matcher `rule_labeler.py` already uses), rather than reintroducing a
bug that module had already found and fixed. **Verified impact before trusting the fix:**
re-classified all 898 already-"confirmed" titles under the corrected matcher -- **221/898 (24.6%)**
changed verdict (from a false "confirm" to the correct, more conservative "uncertain"), including
genuinely geopolitical headlines like "27 Chinese warplanes cross into Taiwan's Air Defense Zone"
that had only spuriously matched via the bare `"war"` substring inside `"warplanes"`, not a real
keyword hit (`"warplane"` was not itself in the list). Those 898 rows' `gdelt_content_checked`
entries were cleared (no `classified_events` data needed reverting -- a Stage-1 "confirm" verdict
never writes anything, it only marks a row as not needing further review) so a subsequent run
re-evaluates them with the fixed matcher. The already-running background batch was left
undisturbed, since it was processing a *different* row set (rows already past Stage 1 into Stage 2)
using an already-fresh, already-fixed process (each run re-imports the current module state).

### Estimated quality

This was not one clean run -- ten rounds, spanning multiple bugs found and fixed mid-flight
(including one mistaken "we're done" claim by the person doing this work, corrected below rather
than left standing), each one disclosed here rather than silently absorbed into a "final" number
with no trace, same discipline as the `rule_based` spot-check above.

**Cumulative final tally (live query against `gdelt_content_checked`, all 10 rounds combined):**

| verdict | count |
|---|---|
| `confirm_stage1` | 1,048 |
| `confirm_stage2` | 598 |
| `reject_stage1` | 689 |
| `reject_stage2` | 8,839 |
| **total checked** | **11,174** |

**9,528/11,174 (85.3%) of re-checked candidates flipped from `geopolitical_military_tension` to
`unclassified`.** That is a genuinely large false-positive rate in the original QuadClass-only
label, not a rounding artifact -- consistent with CAMEO's "Fight" root code covering ordinary
crime/accidents as readily as real conflict (ambiguity 8).

**Known, disclosed residual: 123 titled cluster-representative candidates were never checked.**
Each round's deletions promote a new, smaller batch of cluster winners (the bucket that used to be
won by a just-deleted false positive gets re-won by whatever's next-highest-`K`), so this is an
open-ended cascade, not a fixed-size job -- see the round-by-round history below. The correction
loop was capped at 10 rounds and stopped there deliberately: round-start candidate counts shrank
from 2,301 (round 2) down to 157 (round 10), each round still costing 30-90 minutes end to end
(dominated by Stage 2's local-LLM calls), for a shrinking and increasingly-marginal return against
an already-85%+-corrected corpus. Continuing to a literal zero was judged not worth the wall-clock
cost. **This residual is a real, live gap, not a hypothetical one** -- a final spot-check of the
current `event_catalog` (see below) found actual false positives still sitting in it, e.g.
`"Adultery Is Still A Felony In 16 States Including Massachusetts"` (severity 0.60) and `"New World
Screwworm cases rise to 32 as Texas battles outbreak"` (severity **1.00**, the maximum band) --
both confirmed still `checked=NULL` in `gdelt_content_checked`, i.e. part of the disclosed 123, not
a new bug. Re-running `labeling.run_gdelt_reclassify` + `flux_engine.run_timeseries` a few more
times would continue shrinking this count; it just wasn't carried further in this pass.

**Round history (each round is the newly-promoted cluster-representative set surfaced after the
previous round's deletions let a different, lower-`K` event win its bucket):**

| round | candidates | stage1 confirm | stage1 reject | stage2 sent | stage2 confirm | stage2 reject |
|---|---|---|---|---|---|---|
| 1 | ~3,192 | -- | -- | -- | -- | -- |
| 1.5 | 898 | -- | -- | -- | -- | -- (221 flipped confirm->uncertain on re-check) |
| 1.8/1.9 | 153, then 127 | -- | -- | 127 | 22 | 105 |
| 2 | 2,301 | 223 | 133 | 1,945 | 63 | 1,882 |
| 3 | 1,462 | 112 | 73 | 1,277 | 33 | 1,244 |
| 4 | 966 | 73 | 65 | 828 | 20 | 808 |
| 5 | 650 | 48 | 44 | 558 | 12 | 546 |
| 6 | 477 | 28 | 41 | 408 | 3 | 405 |
| 7 | 340 | 16 | 25 | 299 | 8 | 291 |
| 8 | 253 | 9 | 24 | 220 | 6 | 214 |
| 9 | 193 | 13 | 17 | 163 | 2 | 161 |
| 10 | 157 | 5 | 15 | 137 | 2 | 135 |

(Rounds 1/1.5/1.8/1.9 predate the `gdelt_content_checked` verdict-tallying discipline being fully
wired up per-round -- their per-stage breakdown wasn't captured as cleanly as rounds 2-10's; the
**cumulative live-queried tally above** is the authoritative number, not a hand-sum of this table.)

- **Round 1** (~3,192 titled candidates) -- first full pass. Caught the two motivating examples
  (a Florida murder-suicide, a lizard-conservation legal appeal) plus many more of the same shape.
- **Round 1.5** (898 rows) -- re-run after the Stage 1 substring-matching bug fix (see below); 221
  of these flipped verdict once the matcher was fixed.
- **Round "1.8/1.9"** (153, then 127 rows) -- re-run after two more bugs (overly-generic Stage 1
  keywords, then the "when in doubt" bias) were found and fixed; example corrections from the final
  127-row batch: `"North Korea fires salvo of short-range ballistic missiles"`,
  `"Kim Unveils New North Korea 'Suicide Drones'"`, `"North Korea tests rocket launcher in threat to
  Seoul"` -- all had been wrongly rejected by the biased prompt, all correctly confirmed once it was
  removed.
- **Round 2** (2,301 candidates, the promotion cascade from round 1's ~75% deletion rate) -- 87.6%
  of the round flipped to `unclassified`. Spot-checked corrections were unambiguous: car crashes,
  DUI arrests, phone-review headlines, crypto scams, murder-suicides -- e.g. `"Salisbury Man Killed
  in Bridgeville Head-On Crash"`, `"'Poor driving' caused car crash near Seoul City Hall that killed
  9: police"`, `"Louisiana high school student shot, killed in apparent murder-suicide"`.
- **Rounds 3-10** (1,462 down to 157 candidates, correction rate 90-96% every round) -- the same
  cascade continuing to shrink. A Stage 1 keyword bug was found and fixed mid-cascade (bare
  `"troop"`/`"troops"` wrongly auto-confirming domestic National Guard incidents -- see bug 7
  below), and a verification mistake (the false "0 titled candidates remain" claim -- bug 8 below)
  was caught and corrected rather than left standing. Typical round-10-era corrections: `"Three dead
  in Texas parking lot shooting, man arrested"`, `"2 found dead in Lawrence after apparent
  murder-suicide, DA says"`, `"North Korea Carrying Out Global Espionage To Steal Military Secrets:
  US And Allies"` (this last one flipped to `unclassified` by Stage 2 -- a defensible call, since
  espionage/intelligence activity is a different mechanism than the military/security-posturing
  definition in the Stage 2 prompt, not an obvious miss, but flagged here as a borderline case worth
  a human's judgment rather than silently endorsed).

**Hand spot-check, Stage 2 confirms** (10 random `ollama_assisted geopolitical_military_tension`
rows still in the current `event_catalog`, `random.seed(42)`): **10/10 genuinely geopolitical** --
e.g. `"One killed, two wounded by Russian strikes in Ukraine's Zaporizhzhia region, governor says"`,
`"First batch of M1A2 tanks arrive in Taipei"`, `"Elite US unit training against China"`,
`"South Korean prosecutor seeks death penalty for ex-President Yoon over martial law declaration"`.
No false positives observed in this sample.

**Hand spot-check, Stage 2 rejects** (15 random `reject_stage2` rows): **zero clear false
negatives.** A few borderline-defensible calls -- `"US Launches 'Deadly Strike' In Nigeria: Trump"`
(real strike, but Nigeria isn't a tracked corridor country, so `unclassified` is arguably correct
under this prompt's explicit corridor scoping, not a miss); `"Working on safe passage of more
Indian vessels through Strait of Hormuz: Iranian minister"` and `"Russia Pressing Ahead On Rail Link
To Iran & Azerbaijan..."` (tangential to the energy corridor but about infrastructure/shipping
cooperation, not conflict risk -- defensibly `unclassified`). Everything else in the sample was an
unambiguous non-event (a bridge collapse, a gas explosion, a missing-persons case, a car review, a
wedding story) correctly rejected.

**Final live-catalog spot-check (top-severity and most-recent `geopolitical_military_tension`
`event_catalog` rows, post-round-10):** overwhelmingly real conflict content (`"Mission in Iran war
has pivoted toward ensuring oil flow, Trump official says"`, `"More strikes as Iran warns of
'existential war' with US"`, `"US resumes naval blockade on Iranian ports..."`, `"Houthi leader
threatens Saudi oil facilities if Riyadh escalates in Yemen"`), plus a handful of rows confirmed
part of the disclosed 123-row residual (the Adultery/Screwworm examples above). No *new*,
previously-undisclosed false positives were found outside that known residual.

#### Bugs found and fixed during this pass

1. **Stage 1 substring-matching bug** (same bug class `rule_labeler.py` already fixed once --
   `"war"` matching inside `"toward"`/`"warplanes"`/etc.) -- inherited via `GEO_KEYWORDS` reusing
   `rule_labeler.py`'s raw keyword list. Fixed by switching to the same word-boundary-aware
   `_contains_keyword` regex. Verified impact: 221/898 already-"confirmed" rows changed verdict once
   fixed (see "Bug found and fixed during Stage 1 development" above for the full account) -- but
   this fix was initially validated only against the `confirm_stage1` bucket. A later spot-check of
   the *reject* bucket found the same gap had never been re-checked there: 27/279 `reject_stage1`
   rows didn't reproduce under the corrected matcher (e.g. `"war"` no longer spuriously flagging
   `"preview"`-containing titles as false confirms doesn't help a row that was *rejected* via a
   different stale substring). Those 27 rows' original `gdelt_labeler.py` output was regenerated
   deterministically from `raw_metadata` (confirmed pure/side-effect-free before trusting this),
   reinserted, and reprocessed through the full corrected pipeline -- all 27 resolved to
   `unclassified` on reprocessing, confirming the stale verdicts had been directionally correct by
   coincidence and no real harm had reached `flux_score`.
2. **Stage 2 `natural_disaster` dumping ground** -- see `labeling/ollama_labeler.py`'s module
   docstring for the full account. Two attempts at letting the 3B local model choose among the full
   8-category taxonomy both produced unreliable results (108/370 headlines in one test batch got
   `natural_disaster`, including plain car crashes, a bridge collapse, and a shooting, sometimes with
   self-contradicting `reasoning` text). Fixed by constraining `event_type` to a binary
   `{geopolitical_military_tension, unclassified}` -- removing the failure mode by construction
   rather than a third prompt-wording attempt. Verified via full reprocessing of the affected rows
   (146/146 correctly resolved to `unclassified`) and an 8-title A/B test (7/8 correct) before the
   full rerun.
3. **Overly generic Stage 1 keywords** -- bare `"military"` and `"north korea"` in `GEO_KEYWORDS`
   were auto-confirming headlines with zero conflict content (a North Korea tourism-reopening story,
   a North Korea heatwave story). Quantified: 153/677 `confirm_stage1` rows matched *only* via these
   two terms. Fixed by excluding them from `labeling/gdelt_content_filter.py`'s `GEO_KEYWORDS`
   (`rule_labeler.py` itself left untouched at the time, on the assumption its full-article-text use
   case dilutes the generic hit enough to be safe -- **that assumption turned out to be wrong for
   `"military"` specifically; see bug 9 below**, found later via a live OSINT-feed spot-check).
4. **"When in doubt, choose unclassified" bias overcorrecting** -- a prompt phrase added while Stage
   2 still had the full taxonomy (to stop it guessing a wrong specific category) started causing
   false negatives once `event_type` became binary: unambiguous headlines like `"North Korea fires
   salvo of short-range ballistic missiles"` were rejected, with the model's own `reasoning`
   self-contradicting ("about missile launches... but does not specify... conflict risk"). A/B
   tested removing the phrase against both the new false negatives and the original false-positive
   titles (car crash, oil price, tourism reopening): recovered some false negatives with zero
   regression on the false positives, so it was removed. **Known remaining limitation, disclosed
   rather than chased further:** the model still misses some unambiguous military headlines that
   don't explicitly name a tracked corridor/adversary relationship. False negatives are the safe
   failure direction here (a missed event just doesn't compete for its cluster, vs. a false positive
   injecting wrong signal), so this was judged a real 3B-model capability ceiling worth documenting,
   not worth a fourth prompt iteration for diminishing returns.
5. **Self-caught UTC-vs-local-time SQL mistake** -- while clearing `gdelt_content_checked` rows to
   force reprocessing of the 127 rows affected by bug 4, the filter used a local-time string
   (`checked_at >= '2026-08-15T18:24'`) against a column that stores true UTC timestamps
   (`datetime.now(timezone.utc).isoformat()`). This matched 3,166 rows instead of the intended ~127.
   Caught immediately by a sanity count mismatch, verified that only the tracking table (never
   `classified_events`) had been touched, then reconstructed the exact intended 127-row set
   deterministically (independent of DB state, from the old-vs-new keyword classification masks) and
   restored the 3,039 wrongly-cleared rows exactly, cross-checked arithmetically at each step. No
   data was actually lost, but disclosed in full per this project's standing rule that a caught
   mistake still gets written down, not just silently corrected.
6. **`event_catalog` staleness (pre-existing, unrelated to this GDELT work, found as a byproduct)**
   -- `flux_engine/timeseries.py::store_event_catalog()` used a bare `INSERT OR REPLACE` keyed on
   `event_id` with no deletion step, so a cluster bucket's old (no-longer-winning) representative
   from a prior run was never removed when a different event won that bucket on a later run. Found
   because round 2's candidate count looked implausible: `event_catalog` held 8,292 rows (two
   distinct `computed_at` generations, 3,429 stale + 4,863 fresh) against a run that had only just
   stored 4,863. Fixed by adding a `since`-scoped `DELETE FROM event_catalog WHERE published_at >=
   ?` before insert in `store_event_catalog()` (`flux_engine/run_timeseries.py` already always does
   a full historical rebuild from a fixed start date, so this is a safe, complete rebuild-per-window,
   not a partial one). The existing 3,429 stale rows were manually cleaned from the live DB. This bug
   predates this session's GDELT work and would have kept silently accumulating stale catalog rows on
   every future `run_timeseries` run if left unfixed -- worth fixing at the source rather than just
   noting it.
7. **Overly generic Stage 1 keyword, round 2 (bare `"troop"`/`"troops"`)** -- same failure shape as
   bug 3, found later, in the round-3-through-10 cascade: `"National Guard troops fatally shoot a
   man in downtown Memphis"` was wrongly Stage-1-confirmed on the bare `"troops"` hit. Quantified
   *before* fixing, matching this project's standing discipline: of 47 `confirm_stage1` rows that
   matched *only* via `"troop"`/`"troops"`, 46 were genuinely geopolitical (Russia/Ukraine and North
   Korea troop-deployment headlines) and only 3 were domestic National Guard incidents -- a ~2%
   false-positive rate, nowhere near bug 3's 22.6%, so `"troop"`/`"troops"` was **not** blanket-
   excluded like `"military"`/`"north korea"` were (that would have thrown away 46 good signals to
   fix 3 bad ones). Instead, `"national guard"` was added to `REJECT_KEYWORDS`: since
   `classify_title` treats "both lists hit" as `uncertain` (not a tie-break), this only downgrades a
   headline that says *both* "troops" and "National Guard" to a real Stage 2 read, leaving every
   other `"troops"` hit untouched. Verified: all 3 live "national guard" matches in the corpus were
   wrongly `confirm_stage1`; all 3 were cleared and correctly resolved to `unclassified` on re-check.
8. **Self-caught verification mistake: a false "0 titled candidates remain" claim.** After round 2's
   rebuild, a completion check queried `raw_events.title IS NOT NULL` to determine whether any
   titled cluster-representative candidates were left unprocessed, got 0, and this README briefly
   stated the correction cascade had reached its "natural end." That was wrong: GDELT rows'
   `raw_events.title` is **always NULL** (see `ingestion/gdelt.py` -- GDELT carries no article text
   at ingestion time; the real scraped title lives in `event_catalog.title`, populated later by
   `enrich_event_catalog_urls()`). The correct query (against `event_catalog.title`, matching
   `run_gdelt_reclassify.py::_candidate_rows()`'s own selection logic exactly) found **1,459
   unchecked titled candidates**, not zero -- newly-promoted cluster winners after round 2's mass
   deletions, sitting live in `event_catalog` including actual false positives (`"Judge denied Mayor
   Scott's motion to dismiss IG's records access lawsuit"`, `"Samsung layoffs: Over 800 US workers
   affected..."`, `"Trump admin narrows Endangered Species Act protections"` -- notably the same
   *shape* of false positive as the original lizard-appeal motivating example -- and a Nat Geo shark
   documentary headline, all sitting at `severity_score=1.00`, the maximum band, still driving real
   `flux_score` signal). Caught by directly re-deriving the true candidate count with the correct
   column before declaring the work done, rather than trusting the first "0 remaining" result. This
   triggered rounds 3 through 10 (see the round history above) and is disclosed here in full rather
   than quietly editing the earlier wrong claim away -- the mistake, not just its fix, is the
   record.
9. **Bare `"military"` in `rule_labeler.py`'s `EVENT_TYPE_KEYWORDS` (2026-08-16, found via a live
   OSINT-feed spot-check, not the GDELT cascade above)** -- a user reviewing the live feed flagged
   *"BTS bump? Hong Kong's hotels set for bonanza from superstars' Arirang world tour"* as wrongly
   `geopolitical_military_tension`. Root cause: the article text quotes the group's return "since
   completing mandatory military training" (South Korean conscription) -- nothing to do with
   conflict, but the bare `"military"` keyword hit was enough to win the category via
   `_match_event_type()`'s hit-count comparison. This is `rule_labeler.py`'s own keyword list (used
   directly for `rss`/`newsapi` rows, `label_source="rule_based"`), a completely different code path
   from the GDELT pipeline documented above -- and one with **no Stage-2-equivalent LLM safety net at
   all**, so a bad keyword here has no second check to catch it. Bug 3 (above) had already found and
   fixed this exact failure shape for GDELT headlines, but assumed `rule_labeler.py`'s full-article-
   text context would dilute a bare `"military"` hit enough to be safe; this is a live counter-example
   to that assumption (title+text was the haystack here too). Quantified before fixing, matching this
   project's standing discipline: of 26 `rule_based`/`geopolitical_military_tension` rows in the live
   corpus, 5 matched `"military"` -- 2 legitimately (paired with `"drills"`, genuine Pyongyang/
   Cambodia military-drills stories, unaffected by this fix since `"drills"` alone still wins), and in
   the 3 where `"military"` was the *sole* matched keyword, **all 3 (100%) were false positives**: the
   BTS story, an unrelated HELOC personal-finance article quoting "military service in Afghanistan" in
   a burn-pit health-claim aside, and a WWII historical retrospective -- a higher false-positive rate
   than bug 3's 22.6%. `"north korea"` was deliberately left untouched in `rule_labeler.py`: it has
   *zero* live matches in this label source to quantify against, so excluding it here would itself
   violate the quantify-before-excluding discipline (re-quantify if a live counter-example ever
   surfaces). Fixed by removing bare `"military"` from `rule_labeler.py`'s
   `EVENT_TYPE_KEYWORDS["geopolitical_military_tension"]`. Verified: `DELETE FROM classified_events
   WHERE label_source='rule_based'` + `labeling.run_label` + `flux_engine.run_timeseries` rebuild;
   the `rule_based`/`geopolitical_military_tension` count dropped 26 -> 23 exactly (the BTS, HELOC,
   and WWII rows, all now `unclassified`; the 2 drills rows unchanged), and the BTS `event_id` is
   confirmed absent from the live `event_catalog` (was present as `geopolitical_military_tension`
   before this fix, gone after).
   **Not part of this fix, and worth noting separately:** the other two headlines the same user flagged
   ("Former UWL student found not guilty of sexual assault charge", "Adultery Is Still A Felony In 16
   States Including Massachusetts") turned out to be `gdelt_derived` rows sitting in the 123-row
   residual disclosed in "Known limitations" item 8 below -- simply not yet run through the GDELT
   Stage 1/2 recheck, not a new bug. They'll resolve once more rounds of that cascade are run.
   **Also worth noting: no new `event_type` category was added.** The user's underlying question was
   whether headlines like these (entertainment/human-interest/legal-trivia) need one. They don't --
   `unclassified` already serves exactly that purpose and is structurally invisible downstream
   (`flux_engine/query.py::load_events()` and `flux_engine/formula.py::_EXCLUDED_EVENT_TYPES` both
   exclude it, and `flux_engine/timeseries.py::build_clusters()` only ever writes cluster-winner rows
   into `event_catalog` in the first place, so `unclassified` rows never reach it, the API, or the
   frontend at all). A new named category would need new wiring in
   `propagation/graph.py::EVENT_CHANNEL_ACTIVATION` for no benefit over what `unclassified` already
   provides -- the real gap was classification precision routing these rows to `unclassified`, not a
   missing label.
10. **Regression: `labeling.run_label` silently resurrected all 9,528 prior GDELT corrections
    (2026-08-16, self-caused by the bug-9 fix above, caught the same day when the user re-ran
    `run_gdelt_reclassify` and saw previously-fixed domestic headlines reappear).** Running
    `labeling.run_label` to relabel `rule_based` rows for bug 9 also unconditionally re-ran
    `gdelt_labeler` (`run_label.py`'s `main()` always runs all three labelers, with no way to scope to
    just one). `gdelt_labeler`'s candidate query (`LabelStore.unlabeled_raw_events`) is a plain
    anti-join against `classified_events`, with zero awareness of `gdelt_content_checked`. But
    `run_gdelt_reclassify.py`'s corrections are delete-only -- a reject verdict removes the
    `classified_events` row entirely (see its module docstring's "Correction semantics"). That made
    every one of the 9,528 rows ever rejected across rounds 1-10 look "unlabeled" again, so `run_label`
    silently re-inserted every one of them with the original, uncorrected QuadClass-only label --
    undoing effectively all of the GDELT correction work in one command. Worse, because those event
    ids were already recorded in `gdelt_content_checked`, `run_gdelt_reclassify.py`'s own candidate
    query (`_candidate_rows()`, which excludes anything already checked) would never look at them
    again either -- the resurrected wrong label was permanently stuck, invisible to both scripts'
    idempotency logic. Of the 9,528 resurrected rows, 3,408 were live cluster winners in
    `event_catalog` -- actually visible on the OSINT feed, which is how this was noticed. Fixed at the
    root: `run_label.py`'s `_run_gdelt()` now excludes any event id already present in
    `gdelt_content_checked` (confirm or reject) from its candidate set via a new
    `_already_content_checked_ids()` helper, so a correction can never come back regardless of why or
    how many times `run_label` runs afterward. Recovered by re-deleting all 9,528 resurrected rows
    (using their already-recorded verdicts -- no need to re-run Stage 1/2) and rebuilding
    `event_catalog`; verified 0 remain resurrected in either `classified_events` or `event_catalog`,
    and confirmed a subsequent `run_label` run now reports 0 gdelt candidates (previously reported
    thousands). Disclosed in full per this project's standing rule, same as bug 8/bug 5 -- a
    self-caused regression is still written down, not just quietly patched.
11. **Orchestration gap: `run_gdelt_reclassify.py` was never wired into `scripts/refresh_pipeline.sh`,
    so a routine pipeline refresh could silently regrow the backlog (found 2026-08-16, same day as bugs
    9/10, when the user reported still seeing false positives after those fixes).** `_candidate_rows()`
    in `run_gdelt_reclassify.py` finds candidates by reading `event_catalog` -- i.e. it can only see and
    correct false positives that are *currently a cluster-representative winner*.
    `flux_engine.run_timeseries` does a full historical re-cluster on every run, which can promote a
    fresh batch of previously-non-winning `gdelt_derived` rows into `event_catalog` -- rows that have
    never been through Stage 1/2 and can carry the exact same QuadClass-only false positives this whole
    re-check pipeline exists to catch (the "promotion cascade" already documented above from the
    original 10-round effort). `scripts/refresh_pipeline.sh` runs `run_timeseries` as step 4 (of the
    original 8) but never called `run_gdelt_reclassify.py` at all, so every pipeline refresh could
    reintroduce a fresh, entirely unchecked crop of false positives with nothing to catch them. Found
    live: 99 unchecked candidates (down from the previously-disclosed 123 -- someone had re-run
    `run_gdelt_reclassify.py` manually, then `run_timeseries` ran again afterward with no follow-up
    correction pass, promoting the fresh 99), including concrete confirmed false positives (`"Charges
    dropped against Manassas mom indicted for murder of estranged husband"`, `"Quincy man sentenced to
    four years in prison following child pornography plea"`). **Why this isn't fixed by embedding a
    full convergence loop into the pipeline script:** the historical round-by-round backlog sizes
    (2,301 → ... → 157 → 123 → 99) decay slowly, roughly ~20% per round recently -- draining fully to
    zero would take an estimated 10+ more rounds, each costing tens of minutes of local Ollama time, far
    too expensive for a script whose own header says "run this before a demo." Fixed instead by adding a
    single bounded reclassify+rebuild pass (`run_gdelt_reclassify.py` then a second
    `flux_engine.run_timeseries`) as new steps 5-6 of `scripts/refresh_pipeline.sh` (renumbered 1-10) --
    catches the bulk of each run's fresh promotions immediately (single-pass correction rates have
    historically run 88%+) without making a routine refresh open-ended. The residual this doesn't
    catch is the same kind of disclosed, not-fully-closed gap as the 123/99-row residual above -- run
    `labeling.run_gdelt_reclassify` + `flux_engine.run_timeseries` manually, repeated, for further
    reduction before something high-stakes. The 99-row backlog was pruned as part of this fix (single
    reclassify+rebuild pass); both concrete examples above are confirmed `unclassified` and absent from
    `event_catalog`. The pass itself promoted a further 78-row residual (same cascade, continuing) --
    disclosed rather than chased to zero, same posture as the 123/99 residuals before it.
12. **Redesign: default-deny gate + mandatory Stage 2 confidence scoring, replacing the reactive
    delete-based correction model entirely (2026-08-16, same day, at the user's explicit request for a
    more aggressive strategy after bugs 9-11 kept resurfacing false positives).** Every prior fix this
    session was reactive: find a bad row (or class of rows), correct it, hope nothing regrows it. Bug 11
    showed that hope doesn't hold -- any `flux_engine.run_timeseries` re-cluster can promote a fresh,
    unchecked row, and the reactive model has no way to stop that structurally. The redesign has two
    parts:
    - **Stage 1 "confirm" no longer means "trusted."** It used to skip Stage 2 entirely and keep
      `gdelt_labeler.py`'s original QuadClass-only confidence (a fixed 0.55/0.40/0.15 constant, zero
      content signal, confirmed via live-DB query to have exactly 3 distinct values across 6.37M rows)
      -- the direct root cause of bugs 7 and 9. It now routes into Stage 2 like "uncertain" rows do, and
      must clear `ollama_labeler.MIN_CONFIRM_CONFIDENCE = 0.75` (chosen from the live confirm-confidence
      histogram: 0.7->7 rows, 0.8->352, 0.9->238, 1.0->1) to count as a real confirm. Stage 1 "reject"
      is unchanged -- still a free, no-Ollama-cost fast path, since it's been repeatedly quantified and
      refined (bugs 3, 7, 9, 11) and stays reliable on its own.
    - **A structural gate at `flux_engine/query.py::load_events()`** (the single choke point all of
      `flux_score`, cluster-winner selection, `event_catalog`, and the API feed read through --
      confirmed via investigation that `formula.py` has zero DB access of its own and `api/routers/
      events.py`'s `event_catalog` reads are themselves built purely from `load_events()`'s output):
      any `gdelt_derived` `geopolitical_military_tension` row now additionally requires a `confirm_stage2`
      verdict in `gdelt_content_checked` to be included at all. This is what makes the fix permanent --
      a freshly-promoted, never-checked row from a future rebuild is invisible by construction, with no
      dependency on anyone remembering to re-run `run_gdelt_reclassify.py` afterward.
    - **One-time backfill:** all 1,051 historical `confirm_stage1` verdicts were cleared and
      re-examined under the new logic (1,051 stage-1-confirmed + 70 stage-1-uncertain = 1,121 rows sent
      to Stage 2). Result: only **236/1,051 (22.5%)** of the previously-trusted Stage-1 confirms
      actually cleared the real confidence bar -- a far larger cut than anticipated, and strong direct
      evidence that Stage 1's keyword-only trust was letting substantially more through than the
      keyword-specific bugs already found (7, 9) accounted for on their own. 885 rows rejected (7 of
      those specifically for confidence-below-threshold with `event_type` still geopolitical; the
      remaining 878 were the model's own `unclassified` judgment).
    - **Coverage impact (measured, not estimated):** total `event_catalog` rows with
      `event_type='geopolitical_military_tension'` dropped from **4,364 to 836** (~81% reduction) after
      the gate + backfill. This is the accepted, disclosed tradeoff -- many date/corridor buckets that
      previously showed an unverified QuadClass guess now correctly show nothing, per this project's
      established "false negatives are the safe failure direction" posture (bug 4 above). A final
      spot-check of the 15 most recent surviving rows found all 15 genuinely geopolitical/military
      (Pyongyang/Taiwan drills, Iran-related Mideast escalation, Russia/Ukraine, North Korea missile
      test, China-Taiwan sovereignty claims) -- zero false positives in that sample.
    - **Known residual risk, disclosed rather than hidden:** among the 885 Stage-2 rejects from the
      backfill, some examples look like they could be real geopolitical signal read as `unclassified`
      by the model's own judgment (e.g. `"North Korean tactical nuclear weapon in 'final stages': South"`,
      `"North Korea vows 'total destruction' of enemy on Korean War anniversary"`) -- consistent with
      this module's already-documented capability ceiling (see `ollama_labeler.py`'s module docstring:
      the 3B model under-triggers on headlines that don't explicitly name a tracked corridor/adversary
      relationship in the sentence itself). This redesign trades more of that known false-negative risk
      for a large reduction in false positives, deliberately, at the user's request -- not a new,
      undiscovered failure mode.

#### Measured effect on `flux_score`

A full `flux_engine.run_timeseries` rebuild was run after every round of corrections (10 rebuilds
total). All showed a measurable, directionally-explainable average drop -- consistent with removing
thousands of false-positive `geopolitical_military_tension` cluster winners, never an increase
(since corrections only ever *remove* inflated events, per the delete-only correction semantics
above):

- **Round 1 fix:** ~3% average drop (all 23 tickers).
- **Round 2 fix** (measured directly, mean `flux_score` before -> after, all 23 tickers): NVDA
  0.9361→0.9273, JPM 0.7973→0.7749, PGR 0.7119→0.6870, XOM 0.9355→0.9247, OXY 0.9687→0.9638 --
  **~1.5% average drop**, smaller than round 1's as expected (round 2 is the smaller second-tier
  cascade). Financials/pharma tickers (JPM, SCHW, PGR, COF, VRTX, BMRN, MRNA, REGN: ~2.6-3.5% drop)
  moved more than semiconductors/energy (~0.5-1.8% drop), consistent with round 2's corrections
  skewing toward domestic-crime/legal/consumer-tech headlines that had been diffusely (and wrongly)
  linked across corridors, while genuine surviving military/energy-corridor signal was concentrated
  in the sectors it actually belongs to.
- **Rounds 3-10 combined** (measured for the 5 pharma tickers whose numbers were captured in every
  round's rebuild log; the other 18 tickers weren't individually logged per-round but moved in the
  same direction by construction): PFE 0.7695→0.7451 (-3.2%), VRTX 0.7974→0.7743 (-2.9%), BMRN
  0.7196→0.6938 (-3.6%), MRNA 0.7795→0.7555 (-3.1%), REGN 0.7544→0.7296 (-3.3%) -- **~3.2% further
  average drop** across rounds 3-10 combined, on top of round 2's. Each individual round's drop was
  correspondingly small (the smallest rounds, e.g. round 10's 157 candidates, moved these means by
  well under 0.1 point), consistent with the shrinking round sizes in the history table above.

Both original motivating examples (the Florida murder-suicide, the Dunes Sagebrush Lizard legal
appeal) were confirmed absent from `classified_events` after the final (round 10) rebuild -- the
fix is verified effective for the literal cases that started this investigation, not just in
aggregate. The 123-row residual documented above means this is not a claim that every false
positive is gone -- only that every candidate actually processed across all 10 rounds was, and the
two original cases specifically are among them.

## Known limitations (for Phase C to address)

1. **GDELT's 2-category ceiling** -- `gdelt_derived` rows can never be one of the 5
   economic/sectoral categories, only `geopolitical_military_tension` or `unclassified`. Real
   limitation of the source, not a bug.
2. **Company-name-substring false positives on country entities** -- "America" inside "Bank of
   America", `"US"` matching currency prefixes like "US$" -- found in the spot-check, not fixed
   (no clean regex fix; needs real NER/context, which is out of scope for a keyword bootstrap
   labeler). Flag rows with `countries=["United States"]` and no other US-relevant category as
   lower-trust for now.
3. **Exact-phrase keyword matching misses paraphrases** ("chip worker shortage" vs. the
   dictionary's "chip shortage") -- a precision-over-recall tradeoff inherent to keyword
   dictionaries, not fixed here.
4. **Weak default severity band (2) when no severity keyword matches** -- documented in
   `progress/event_taxonomy.md` §4 as a flagged ambiguity, now confirmed as a real, recurring pattern
   in the spot-check.
5. **`macroeconomic_conditions` over-absorption of routine/recurring content** (e.g. daily rate
   tables) -- documented ambiguity, now confirmed in practice.
6. **Single-label only** -- no row can carry two `event_type`s from the same labeler, per
   `progress/event_taxonomy.md` ambiguity 4. A CHIPS-Act-subsidy-tied-to-a-capacity-announcement
   article, for instance, can only get one of `regulatory_subsidy_action` /
   `corporate_strategic`, whichever the keyword-hit-count tie-break picks.
7. **`llm_assisted` labeler is unverified against the live API** -- see above.
8. **GDELT-derived `geopolitical_military_tension` had zero content validation until 2026-08-15**
   -- QuadClass=3/4 was treated as authoritative with no check against the actual headline (CAMEO's
   "Fight" root code covers armed conflict and ordinary domestic violence/crime identically). See
   `progress/event_taxonomy.md` "Flagged ambiguity 8" and the "GDELT content re-check" section below
   for the two-stage mitigation and its measured precision. Only the cluster-representative winners
   (the events that actually drive `flux_score`, per `flux_engine/timeseries.py`'s winner-take-all
   clustering) have been re-checked -- the full raw `classified_events` corpus still carries the
   original, unvalidated `gdelt_derived` labels, since re-checking millions of non-winning rows
   would have no effect on the signal and wasn't attempted. Even among cluster-representative
   winners, a small, disclosed residual (123 titled candidates as of 2026-08-16, after 10 rounds of
   an open-ended promotion cascade -- see "Estimated quality" below) remains genuinely unchecked and
   confirmed to still contain live false positives; this is not fully closed, just mostly closed.
9. **`rule_labeler.py` has no Stage-2-equivalent LLM safety net** -- unlike the GDELT path (Stage 1
   keyword filter + Stage 2 local-LLM read), `rss`/`newsapi` rows get a single keyword pass with
   nothing to catch a bad keyword list entry (see bug 9 in "Bugs found and fixed during this pass"
   above for a concrete case this let through). Keyword-list precision matters more here than on the
   GDELT path precisely because there's no second check; other generic single-word entries in
   `rule_labeler.py`'s `EVENT_TYPE_KEYWORDS` haven't been systematically audited the way `"military"`
   now has been.

None of these are silently patched further in this phase -- they're handed off as concrete,
observed (not hypothetical) input for Phase C's trained classifier and for whoever reviews
`progress/event_taxonomy.md`.
