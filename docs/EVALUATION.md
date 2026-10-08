# Evaluation report

Evaluation date: 2026-10-08.

## Deterministic correctness

Command:

```bash
python -m unittest discover -s tests -v
```

All 78 tests passed. The suite includes explicit null, invalid-period,
zero-demand, cutoff-tie, distinct mean-versus-individual lead-time, and SQL-error
cases. It separates:

- tool correctness: direct pandas expected values, sample standard deviations,
  eligible/evaluated/excluded counts, ties, PO source rows, demand intervals,
  schema/grain rules, and existing SS/ROP regressions;
- security behavior: SQL AST rejection, external/unapproved readers and tables,
  writes/multiple statements, uploaded-cell instructions, bounded plan schema,
  query errors, and a real interrupt deadline for an expensive query;
- orchestration: malformed-plan repair, provider outage, ambiguity,
  dataset invalidation, and the three-turn SKU-to-PO follow-up.

The bundled-data benchmark (`python evaluate_assistant.py`) evaluated all 60
eligible SKUs for lead-time variability. Its top five were MAT-1019, MAT-1018,
MAT-1016, MAT-1057, and MAT-1010. The highest mean lead time and longest
individual receipt both happened to be MAT-1016 in this dataset, but they were
computed through distinct definitions. The longest source receipt was
PO-MAT-1016-017 at 123.4 calendar days.

## Live planning evaluation

Command:

```bash
python evaluate_assistant.py --live --runs 2
```

Configuration:

- provider: NVIDIA OpenAI-compatible endpoint;
- model: `nvidia/nemotron-3-super-120b-a12b`;
- temperature: 0;
- validated JSON plan with one repair attempt;
- repeated runs of the analytical/follow-up sequence plus contextual recommendation.

Sequence:

1. “Which SKU is having the most lead-time variability?”
2. “Which PO for this SKU has the longest lead time?”
3. “Can you pick the specific PO please?”
4. “How can we reduce this variability?”

The first two recorded runs selected `rank_skus`, then `longest_receipt`, then
`show_prior_result`, and exactly matched `MAT-1019` followed by
`PO-MAT-1019-018`. Across later repeat checks, one pre-hardening run returned
invalid model requests twice for the third turn. It failed closed with no tool
execution or invented PO. A narrow verified-context resolver was then added for
explicit “pick the specific PO” requests: one unambiguous prior PO reuses its
result ID and multiple prior POs force clarification. The final post-change run
passed all three turns with exact identifiers. The 78-test suite covers both the
unambiguous and tied-reference branches.

The expanded final live run also passed four independent cases: exact top-five
variability membership, separate highest-mean versus longest-individual
lead-time operations with exact SKU/PO identifiers, clarification for an
undefined “highly inconsistent” threshold, and an unsupported outcome for
missing OTIF data. There were no API errors, tool errors, or grounding failures
in that final run.

The response-path extension was also evaluated live. “What is safety stock?”
selected the knowledge path with no analytical operation. After the verified
MAT-1019 variability and PO sequence, “How can we reduce this variability?”
selected the recommendation path, retained MAT-1019, and referenced both prior
verified result IDs without asking for the SKU again. The provider-generated
recommendation did not satisfy the strict conceptual grounding contract, so the
curated methodology fallback was used; the answer remained available, labeled
possible causes/actions, and introduced no unverified numerical claim. Mocked
regressions additionally cover a successful generated recommendation, a mixed
analysis-plus-recommendation response, and deliberate unverified SKU/quantity
injection falling back safely.

This small live result is evidence for the measured prompts only. It is not a
claim that the provider interprets every complex or arbitrary question
correctly. Scripted-provider success validates orchestration, not model semantic
coverage. Unsupported fields and undefined business thresholds still require
an explicit limitation or clarification.
