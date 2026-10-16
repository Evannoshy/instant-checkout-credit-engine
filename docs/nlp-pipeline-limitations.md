# NLP pipeline limitations

Updated 15 October 2026 following Ari's review of [PR #18](https://github.com/Evannoshy/instant-checkout-credit-engine/pull/18).

## NLP-L01: purpose metadata in the composite keyword feature

The `medical` purpose supplies 849 of the 1,970 distress-keyword hits within medical
training loans, or 43.1%. Across all training purposes, metadata supplies 11.6% of
keyword hits. In 43.8% of medical rows, the purpose supplies a hit while the
description has none. The title may still contain a keyword in these rows.

The [description-only export](nlp-desc-distress-export.md) removes title and purpose
from both the numerator and denominator. Use it for the requested borrower-language
handoff. The original six-feature table remains an historical experiment input.

When purpose is also an explicit model input, composite density can duplicate some
category information and make attribution harder to interpret. This is a risk to
test, not evidence that a tree automatically counts the category twice. The current
`tabular_features_v1` contract contains term and home ownership as its categorical
inputs, not purpose. Description-only values may still correlate with category
because borrowers discuss their loan purpose in prose.

## NLP-L02: the error-review sample is deliberately uneven

The [audit notebook](../src/nlp/false_prediction_audit.ipynb) selects 10 extreme and
10 seeded-random errors of each type. Counts such as `bills` in 7/20 reviewed false
positives refer to the five strongest upward term contributions in those selected
cases. They are neither the frequency of every textual mention nor estimated
rates across all validation errors. Overlapping unigrams and bigrams also mean the
term counts cannot be added. Any population-frequency study needs a separately
specified census or representative sampling design.

## NLP-L03: TF-IDF bigrams can cross payload fields

`build_text_payload` puts each field on its own line and expands purpose
underscores into spaces. `strip_field_labels` removes the prefixes, but preserves
the newlines. TF-IDF tokenization treats those newlines as whitespace, so the last
token of one field can pair with the first surviving token of the next field.

A synthetic purpose `debt_consolidation` followed by a description `Pay bills`
produces the bigram `consolidation pay`. It is a join artifact, not a phrase from
the description. English stop-word removal can introduce additional token
adjacencies. This behavior is covered by a synthetic regression test using the
published tuned model configuration.

The audit preserves the published model so its explanations remain tied to the
saved scores. A separate experiment could vectorize fields independently or use
description-only inputs. Adding punctuation alone would not reliably block
cross-field bigrams with the current tokenizer.

## NLP-L04: word contributions do not explain repayment capacity

The reviewed cases show that bill and business vocabulary can raise a linear
TF-IDF score, while refinancing language can lower it. These fitted associations
do not establish the truth of self-reported income or the cause of a later default.
The scores are uncalibrated and the 0.50 audit threshold is not an approval rule.

This evidence supports evaluating text with tabular inputs and comparing models
that preserve more context. It does not establish a performance gain from fusion,
nonlinear models, or DistilBERT. Those claims require matched evaluation and
calibration evidence.
