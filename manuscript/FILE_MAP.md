# Manuscript file map

- `manuscript_original.tex`: byte-identical copy of the originally submitted manuscript.
- `refs_original.bib`: byte-identical bibliography snapshot from the original manuscript commit.
- `manuscript_original.pdf`: rendering of that source; one malformed math token was corrected only in a temporary `/tmp` build stream because the preserved source itself must remain byte-identical.
- `manuscript.tex`: preserved original source retained for repository history and hash comparison.
- `manuscript_clean.tex` / `manuscript_clean.pdf`: clean revised manuscript with no revision coloring.
- `manuscript_updated.tex` / `manuscript_updated.pdf`: revised manuscript with every changed sentence shown in blue.
- `../reviews/response_to_reviewers.tex` / `.pdf`: point-by-point reviewer response.

The original files are not generated from either revised manuscript and must not be overwritten during marked-manuscript generation.
