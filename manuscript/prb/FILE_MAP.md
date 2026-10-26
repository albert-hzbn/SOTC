# Manuscript file map

- `manuscript_original.tex`: byte-identical copy of the originally submitted manuscript.
- `refs_original.bib`: byte-identical bibliography snapshot from the original manuscript commit.
- `manuscript_original.pdf`: rendering of that source; one malformed math token was corrected only in a temporary `/tmp` build stream because the preserved source itself must remain byte-identical.
- `manuscript_clean.tex` / `manuscript_clean.pdf`: clean revised manuscript with no revision coloring.
- `manuscript_updated.tex` / `manuscript_updated.pdf`: selective marked revised manuscript; additions and changed equations are blue, unchanged material is black, and deleted text is omitted. The source contains no hidden deletion or latexdiff tracking macros.
- `cover_letter.tex` / `cover_letter.pdf`: resubmission cover letter aligned with the narrowed claims in the revised manuscript.
- `reviews/response_to_reviewers.tex` / `.pdf`: point-by-point reviewer response.

The original files are not generated from either revised manuscript and must not be overwritten during marked-manuscript generation.
