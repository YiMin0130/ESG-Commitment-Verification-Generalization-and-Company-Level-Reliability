# Validation performed during preparation

- Before privacy cleanup, all 631 result files plus the company metadata CSV matched the relocated source byte-for-byte. Public-release cleanup changes only private paths in historical text logs and documentation/tool display behavior; CSV/JSON research data and NPZ embedding caches remain byte-for-byte unchanged. Updated file hashes describe the sanitized version.
- All 12 experiment source files parsed successfully. All 12 launcher dry runs resolved a script, ordered input and fresh output path; metadata was supplied where required. These checks did not import or run the ML frameworks.
- All 60 saved fold partitions passed ID/content/coverage checks. The intended company/report and industry isolation constraints passed. Test membership matches across all four methods within each split protocol.
- The recovered base input order matched stored positional indices wherever archived indices were available.
- All 360 recomputed classification statistics matched the archived fold CSVs to floating-point tolerance (1e-12).
- The corrected manuscript dated 2026-10-22 is the current reference. All 464 values in Tables 2–4 (440 mean/SD/interval values plus 24 Table 2 degradation/delta values) match at displayed precision, with zero mismatches. Reference values were extracted from the corrected LaTeX and checked against its compiled PDF before comparing to archive statistics. All 13 quantitative-description checks pass, including signs/ranges, company indicator errors and all 18 unseen-industry ranking intervals crossing zero.
- All local Markdown links resolved in the prepared directory. A split comparison against the archived company QLoRA run returned no differences.
- Training, GPU inference and complete company-metric recalculation were not performed. Original dependency versions and exact model revision are unknown; the unpinned requirements file has not been established as a working training environment.

Re-run the three standard-library analysis commands in the README after changing archive data. Run the audit last when refreshing the checksum inventory, so it includes all current generated reports. The manifest excludes itself; it is a content inventory, not a signature of trusted authorship.
