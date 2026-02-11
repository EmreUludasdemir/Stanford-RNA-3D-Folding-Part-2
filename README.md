<p align="center">
  <img src="assets/cover.svg" alt="Stanford RNA 3D Folding 2 banner" width="100%" />
</p>

# Stanford RNA 3D Folding 2: Hybrid TBM + Ab-Initio

<p align="center">
  <a href="https://www.kaggle.com/competitions/stanford-rna-3d-folding-2">
    <img alt="Kaggle Part 2" src="https://img.shields.io/badge/Kaggle-Part%202-20BEFF?logo=kaggle&logoColor=white" />
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Active%20Development-2ea44f" />
  <img alt="Focus" src="https://img.shields.io/badge/Focus-Leaderboard%20Optimization-f39c12" />
</p>

<p align="center"><b>Language:</b> English | <a href="README_TR.md">Turkce</a></p>

High-performance Kaggle baseline for RNA 3D structure prediction (C1' coordinates), built for iterative leaderboard optimization.

## Competition

| Resource | Link |
| --- | --- |
| Stanford RNA 3D Folding (Part 1) | https://www.kaggle.com/competitions/stanford-rna-3d-folding |
| Stanford RNA 3D Folding 2 (Part 2) | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2 |
| Data | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data |
| Code | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/code |
| Models | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/models |
| Leaderboard | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/leaderboard |
| Submissions | https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/submissions |

## Performance Snapshot

| Script | Targets | Submission Shape | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `stanford_rna_3d_folding_2.py` | `28` | `(9762, 18)` | `61.3 min` | After long-sequence optimization and CIF caching |

## Approach

1. Build train-template and PDB `seqres` indices.
2. Run k-mer prefilter to select high-value candidates.
3. Align query/template sequences (SW for normal lengths, fast path for long sequences).
4. Transfer coordinates from train labels or PDB mmCIF chains.
5. Create 5-model ensemble with weighted templates + controlled diversity noise.
6. Fill missing residues with interpolation and enforce output bounds.

## Runtime Optimizations

- Sparse k-mer cosine similarity to reduce temporary array overhead.
- Long-sequence fast alignment path via `SequenceMatcher`.
- Adaptive prefilter cap for very long targets.
- mmCIF cache by PDB ID to avoid repeated parse cost.
- Fixed random seed (`RANDOM_SEED=42`) for reproducibility.

## Repository Layout

```text
.
|- assets/
|  `- cover.svg
|- stanford_rna_3d_folding_2.py
|- README.md
|- README_TR.md
|- Stanford RNA 3D Folding 2_ Comprehensive Guide from Zero to Leaderboard Top.pdf
`- Stanford RNA 3D Folding Part 2_ High-Ranking Solution Pipeline.pdf
```

## Quick Start (Kaggle Notebook)

```bash
!PYDEVD_DISABLE_FILE_VALIDATION=1 python -Xfrozen_modules=off /kaggle/working/stanford_rna_3d_folding_2.py
```

Notes:
- `-Xfrozen_modules=off` helps avoid debugger breakpoint warnings.
- `nbconvert/traitlets` warnings at the end of Kaggle logs are usually notebook export warnings, not submission blockers.

## Local Run

```bash
pip install numpy pandas
python stanford_rna_3d_folding_2.py
```

The script uses Kaggle paths by default. For local usage, update `INPUT_DIR` and `OUTPUT_PATH` in `stanford_rna_3d_folding_2.py`.

## Tunable Config

| Parameter | Purpose |
| --- | --- |
| `MAX_TEMPLATES_PER_TARGET` | Upper bound for selected templates |
| `IDENTITY_THRESHOLD` | Minimum sequence identity cutoff |
| `COVERAGE_THRESHOLD` | Minimum alignment coverage cutoff |
| `LONG_SEQUENCE_THRESHOLD` | Length threshold for long-sequence alignment path |
| `SHORT_PREFILTER_CAP` | Candidate cap for regular targets |
| `LONG_PREFILTER_CAP` | Candidate cap for long targets |
| `NOISE_SCALES` | Diversity controls for 5 output models |

## Roadmap

- [ ] Better ultra-long target handling (chunked alignment and assembly)
- [ ] Stoichiometry-aware post-assembly refinement
- [ ] MSA confidence weighting and stronger remote-homolog scoring
- [ ] Per-target profiler report for speed/quality tradeoff
- [ ] Unit tests for alignment and coordinate transfer
- [ ] Experiment tracking and config presets

## Disclaimer

This repository is an independent competition implementation and is not affiliated with Kaggle or Stanford.
