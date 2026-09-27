# Data Access

This project uses the **EdNet-KT3** dataset, released by Riiid! Labs (the makers of the
**Santa** AI tutoring app for TOEIC preparation).

- **Source / download:** https://github.com/riiid/ednet
- **Paper:** Choi, Y. et al. (2020). *"EdNet: A Large-Scale Hierarchical Dataset in
  Education."* International Conference on Artificial Intelligence in Education (AIED
  2020). https://arxiv.org/abs/1912.03072
- **Subset used:** KT3 — 89,270,654 interactions from 297,915 students (Aug 2018–Nov
  2019), including question-answering plus reading/lecture-watching behaviour.

## Why the raw data isn't in this repo

The raw KT3 CSV is tens of millions of rows (multiple GB uncompressed) and is excluded
via `.gitignore`. To reproduce this project:

1. Download EdNet-KT3 from the link above.
2. Combine/convert the per-student files into a single CSV (or adapt the notebook's
   ingestion cell to read the per-student file structure directly).
3. Place the file at a path the notebook can read, and update the path in the data
   loading cell (currently `/Volumes/workspace/default/ednetdata/EdNet dataset.csv`,
   which is a Databricks-specific Volume path — see the portability note in
   `README.md`).

## License / usage

EdNet is released by Riiid! for research purposes. Check the current terms on the
[riiid/ednet](https://github.com/riiid/ednet) repository before any use beyond academic
coursework.
