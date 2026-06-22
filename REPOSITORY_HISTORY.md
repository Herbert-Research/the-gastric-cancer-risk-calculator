# Repository History

This repository is a public release snapshot.

Before publication, the project files were moved from a private working
directory into a clean repository so that readers can clone, inspect, and
reproduce the analysis from a stable starting point. Local environment files,
caches, temporary outputs, and earlier setup iterations were not included.

The source code, tests, configuration files, data provenance notes, and
reference documentation are intended to be reviewed from this release state.
The raw TCGA clinical data file is not distributed with this repository; see
`data/README.md` for access instructions.

To verify the project from a fresh clone, run:

```powershell
python -m pip install -r requirements.lock
python -m pytest tests/ -q
```
