# DBA Residency Research Package

Topic: Explainable knowledge-based AI for data-driven requirements prioritization during digital transformation.

## Contents
- `Hemasri_DBA_Research_Manuscript.docx` — working short-communication draft, with 2 figures and 2 main tables.
- `Residency_QA_Guide.docx` — class discussion notes.
- `experiment.py` — the reproducible synthetic experiment.
- `requirements.txt` — Python package dependencies.
- `table1_metrics.csv`, `table2_class_counts.csv` — actual computed metrics and distribution.
- `figure1_framework.png`, `figure2_results.png` — presentation-ready figures.
- `experiment_summary.json` — parameters and bootstrap uncertainty.
- `shap_feature_importance.csv` — optional XGBoost SHAP attribution (if produced).

## Run
```bash
python -m pip install -r requirements.txt
python experiment.py
```

The data are **fully synthetic**; no organizational records were used. The hybrid weight is selected on validation records, and the test set is held out. The experiment may not reproduce exact figures across substantially different library versions. The earlier one-page abstract reported metrics for an unavailable experiment, and those numbers are *not* endorsed or used here.

## Important before any actual journal submission
Verify complete bibliographic author names against publishers; perform a fuller systematic novelty assessment; choose a journal and follow its instructions; collect real-world data and stakeholder evidence if claiming managerial effectiveness; review the study design with your professor; comply with AI disclosure requirements. No GitHub repository or journal submission is made by this package.
