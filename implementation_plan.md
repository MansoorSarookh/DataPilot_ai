# DataPilot-AI v3.0 — Implementation Plan

Upgrade from v2.0 academic prototype to production-grade AI analytics platform per the [PRD.md](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/PRD.md).

---

## Current State Analysis

### ✅ Backend Modules Already Created (29 modules in `app/modules/`)

Many v3.0 PRD modules **already exist** as backend files but are **NOT yet integrated** into the main UI:

| Module | File Exists | Integrated in UI | Status |
|--------|:-----------:|:-----------------:|--------|
| `session_manager.py` | ✅ | ❌ | Backend only — not wired into `main.py` |
| `llm_router.py` | ✅ | ❌ | Backend only — `ai_engine.py` still uses Groq-only |
| `regression_engine.py` | ✅ | ❌ | No `regression_panel.py` component |
| `anomaly_detector.py` | ✅ | ❌ | Not integrated in `clean_panel.py` |
| `audit_trail.py` | ✅ | ❌ | Not integrated in `clean_panel.py` |
| `chart_recommender.py` | ✅ | ❌ | Not integrated in Visualize tab |
| `cluster_profiler.py` | ✅ | ❌ | Not integrated in `ml_studio.py` |
| `dashboard_exporter.py` | ✅ | ❌ | Not integrated in `report_panel.py` |
| `data_dictionary.py` | ✅ | ❌ | Not integrated in `report_panel.py` |
| `fuzzy_dedup.py` | ✅ | ❌ | Not integrated in `clean_panel.py` |
| `gdpr_scanner.py` | ✅ | ❌ | Not integrated in Overview tab |
| `hpo_engine.py` | ✅ | ❌ | Not integrated in `ml_studio.py` |
| `model_arena.py` | ✅ | ❌ | Not integrated in `ml_studio.py` |
| `power_analysis.py` | ✅ | ❌ | Not integrated in `stats_panel.py` |
| `shap_explainer.py` | ✅ | ❌ | Not integrated in `ml_studio.py` |
| `stats_narrator.py` | ✅ | ❌ | Not integrated in `stats_panel.py` |
| `ts_stats_engine.py` | ✅ | ❌ | Not integrated in `stats_panel.py` |
| `type_fixer.py` | ✅ | ❌ | Not integrated in `clean_panel.py` |

### ❌ Modules Still Missing (per PRD)

| Module | PRD Section | Priority |
|--------|-------------|----------|
| `rag_engine.py` | §7.2.1 | P1 |
| `text2viz.py` | §6.2.2 | P2 |
| `automl_engine.py` | §10.2.6 | P2 |
| `pptx_exporter.py` | §11.2.2 | P2 |
| `column_renamer.py` | §5.2.3 | P2 |
| `dataset_comparator.py` | §5.2.5 | P3 |
| `relationship_detector.py` | §5.2.6 | P3 |
| `geo_engine.py` | §6.2.5 | P3 |
| `story_generator.py` | §6.2.6 | P3 |
| `agent_analyzer.py` | §7.2.6 | P3 |
| `bayesian_engine.py` | §8.2.6 | P3 |
| `causal_engine.py` | §8.2.8 | P3 |
| `data_contract.py` | §9.2.6 | P3 |
| `cleaning_simulator.py` | §9.2.7 | P3 |

### 🔴 Critical Issues Found

1. **`main.py` has ~1,000 lines of commented-out duplicate code** (lines 527-1554) — 3 full copies of the entire app commented out with error notes
2. **`requirements.txt` is outdated** — missing `shap`, `pingouin`, `optuna`, `rapidfuzz`, `joblib`, `jinja2`, `google-generativeai`, `catboost`, `python-pptx`, `pandera`
3. **`eda_engine.py` `get_correlation_matrix()`** only supports Pearson — no method parameter
4. **No VIF computation** anywhere in Overview tab
5. **`ai_engine.py` still uses Groq-only** — `llm_router.py` exists but isn't used
6. **No `.streamlit/config.toml`** theme config file
7. **Version string still says "v2.0"** in sidebar footer and main.py docstring

---

## User Review Required

> [!IMPORTANT]
> This implementation involves **substantial** changes to nearly every component. The plan focuses on P0 (critical fixes) and P1 (core features) first, deferring P3 features for later.

> [!WARNING]
> **Dependency weight**: The PRD specifies `chromadb`, `sentence-transformers`, `catboost`, `flaml`, `optuna` which add ~800MB+ of dependencies. For Streamlit Cloud compatibility, I'll implement with **lazy imports** and defer the heaviest optional deps (PyMC, DoWhy, spacy/presidio).

---

## Open Questions

1. **API Keys**: Do you have Groq and/or Gemini API keys to test multi-model LLM? (Can set them in `.streamlit/secrets.toml`)
2. **CatBoost**: Should we include CatBoost (adds ~300MB) or skip it to stay within Streamlit Cloud limits?
3. **RAG (ChromaDB + sentence-transformers)**: Include in this phase or defer? Adds ~500MB.
4. **Scope**: Should I implement ALL P1+P2 features now, or focus on P0+P1 first and do P2 in a follow-up?

---

## Proposed Changes

### Phase 0: Critical Fixes & Cleanup

#### [MODIFY] [main.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/main.py)
- **Remove ~1,000 lines of commented-out duplicate code** (lines 527–1554)
- Update version string from "v2.0" to "v3.0"
- Integrate `session_manager` at init
- Add LLM provider selector in sidebar
- Add VIF display in Overview tab
- Add correlation method toggle (Pearson/Spearman/Kendall)
- Integrate chart recommender in Visualize tab
- Wire up all new backend modules to their respective tabs

#### [MODIFY] [requirements.txt](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/requirements.txt)
- Remove duplicate `pdfplumber` entry
- Add all new v3.0 dependencies: `pingouin`, `shap`, `optuna`, `rapidfuzz`, `joblib`, `jinja2`, `google-generativeai`, `python-pptx`, `pandera`, `python-dateutil`
- Remove stale `tabula-py` (unused)

#### [NEW] `.streamlit/config.toml`
- Streamlit theme config per PRD §14.1

---

### Phase 1: Overview Tab Upgrades

#### [MODIFY] [eda_engine.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/modules/eda_engine.py)
- Add `compute_vif(df)` function using `statsmodels.stats.outliers_influence`
- Add `method` parameter to `get_correlation_matrix(df, method='pearson')`

---

### Phase 2: Statistics Tab Upgrades

#### [MODIFY] [stats_engine.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/modules/stats_engine.py)
- Add effect size computation (`compute_effect_size()`) — Cohen's d, eta-squared, Cramér's V
- Add multiple comparison correction (Bonferroni, Benjamini-Hochberg)

#### [MODIFY] [stats_panel.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/components/stats_panel.py)
- Integrate effect size display alongside p-values
- Integrate AI Statistical Narrator (`stats_narrator.py`)
- Add regression analysis sub-tab (wiring `regression_engine.py`)
- Add time-series statistics sub-tab (wiring `ts_stats_engine.py`)
- Add power analysis sub-tab (wiring `power_analysis.py`)

#### [NEW] `app/components/regression_panel.py`
- Regression analysis UI with residual diagnostics plots

---

### Phase 3: Clean Tab Upgrades

#### [MODIFY] [cleaner.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/modules/cleaner.py)
- Add KNN Imputer and Iterative Imputer (MICE) methods

#### [MODIFY] [clean_panel.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/components/clean_panel.py)
- Integrate audit trail UI (`audit_trail.py`)
- Integrate anomaly detection section (`anomaly_detector.py`)
- Integrate type fixer section (`type_fixer.py`)
- Integrate fuzzy deduplication section (`fuzzy_dedup.py`)
- Add ML imputation options to dropdown

---

### Phase 4: ML Studio Upgrades

#### [MODIFY] [ml_advisor.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/modules/ml_advisor.py)
- Add SVM, KNN, Decision Tree, Naive Bayes, ElasticNet, Ridge, Lasso, LightGBM algorithms
- Add model serialization (joblib download)

#### [MODIFY] [ml_studio.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/components/ml_studio.py)
- Integrate SHAP explainability (`shap_explainer.py`)
- Integrate model comparison arena (`model_arena.py`)
- Integrate hyperparameter tuning (`hpo_engine.py`)
- Integrate cluster profiler (`cluster_profiler.py`)
- Add model download (.joblib) button

---

### Phase 5: AI Copilot Upgrades

#### [MODIFY] [ai_engine.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/modules/ai_engine.py)
- Replace `get_groq_client()` with `LLMRouter` integration
- Add code generation mode to system prompt

#### [MODIFY] [ai_chat.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/components/ai_chat.py)
- Add model selector dropdown
- Add code generation toggle
- Show active LLM provider badge

---

### Phase 6: Visualize Tab Upgrades

#### [MODIFY] [viz_engine.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/modules/viz_engine.py)
- Add `create_sankey()` function
- Add animated bar race / area chart functions

#### Integration in main.py Visualize tab:
- Wire chart recommender above chart selector
- Add Sankey diagram option to Multivariate charts

---

### Phase 7: Export Tab Upgrades

#### [MODIFY] [report_panel.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/components/report_panel.py)
- Integrate interactive HTML dashboard export (`dashboard_exporter.py`)
- Integrate data dictionary export (`data_dictionary.py`)
- Enhance Jupyter notebook with rich mode (`notebook_exporter.py`)

---

### Phase 8: Time-Series Parser Fix

#### [MODIFY] [time_series.py](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/app/modules/time_series.py)
- Add `robust_datetime_parse()` with multi-strategy fallback
- Support Unix timestamp detection
- Use `format="mixed"` parameter

---

### Phase 9: README & Config

#### [MODIFY] [README.md](file:///c:/Users/aikasaal/OneDrive/Documents/DataPilot-AI/README.md)
- Complete professional rewrite covering all v3.0 features A-to-Z

#### [NEW] `.streamlit/config.toml`
#### [NEW] `requirements-optional.txt` (heavy deps)

---

## Verification Plan

### Automated Tests
```bash
# Activate venv and install deps
datapilot_venv\Scripts\activate
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/main.py
```

### Manual Verification
- Upload a CSV file and verify all 7 tabs load without errors
- Test each new feature integration (VIF, correlation toggle, effect sizes, etc.)
- Verify LLM router fallback works
- Test export features (PDF, HTML dashboard, data dictionary)
