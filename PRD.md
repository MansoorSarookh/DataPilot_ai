# DataPilot-AI v3.0 — Product Requirements Document (PRD)

> **Project:** DataPilot-AI — Next-Generation AI Data Intelligence Copilot  
> **Author:** Mansoor Sarookh  
> **Date:** June 2026  
> **Version:** 3.0 (Major Upgrade)  
> **Framework:** Streamlit (Python 3.11)  
> **Deployment Targets:** Streamlit Cloud · Hugging Face Spaces  
> **Repository:** [github.com/MansoorSarookh/DataPilot-AI](https://github.com/MansoorSarookh/DataPilot-AI)  
> **Live App:** [datapilot-ai.streamlit.app](https://datapilot-ai.streamlit.app)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Vision & Goals](#2-product-vision--goals)
3. [Current Architecture Analysis](#3-current-architecture-analysis)
4. [System-Wide Upgrades](#4-system-wide-upgrades)
5. [Tab 1: Overview — Specifications](#5-tab-1-overview--specifications)
6. [Tab 2: Visualize — Specifications](#6-tab-2-visualize--specifications)
7. [Tab 3: AI Copilot — Specifications](#7-tab-3-ai-copilot--specifications)
8. [Tab 4: Statistics — Specifications](#8-tab-4-statistics--specifications)
9. [Tab 5: Clean — Specifications](#9-tab-5-clean--specifications)
10. [Tab 6: ML Studio — Specifications](#10-tab-6-ml-studio--specifications)
11. [Tab 7: Export — Specifications](#11-tab-7-export--specifications)
12. [Non-Functional Requirements](#12-non-functional-requirements)
13. [Technology Stack & Dependencies](#13-technology-stack--dependencies)
14. [Deployment Strategy](#14-deployment-strategy)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Risk Assessment & Mitigations](#16-risk-assessment--mitigations)
17. [Success Metrics & KPIs](#17-success-metrics--kpis)
18. [Appendix: File Structure](#18-appendix-file-structure)

---

## 1. Executive Summary

DataPilot-AI is an AI-powered Exploratory Data Analysis (EDA) platform built with Streamlit that enables users to upload datasets and instantly receive statistical summaries, interactive visualizations, AI-driven insights, ML training capabilities, and exportable reports — all through a no-code interface.

**Current State (v2.0):** The platform has 7 functional tabs (Overview, Visualize, AI Copilot, Statistics, Clean, ML Studio, Export) with core features including a Dataset Trust Score, Groq LLM integration, statistical test recommendation, and Jupyter notebook export. However, it suffers from limited ML algorithms, no persistent memory, fragile time-series parsing, and shallow statistical/cleaning capabilities.

**Target State (v3.0):** A comprehensive upgrade that transforms DataPilot-AI from an academic prototype into a production-grade, research-worthy AI analytics platform with:
- **56+ new features** across all 7 tabs
- **RAG-powered AI memory** with ChromaDB vector storage
- **AutoML pipeline** with SHAP explainability
- **Bayesian inference** and causal analysis modules
- **Interactive HTML dashboard** export
- **Multi-model LLM** support (Groq + Gemini + Ollama)
- **Advanced cleaning** with ML-based imputation and anomaly detection

---

## 2. Product Vision & Goals

### 2.1 Vision Statement
> *"Make DataPilot-AI the most intelligent, comprehensive, and user-friendly no-code data analytics platform available as open-source — capable of replacing 80% of a junior data scientist's daily workflow."*

### 2.2 Strategic Goals

| # | Goal | Measurable Outcome |
|---|------|-------------------|
| G1 | Deepen analytics capabilities | Support 15+ statistical tests, 10+ ML algorithms, Bayesian inference |
| G2 | Achieve AI-native experience | RAG memory, code generation, chart-from-chat, agentic analysis |
| G3 | Enterprise-grade data quality | ML imputation, anomaly detection, data contracts, audit trail |
| G4 | Professional export ecosystem | Interactive HTML dashboards, PowerPoint decks, rich Jupyter notebooks |
| G5 | Deployment reliability | Stable deployment on Streamlit Cloud + Hugging Face Spaces |
| G6 | Academic research contribution | Novel features worthy of FYP publication (Trust Score v2, AI Narrator, Causal Inference) |

### 2.3 Target Users

| Persona | Needs | DataPilot-AI Value |
|---------|-------|--------------------|
| **Data Science Students** | Quick EDA, learning statistical tests, building ML models | No-code interface with educational explanations |
| **Academic Researchers** | Statistical rigor, reproducibility, hypothesis testing | Bayesian inference, effect sizes, Jupyter export |
| **Business Analysts** | Visual insights, shareable reports, trend detection | AI Chart Recommender, interactive dashboards, PDF/PPT export |
| **Hackathon Teams** | Rapid prototyping, quick model training | AutoML, 1-click reports, AI Copilot |
| **Educators** | Teaching aid for statistics and ML concepts | AI Statistical Narrator, What-If Analysis, distribution visualizations |

### 2.4 Constraints

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| Streamlit single-page architecture | No true multi-page routing, limited concurrency | Use tabs + session state isolation |
| Streamlit Cloud 1GB memory limit | Cannot load very large datasets | Implement sampling for datasets > 100K rows |
| Hugging Face Spaces 16GB disk limit | Must manage dependencies carefully | Optimize requirements.txt, use lazy imports |
| No persistent backend database | Session data lost on refresh | Use `st.session_state` + optional SQLite/localStorage bridge |
| Groq API rate limits | May throttle under heavy use | Implement multi-model fallback (Groq → Gemini → local heuristics) |

---

## 3. Current Architecture Analysis

### 3.1 Existing File Structure

```
DataPilot-AI/
├── app/
│   ├── main.py                    # Main Streamlit app (1,554 lines — monolithic)
│   ├── config.py                  # App constants, chart types, themes
│   ├── components/                # UI components
│   │   ├── ai_chat.py             # AI Copilot chat UI
│   │   ├── clean_panel.py         # Data cleaning UI
│   │   ├── data_preview.py        # Dataset preview cards
│   │   ├── export.py              # Chart export buttons
│   │   ├── ml_studio.py           # ML training UI
│   │   ├── report_panel.py        # Report generation UI
│   │   ├── sidebar.py             # Sidebar (unused — inlined in main.py)
│   │   ├── statistics.py          # Basic statistics display
│   │   ├── stats_panel.py         # Statistical Intelligence Engine UI
│   │   └── trust_score_display.py # Trust Score visualization
│   └── modules/                   # Backend logic
│       ├── ai_engine.py           # Groq LLM integration + fallback
│       ├── cleaner.py             # Data cleaning pipeline
│       ├── eda_engine.py          # EDA functions (column info, stats, correlation)
│       ├── file_parser.py         # Multi-format file parsing
│       ├── ml_advisor.py          # ML pipeline (RF, LR, XGB, K-Means)
│       ├── notebook_exporter.py   # Jupyter notebook generation
│       ├── report_generator.py    # PDF/HTML report generation
│       ├── stats_engine.py        # Statistical test engine
│       ├── time_series.py         # Time-series detection & analysis
│       ├── trust_score.py         # Trust Score computation
│       └── viz_engine.py          # Plotly visualization factory
├── assets/
│   └── styles.css                 # Custom CSS styling
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version (3.11)
└── README.md                      # Project documentation
```

### 3.2 Strengths (Preserve & Enhance)

| Feature | Current Implementation | Assessment |
|---------|----------------------|------------|
| **Dataset Trust Score** | 5-dimension composite score (completeness, consistency, variance, balance, uniqueness) with PII detection | ✅ Novel — enhance with GDPR heatmap |
| **AI Copilot (Groq)** | Conversational dataset Q&A with quick-action buttons and rule-based fallback | ✅ Production-grade — add RAG + multi-model |
| **Statistical Intelligence Engine** | Auto-recommends Mann-Whitney U, ANOVA, Chi-Square based on variable types | ✅ Research-worthy — expand to 15+ tests |
| **Hypothesis Builder** | Natural-language hypothesis construction with auto test selection | ✅ UX Innovation — add effect sizes and plain-English interpretation |
| **Jupyter Export** | Generates reproducible .ipynb from analysis session | ✅ Unique — enhance with rich narrative |
| **PII Detection** | Regex-based column name matching for GDPR awareness | ✅ Enterprise-grade thinking — add NER-based detection |
| **Multi-format Input** | CSV, XLSX, TSV, ODS, PDF, DOCX, JSON, Parquet, Feather, HTML | ✅ Comprehensive — maintain |
| **Distribution Analyzer** | Shapiro-Wilk normality test, Q-Q plots, transform suggestions | ✅ Textbook-to-practice — add more tests |

### 3.3 Weaknesses (Fix in v3.0)

| # | Weakness | Severity | Fix Priority |
|---|----------|----------|-------------|
| W1 | No persistent session — every refresh loses all work | 🔴 Critical | P0 |
| W2 | LLM dependency is Groq-only — no fallback, no local model | 🟡 High | P1 |
| W3 | ML Studio shallow: only 3 algorithms, no CV UI, no SHAP | 🟡 High | P1 |
| W4 | No data versioning or lineage tracking | 🟡 High | P1 |
| W5 | Time-series parser is fragile — datetime parsing fails frequently | 🔴 Critical | P0 |
| W6 | No model explainability (SHAP/LIME) | 🟡 High | P1 |
| W7 | Export reports are static — not interactive dashboards | 🟡 High | P1 |
| W8 | No RAG system — AI copilot loses context between messages | 🟡 High | P1 |
| W9 | Correlation only shows Pearson — no Spearman/Kendall toggle | 🟢 Medium | P2 |
| W10 | No VIF (Variance Inflation Factor) computation | 🟢 Medium | P2 |
| W11 | Only 3 statistical tests available | 🟡 High | P1 |
| W12 | No effect size reporting (Cohen's d, eta-squared) | 🟡 High | P1 |
| W13 | Imputation only uses simple statistics (mean/mode/median) | 🟢 Medium | P2 |
| W14 | No anomaly detection beyond IQR | 🟢 Medium | P2 |
| W15 | No cleaning audit trail with undo | 🟢 Medium | P2 |
| W16 | Commented-out duplicate code in main.py and report_panel.py | 🟢 Low | P2 |

---

## 4. System-Wide Upgrades

### 4.1 Session Persistence Layer

**Problem:** Every page refresh loses the entire analysis session.

**Solution:**
```python
# app/modules/session_manager.py
class SessionManager:
    """Manages persistent session state using st.session_state + optional SQLite."""
    
    def save_state(self, key: str, value: Any) -> None
    def load_state(self, key: str) -> Any
    def get_session_id(self) -> str
    def export_session(self) -> dict
    def import_session(self, data: dict) -> None
```

**Implementation:**
- Use `st.session_state` as primary storage (Streamlit-native)
- Implement browser `localStorage` bridge via `streamlit-js-eval` for cross-refresh persistence
- Optional: SQLite backend for session snapshots (for local development)
- Auto-save state on every major user action (file upload, cleaning, model training)

**Files Affected:**
- `[NEW] app/modules/session_manager.py`
- `[MODIFY] app/main.py` — integrate session manager at init

---

### 4.2 Multi-Model LLM Architecture

**Problem:** Groq-only dependency with no fallback.

**Solution:**
```python
# app/modules/llm_router.py
class LLMRouter:
    """Routes LLM requests to available providers with automatic fallback."""
    
    PROVIDERS = {
        "groq": {"model": "llama-3.3-70b-versatile", "priority": 1},
        "gemini": {"model": "gemini-2.0-flash", "priority": 2},
        "ollama": {"model": "llama3.2", "priority": 3},  # local fallback
    }
    
    def chat(self, messages: list, max_tokens: int = 600) -> str
    def get_available_providers(self) -> list
    def set_preferred_provider(self, provider: str) -> None
```

**Implementation:**
- UI toggle in sidebar to select LLM provider
- Automatic fallback chain: Groq → Gemini Flash → Ollama → Rule-based heuristics
- API key management via `st.secrets` (Streamlit Cloud) or environment variables
- Unified response format across all providers

**Files Affected:**
- `[NEW] app/modules/llm_router.py`
- `[MODIFY] app/modules/ai_engine.py` — replace `get_groq_client()` with `LLMRouter`
- `[MODIFY] app/main.py` — add LLM provider selector to sidebar

---

### 4.3 Robust DateTime Parser

**Problem:** Time-series parsing fails frequently on non-standard date formats.

**Solution:**
```python
# In app/modules/time_series.py — enhanced datetime inference
def robust_datetime_parse(series: pd.Series) -> pd.Series:
    """Multi-strategy datetime parser with fallback chain."""
    strategies = [
        pd.to_datetime(series, infer_datetime_format=True),
        pd.to_datetime(series, format="mixed", dayfirst=False),
        dateutil_parse_series(series),  # dateutil.parser for edge cases
        epoch_detection(series),         # Unix timestamp detection
    ]
```

**Implementation:**
- Use `pandas.to_datetime` with `format="mixed"` (pandas 2.0+)
- Fallback to `dateutil.parser` for exotic formats
- Auto-detect Unix timestamps (seconds/milliseconds)
- User override: manual format string input via text field
- Show parsed sample for user confirmation before proceeding

**Files Affected:**
- `[MODIFY] app/modules/time_series.py` — rewrite `detect_datetime_columns()`
- `[MODIFY] app/main.py` — improve Time Series tab error handling

---

### 4.4 User Onboarding System

**Problem:** No guided tour for new users.

**Solution:**
- Implement a tips/tour system using `st.session_state` tracking
- Show contextual tooltips on first visit to each tab
- "Getting Started" expander on the Welcome Screen with walkthrough steps
- Quick-start sample dataset button (load bundled demo CSV)

**Files Affected:**
- `[NEW] app/components/onboarding.py`
- `[MODIFY] app/main.py` — integrate onboarding on first visit

---

### 4.5 Code Cleanup

**Problem:** Commented-out duplicate code in `main.py` (lines 527–800+) and `report_panel.py` (lines 282–458).

**Solution:**
- Remove all commented-out duplicate code blocks
- Extract remaining monolithic logic from `main.py` into proper component functions
- Ensure `main.py` becomes a thin orchestrator (~200 lines)

**Files Affected:**
- `[MODIFY] app/main.py` — remove dead code, refactor tab logic into components
- `[MODIFY] app/components/report_panel.py` — remove commented-out block

---

## 5. Tab 1: Overview — Specifications

### 5.1 Current State

| Feature | Status | Component |
|---------|--------|-----------|
| Dataset preview (head/tail/sample) | ✅ Working | `data_preview.py` |
| Trust Score display (5 dimensions) | ✅ Working | `trust_score_display.py` |
| Column info table | ✅ Working | `eda_engine.py` |
| Descriptive statistics | ✅ Working | `statistics.py` |
| Pearson correlation matrix | ✅ Working | `eda_engine.py` + `viz_engine.py` |
| Missing values visualization | ✅ Working | `eda_engine.py` + `viz_engine.py` |
| Quick insights | ✅ Rule-based | Inline in `main.py` |

### 5.2 Upgrade Features

#### 5.2.1 VIF (Variance Inflation Factor)
- **Priority:** P1 | **Difficulty:** Easy | **Innovation:** 8/10
- **What:** Automatically compute VIF for all numeric features and flag high multicollinearity (VIF > 5 = warning, VIF > 10 = critical).
- **Why:** Current correlation matrix shows multicollinearity warning but never computes VIF — the industry standard metric for detecting it.
- **Tech Stack:** `statsmodels.stats.outliers_influence.variance_inflation_factor`
- **UI Spec:**
  - Table showing each numeric column with its VIF value
  - Color-coded: Green (VIF < 5), Orange (5-10), Red (> 10)
  - Actionable recommendation: "Consider removing or combining these features"
  - Display below correlation matrix in Overview tab
- **Files:**
  - `[MODIFY] app/modules/eda_engine.py` — add `compute_vif(df)` function
  - `[MODIFY] app/main.py` — render VIF table in Overview tab

#### 5.2.2 Smart Correlation Heatmap (Toggle Pearson/Spearman/Kendall)
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Allow users to toggle between Pearson, Spearman, and Kendall correlation methods. Click on any cell to see the scatter plot + regression line for that pair.
- **Why:** Pearson-only is misleading for non-linear relationships or ordinal data.
- **Tech Stack:** `scipy.stats`, `plotly`
- **UI Spec:**
  - Three-button toggle above the heatmap: [Pearson] [Spearman] [Kendall]
  - Click on a cell → popup/expander with scatter plot of those two variables
  - Regression line with equation and R² displayed
- **Files:**
  - `[MODIFY] app/modules/eda_engine.py` — add `get_correlation_matrix(df, method)` parameter
  - `[MODIFY] app/modules/viz_engine.py` — add cell-click scatter creation
  - `[MODIFY] app/main.py` — add toggle UI and click handler

#### 5.2.3 AI-Powered Column Renaming
- **Priority:** P2 | **Difficulty:** Easy | **Innovation:** 7/10
- **What:** Use LLM to suggest cleaner, standardized column names based on data types and sample values.
- **Why:** Real-world datasets have messy column names (e.g., `stu_id_2`, `Unnamed: 0`). Auto-suggesting `student_id` improves readability.
- **Tech Stack:** LLM Router + regex normalization
- **UI Spec:**
  - Button: "🔤 Suggest Clean Column Names"
  - Side-by-side table: Original Name → Suggested Name
  - Checkboxes to accept/reject individual suggestions
  - "Apply Selected" button
- **Files:**
  - `[NEW] app/modules/column_renamer.py`
  - `[MODIFY] app/main.py` — add column rename section in Overview

#### 5.2.4 Interactive Schema Editor
- **Priority:** P2 | **Difficulty:** Easy | **Innovation:** 7/10
- **What:** Let users correct detected data types (e.g., ID column misdetected as numeric) with dropdown selectors.
- **Why:** pandas frequently misdetects types — IDs as int64, dates as object. Users need manual override.
- **Tech Stack:** Streamlit widgets, `pandas.astype()`
- **UI Spec:**
  - Editable table: Column Name | Detected Type | Override Type (dropdown) | Sample Values
  - "Apply Type Corrections" button
  - Show before/after comparison
- **Files:**
  - `[NEW] app/components/schema_editor.py`
  - `[MODIFY] app/main.py` — integrate in Overview tab

#### 5.2.5 Dataset Comparison Mode
- **Priority:** P3 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Upload two datasets and get a side-by-side diff — schema changes, distribution shifts, statistical drift.
- **Why:** Critical for detecting data drift between training and production data, or comparing versions.
- **Tech Stack:** `scipy.stats` (KS-test), `deepdiff`
- **UI Spec:**
  - Secondary file uploader: "📂 Upload Comparison Dataset"
  - Side-by-side metrics: row counts, column overlap, distribution KS-test results
  - Diff table showing added/removed/changed columns
  - Distribution overlay plots for matching columns
- **Files:**
  - `[NEW] app/modules/dataset_comparator.py`
  - `[NEW] app/components/comparison_view.py`
  - `[MODIFY] app/main.py` — add comparison mode toggle in Overview

#### 5.2.6 Semantic Relationship Graph
- **Priority:** P3 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Auto-detect likely foreign-key relationships between columns and visualize as an interactive network graph.
- **Why:** Understanding inter-column relationships is crucial for feature engineering and database normalization.
- **Tech Stack:** `networkx`, `pyvis` (or Plotly network graph)
- **UI Spec:**
  - Interactive network graph: nodes = columns, edges = detected relationships
  - Edge types: high correlation, shared unique values, potential FK
  - Hover to see relationship strength and type
  - Collapsible panel below the correlation matrix
- **Files:**
  - `[NEW] app/modules/relationship_detector.py`
  - `[MODIFY] app/main.py` — add relationship graph section

#### 5.2.7 GDPR Risk Heatmap
- **Priority:** P3 | **Difficulty:** Hard | **Innovation:** 9/10
- **What:** Visualize PII risk per column as a color-coded heatmap with remediation recommendations (anonymize, hash, drop).
- **Why:** Current PII detection is regex-only on column names. This adds actual value scanning using NER.
- **Tech Stack:** `presidio` or lightweight regex NER, `spacy` (optional)
- **UI Spec:**
  - Heatmap: columns on Y-axis, risk categories on X-axis (names, emails, phones, SSNs, etc.)
  - Color intensity = detection confidence
  - Click column → see detected PII samples (masked) + remediation options
  - Export: "📥 Download GDPR Compliance Report"
- **Files:**
  - `[NEW] app/modules/gdpr_scanner.py`
  - `[NEW] app/components/gdpr_heatmap.py`
  - `[MODIFY] app/modules/trust_score.py` — integrate deeper PII scan

---

## 6. Tab 2: Visualize — Specifications

### 6.1 Current State

| Feature | Status | Component |
|---------|--------|-----------|
| Univariate: Histogram, Box, Violin, Bar, Pie, KDE | ✅ Working | `viz_engine.py` |
| Bivariate: Scatter, Line, Grouped Bar, Heatmap | ✅ Working | `viz_engine.py` |
| Multivariate: 3D Scatter, Parallel Coordinates, Bubble, Sunburst, Treemap, Pair Plot | ✅ Working | `viz_engine.py` |
| Time Series: Line with slider, Area Chart, Rolling Statistics | ✅ Working (fragile) | `time_series.py` |
| Chart export (PNG, SVG, HTML, GIF) | ✅ Working | `export.py` |

### 6.2 Upgrade Features

#### 6.2.1 AI Chart Recommender
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Auto-suggest the best 3 chart types for selected columns based on data type, cardinality, distribution shape, and relationship type.
- **Why:** Users (especially non-technical ones) don't know which chart type suits their data. Wrong chart choice leads to misleading insights.
- **Tech Stack:** Rule engine + LLM Router for explanation
- **UI Spec:**
  - After selecting columns, show: "🤖 Recommended Charts: 1. Violin Plot (best for skewed distributions), 2. Box Plot, 3. Histogram"
  - Each recommendation has a brief explanation
  - Click to auto-generate the recommended chart
  - Below the chart selector dropdown as a collapsible info panel
- **Logic:**
  ```
  IF numeric + high cardinality → Histogram/KDE/Box Plot
  IF numeric + low cardinality → Bar Chart
  IF 2 numeric → Scatter (+ trendline if correlation > 0.5)
  IF numeric + categorical → Violin/Box grouped by category
  IF 2 categorical → Grouped Bar / Stacked Bar
  IF datetime + numeric → Line/Area Chart
  IF 3+ numeric → Parallel Coordinates / 3D Scatter
  ```
- **Files:**
  - `[NEW] app/modules/chart_recommender.py`
  - `[MODIFY] app/main.py` — integrate recommendations above chart selector

#### 6.2.2 NL-to-Chart (Text2Viz)
- **Priority:** P2 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** Type natural language like "show salary distribution by gender as violin plot" and get the chart rendered automatically.
- **Why:** This is the holy grail of no-code analytics — true natural-language-to-visualization.
- **Tech Stack:** LLM Router + Plotly code execution (sandboxed)
- **UI Spec:**
  - Text input at top of Visualize tab: "🗣️ Describe the chart you want..."
  - Example prompts shown as clickable chips: "salary by department", "age distribution", "GPA vs income scatter"
  - Generated chart displayed below with the Plotly code in an expander
  - Error handling: if LLM generates bad code, show friendly error + suggest manual creation
- **Safety:**
  - Sandbox LLM-generated code: only allow `plotly`, `pandas` operations on the loaded DataFrame
  - No `exec()` of arbitrary code — use AST parsing to validate generated code
  - Whitelist allowed function calls
- **Files:**
  - `[NEW] app/modules/text2viz.py`
  - `[MODIFY] app/main.py` — add NL input field above Visualize tab content

#### 6.2.3 Smart Correlation Heatmap (Toggle Methods)
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 8/10
- *(Shared with Overview tab — see Section 5.2.2)*

#### 6.2.4 Animated Time-Series
- **Priority:** P2 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Plotly animated bar race charts, area charts with play/pause slider for temporal data.
- **Why:** Animated visualizations are more engaging and reveal temporal patterns that static charts miss.
- **Tech Stack:** `plotly.express` animation frames
- **UI Spec:**
  - New chart option under Time Series: "Animated Bar Race" and "Animated Area"
  - Play/pause controls with speed slider
  - Frame-by-frame navigation
- **Files:**
  - `[MODIFY] app/modules/viz_engine.py` — add `create_animated_bar_race()`, `create_animated_area()`
  - `[MODIFY] app/config.py` — add new chart types to `TIME_SERIES_CHARTS`

#### 6.2.5 Geospatial Mapping
- **Priority:** P3 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** If dataset has city/country/lat-lon columns, auto-render choropleth and scatter maps.
- **Why:** Spatial data is increasingly common; no-code geo-visualization is high-value.
- **Tech Stack:** `plotly.express` geo, `pycountry`, `geopy`
- **UI Spec:**
  - Auto-detect geo columns (lat/lon, country, city, state, zip)
  - Map type selector: Choropleth, Scatter Map, Bubble Map
  - Color by any numeric column
  - Zoom controls + tooltip on hover
- **Files:**
  - `[NEW] app/modules/geo_engine.py`
  - `[MODIFY] app/modules/viz_engine.py` — add geo chart functions
  - `[MODIFY] app/config.py` — add "Geospatial" analysis type

#### 6.2.6 Chart Story Mode
- **Priority:** P3 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** Convert a sequence of charts into a narrated data story with AI-written transitions.
- **Why:** Data storytelling is how insights become action. Turning charts into narratives bridges the analyst-stakeholder gap.
- **Tech Stack:** LLM Router, Streamlit carousel/tabs
- **UI Spec:**
  - Button: "📖 Create Data Story" after generating 2+ charts
  - AI generates narrative transitions between each chart
  - Rendered as a scrollable story with charts + text interleaved
  - Export as HTML story page
- **Files:**
  - `[NEW] app/modules/story_generator.py`
  - `[NEW] app/components/story_view.py`

#### 6.2.7 Sankey / Flow Diagrams
- **Priority:** P3 | **Difficulty:** Easy | **Innovation:** 7/10
- **What:** For categorical flow data, auto-generate Sankey diagrams showing how data flows between categories.
- **Tech Stack:** `plotly.graph_objects.Sankey`
- **UI Spec:**
  - New chart option under Multivariate: "Sankey Diagram"
  - Select source column, target column, and optional value column
  - Color-coded flow paths
- **Files:**
  - `[MODIFY] app/modules/viz_engine.py` — add `create_sankey()`
  - `[MODIFY] app/config.py` — add to `MULTIVARIATE_CHARTS`

---

## 7. Tab 3: AI Copilot — Specifications

### 7.1 Current State

| Feature | Status | Component |
|---------|--------|-----------|
| Groq LLM chat (llama-3.3-70b) | ✅ Working | `ai_engine.py` + `ai_chat.py` |
| Dataset context builder | ✅ Working | `ai_engine.py` |
| Rule-based fallback | ✅ Working | `ai_engine.py` |
| Quick-action buttons | ✅ Working | `ai_chat.py` |
| Chart narrative generation | ✅ Working | `ai_engine.py` |
| Executive summary generation | ✅ Working | `ai_engine.py` |

### 7.2 Upgrade Features

#### 7.2.1 RAG-Powered Data Memory
- **Priority:** P1 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** Embed dataset rows + metadata into ChromaDB vectors. AI retrieves relevant rows before answering, enabling true data-aware responses instead of summary-only context.
- **Why:** Current AI only sees a truncated JSON context — it cannot answer questions about specific rows, values, or edge cases.
- **Tech Stack:** `chromadb`, `sentence-transformers` (all-MiniLM-L6-v2)
- **Implementation:**
  ```python
  # app/modules/rag_engine.py
  class RAGEngine:
      def __init__(self, df: pd.DataFrame):
          self.collection = chromadb.Client().create_collection("dataset")
          self._embed_dataset(df)
      
      def _embed_dataset(self, df: pd.DataFrame):
          """Embed rows as text chunks + column metadata."""
          # Chunk strategy: every 10 rows → 1 document
          # Also embed: column descriptions, data types, statistics
      
      def query(self, question: str, n_results: int = 5) -> list:
          """Retrieve most relevant data chunks for a question."""
      
      def build_augmented_prompt(self, question: str, context: dict) -> str:
          """Combine RAG results with existing context for LLM."""
  ```
- **UI Impact:**
  - No visible UI change — RAG is transparent to user
  - AI responses become more accurate and data-specific
  - Status indicator: "🧠 RAG Active" in chat header
- **Files:**
  - `[NEW] app/modules/rag_engine.py`
  - `[MODIFY] app/modules/ai_engine.py` — integrate RAG into `chat_with_data()`
  - `[MODIFY] app/components/ai_chat.py` — show RAG status

#### 7.2.2 Code Generation Mode
- **Priority:** P1 | **Difficulty:** Hard | **Innovation:** 9/10
- **What:** AI generates runnable Python/pandas code alongside answers. User can copy and execute in their own environment.
- **Why:** Bridges the GUI-to-code gap for users who want to learn or reproduce results programmatically.
- **Tech Stack:** LLM Router, code formatting
- **UI Spec:**
  - Toggle in chat settings: "🐍 Show Code"
  - When enabled, AI appends a code block with the equivalent pandas/plotly code
  - "📋 Copy Code" button on each code block
  - Code is syntax-highlighted with `st.code()`
- **Files:**
  - `[MODIFY] app/modules/ai_engine.py` — add code generation to system prompt
  - `[MODIFY] app/components/ai_chat.py` — render code blocks

#### 7.2.3 Multi-Model Toggle
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Switch between Groq (llama-3.3-70b), Google Gemini Flash, and local Ollama models from a dropdown in the chat.
- **Why:** Single-provider lock-in is a reliability risk. Different models have different strengths.
- **Tech Stack:** LLM Router (see Section 4.2)
- **UI Spec:**
  - Dropdown at top of AI Copilot tab: "🤖 Model: [Groq llama-3.3-70b ▼]"
  - Options: Groq, Gemini Flash, Ollama (local), Auto (fallback chain)
  - Model info tooltip showing capabilities and speed
  - Persist selection in session state
- **Files:**
  - `[NEW] app/modules/llm_router.py` (shared system-wide)
  - `[MODIFY] app/components/ai_chat.py` — add model selector

#### 7.2.4 AI Chart-from-Chat
- **Priority:** P2 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** User asks "plot salary vs GPA" in chat → AI generates and renders the chart inline within the chat interface.
- **Why:** Eliminates context-switching between Chat and Visualize tabs.
- **Tech Stack:** LLM Router + Plotly code execution (sandboxed)
- **UI Spec:**
  - AI detects visualization intent from natural language
  - Generates Plotly code and renders the chart within the chat message
  - Chart is interactive (zoom, hover) within the chat
  - "📊 Open in Visualize Tab" button to expand
- **Files:**
  - `[MODIFY] app/modules/ai_engine.py` — add chart intent detection
  - `[MODIFY] app/components/ai_chat.py` — render Plotly figures inline

#### 7.2.5 Persistent Memory (Sessions)
- **Priority:** P2 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Store chat history + dataset context across sessions for returning users.
- **Why:** Current chat resets on every page refresh.
- **Tech Stack:** `st.session_state` + optional SQLite/JSON file persistence
- **UI Spec:**
  - Chat history persists across tab switches (already works)
  - "💾 Save Chat" button to export chat as JSON
  - "📂 Load Previous Chat" to import
  - Auto-persist last 100 messages in session
- **Files:**
  - `[MODIFY] app/components/ai_chat.py` — add save/load buttons
  - `[MODIFY] app/modules/session_manager.py` — chat persistence methods

#### 7.2.6 Agentic Analysis Mode
- **Priority:** P3 | **Difficulty:** Very Hard | **Innovation:** 10/10
- **What:** User says "analyze this dataset for me" — agent autonomously plans and executes a multi-step analysis (EDA → cleaning → modeling → report).
- **Why:** True AI agent behavior — not just Q&A but autonomous task execution.
- **Tech Stack:** LLM Router with multi-step planning (lightweight agent loop)
- **UI Spec:**
  - Button: "🤖 Auto-Analyze Dataset"
  - Shows progress steps: "Step 1/5: Profiling data... Step 2/5: Detecting issues..."
  - Each step generates a finding with expandable details
  - Final summary with key insights and recommendations
- **Files:**
  - `[NEW] app/modules/agent_analyzer.py`
  - `[MODIFY] app/components/ai_chat.py` — add auto-analyze button

---

## 8. Tab 4: Statistics — Specifications

### 8.1 Current State

| Feature | Status | Component |
|---------|--------|-----------|
| Test Recommender (Mann-Whitney, ANOVA, Chi-Square) | ✅ Working | `stats_engine.py` |
| Distribution Analyzer (Shapiro-Wilk, Q-Q) | ✅ Working | `stats_engine.py` + `stats_panel.py` |
| Hypothesis Builder (no-code) | ✅ Working | `stats_engine.py` + `stats_panel.py` |

### 8.2 Upgrade Features

#### 8.2.1 Effect Size Calculator
- **Priority:** P1 | **Difficulty:** Easy | **Innovation:** 9/10
- **What:** Alongside p-values, always show Cohen's d, eta-squared, or Cramer's V with practical significance labels (small/medium/large).
- **Why:** P-values alone are misleading — effect size tells you if the finding is practically meaningful.
- **Tech Stack:** `pingouin`, `scipy`
- **UI Spec:**
  - New metric card next to P-Value: "Effect Size: Cohen's d = 0.45 (Medium)"
  - Color-coded: Green (large), Orange (medium), Red (small/negligible)
  - Tooltip explaining what the effect size means in context
- **Effect Size Selection Logic:**
  ```
  2 numeric groups → Cohen's d
  3+ groups (ANOVA) → Eta-squared (η²)
  2 categorical → Cramér's V
  Correlation → r (already available)
  ```
- **Files:**
  - `[MODIFY] app/modules/stats_engine.py` — add `compute_effect_size()` to each test
  - `[MODIFY] app/components/stats_panel.py` — display effect size in UI

#### 8.2.2 AI Statistical Narrator
- **Priority:** P1 | **Difficulty:** Easy | **Innovation:** 10/10
- **What:** After every statistical test, generate a plain-English paragraph explaining results to a non-statistician.
- **Why:** Most users can't interpret p-values, effect sizes, or assumption violations. AI narration makes stats accessible.
- **Tech Stack:** LLM Router
- **UI Spec:**
  - Collapsible panel below each test result: "📝 Plain English Explanation"
  - Example: "The Mann-Whitney U test found a statistically significant difference in exam scores between male and female students (U = 1234, p = 0.003). The effect size (d = 0.62) is medium, meaning this difference is practically meaningful. You can confidently say that gender is associated with exam performance in this dataset."
  - Fallback: template-based narration when LLM unavailable
- **Files:**
  - `[NEW] app/modules/stats_narrator.py`
  - `[MODIFY] app/components/stats_panel.py` — call narrator after each test

#### 8.2.3 Full Regression Suite
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** OLS, Logistic, Ridge, Lasso, Poisson regression with residual diagnostics and assumption checks.
- **Why:** Regression is the workhorse of statistics and the current app has none in the Statistics tab.
- **Tech Stack:** `statsmodels`, `scikit-learn`
- **UI Spec:**
  - New sub-tab in Statistics: "📈 Regression Analysis"
  - Select dependent variable, independent variables, regression type
  - Results: coefficient table, R², adjusted R², F-statistic, p-values
  - Diagnostic plots: residuals vs fitted, Q-Q of residuals, scale-location, Cook's distance
  - VIF computed for all predictors
- **Files:**
  - `[NEW] app/modules/regression_engine.py`
  - `[NEW] app/components/regression_panel.py`
  - `[MODIFY] app/components/stats_panel.py` — add 4th tab for regression

#### 8.2.4 Power Analysis Tool
- **Priority:** P2 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Given desired effect size and alpha, compute required sample size — critical for study design.
- **Why:** Essential for researchers planning experiments. Tells them if their sample is large enough.
- **Tech Stack:** `statsmodels.stats.power`
- **UI Spec:**
  - Input: Effect size (with presets: small/medium/large), Alpha (0.05 default), Power (0.80 default)
  - Output: Required sample size per group
  - Interactive plot: power curve showing how sample size affects power
  - Reverse mode: given current sample size, what's the achievable power?
- **Files:**
  - `[NEW] app/modules/power_analysis.py`
  - `[MODIFY] app/components/stats_panel.py` — add power analysis sub-tab

#### 8.2.5 Multiple Comparison Correction
- **Priority:** P2 | **Difficulty:** Easy | **Innovation:** 8/10
- **What:** When running many tests, apply Bonferroni or Benjamini-Hochberg FDR correction and show adjusted p-values.
- **Why:** Running multiple tests inflates false positive rate. Correction is mandatory for rigorous analysis.
- **Tech Stack:** `statsmodels.stats.multitest`
- **UI Spec:**
  - Option in Test Recommender: "Apply Multiple Comparison Correction"
  - Dropdown: Bonferroni, Holm, Benjamini-Hochberg (FDR)
  - Show both raw and adjusted p-values
  - Highlight which results survive correction
- **Files:**
  - `[MODIFY] app/modules/stats_engine.py` — add correction functions
  - `[MODIFY] app/components/stats_panel.py` — add correction UI

#### 8.2.6 Bayesian Inference Module
- **Priority:** P3 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** Compute Bayesian A/B test results with posterior distributions, credible intervals, and probability of superiority.
- **Why:** Bayesian methods provide richer inference than frequentist p-values. Increasingly demanded in industry and academia.
- **Tech Stack:** `PyMC` (or lightweight conjugate priors), `arviz`
- **UI Spec:**
  - New sub-tab: "🔮 Bayesian Analysis"
  - Select two groups and metric
  - Output: posterior distribution plots, 95% HDI (Highest Density Interval), P(A > B)
  - Visual: overlapping posterior curves with shaded credible intervals
- **Note:** PyMC is heavy (~200MB). Consider lightweight conjugate prior implementation for Streamlit Cloud compatibility.
- **Files:**
  - `[NEW] app/modules/bayesian_engine.py`
  - `[NEW] app/components/bayesian_panel.py`

#### 8.2.7 Time-Series Statistics
- **Priority:** P2 | **Difficulty:** Hard | **Innovation:** 9/10
- **What:** ACF/PACF plots, stationarity tests (ADF, KPSS), seasonal decomposition, Granger causality.
- **Why:** Current time-series module only has visualization — no statistical testing for time-series properties.
- **Tech Stack:** `statsmodels`, `pmdarima` (optional)
- **UI Spec:**
  - New sub-tab: "⏱️ Time-Series Statistics"
  - ACF/PACF plots with significance bands
  - ADF test result: "Series is stationary (p = 0.02)" / "Series is non-stationary"
  - Seasonal decomposition (trend, seasonal, residual)
  - Granger causality test between two time-series columns
- **Files:**
  - `[NEW] app/modules/ts_stats_engine.py`
  - `[MODIFY] app/components/stats_panel.py` — add time-series statistics sub-tab

#### 8.2.8 Causal Inference Module
- **Priority:** P3 | **Difficulty:** Very Hard | **Innovation:** 10/10
- **What:** Estimate causal effects using propensity score matching, instrumental variables, and difference-in-differences.
- **Why:** Correlation ≠ causation. Causal inference separates DataPilot-AI from basic EDA tools.
- **Tech Stack:** `DoWhy`, `econml`
- **Note:** Consider as a future phase (v3.1) due to complexity and dependency weight.
- **Files:**
  - `[NEW] app/modules/causal_engine.py`
  - `[NEW] app/components/causal_panel.py`

---

## 9. Tab 5: Clean — Specifications

### 9.1 Current State

| Feature | Status | Component |
|---------|--------|-----------|
| Missing value imputation (mean/median/mode/ffill/bfill/zero/drop) | ✅ Working | `cleaner.py` |
| Duplicate row removal | ✅ Working | `cleaner.py` |
| Constant column detection | ✅ Working | `cleaner.py` |
| Outlier handling (IQR clip, zscore clip, winsorize, IQR drop) | ✅ Working | `cleaner.py` |
| Categorical encoding (onehot, label, frequency) | ✅ Working | `cleaner.py` |
| Feature scaling (minmax, zscore, robust) | ✅ Working | `cleaner.py` |
| Before/after comparison | ✅ Working | `clean_panel.py` |
| CSV/Excel download | ✅ Working | `clean_panel.py` |

### 9.2 Upgrade Features

#### 9.2.1 ML-Based Smart Imputation
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Use KNNImputer, IterativeImputer (MICE), or missForest for smarter missing value imputation that considers inter-column relationships.
- **Why:** Simple mean/median imputation destroys variance and relationships. ML imputation preserves data structure.
- **Tech Stack:** `scikit-learn.impute.KNNImputer`, `scikit-learn.impute.IterativeImputer`
- **UI Spec:**
  - New options in missing value strategy dropdown: "KNN Imputer", "MICE (Iterative)", "Most Frequent (ML)"
  - Settings: K neighbors (for KNN), Max iterations (for MICE)
  - Show imputation quality metrics: distribution comparison before/after
- **Files:**
  - `[MODIFY] app/modules/cleaner.py` — add ML imputation methods
  - `[MODIFY] app/components/clean_panel.py` — add new options to dropdown

#### 9.2.2 Cleaning Audit Trail
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Every operation is logged with timestamp, operation type, columns affected, and rows changed. Supports undo.
- **Why:** Data cleaning without an audit trail is unscientific. Users need to track what was changed and revert if needed.
- **Tech Stack:** `st.session_state` diff tracking
- **UI Spec:**
  - Scrollable log panel at bottom of Clean tab
  - Each entry: `[10:32:15] Imputed 'Age' with median (47 values changed)`
  - "↩️ Undo Last" button to revert the most recent operation
  - "📋 Export Cleaning Log" as CSV
  - Maintain history stack of DataFrame snapshots (last 5)
- **Files:**
  - `[NEW] app/modules/audit_trail.py`
  - `[MODIFY] app/components/clean_panel.py` — integrate audit trail UI

#### 9.2.3 Anomaly Detection (Advanced)
- **Priority:** P2 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Use Isolation Forest, LOF, or Autoencoder to detect anomalous rows beyond simple IQR/z-score.
- **Why:** IQR only works for univariate outliers. Multivariate anomalies require ML-based detection.
- **Tech Stack:** `scikit-learn` (IsolationForest, LocalOutlierFactor), `PyOD` (optional)
- **UI Spec:**
  - New section: "🔍 Advanced Anomaly Detection"
  - Algorithm selector: Isolation Forest, LOF, Z-Score (ensemble)
  - Contamination parameter slider (0.01 - 0.20)
  - Results: highlighted anomalous rows with anomaly scores
  - Options: "Remove Anomalies", "Flag Anomalies (add column)", "Export Anomalies"
- **Files:**
  - `[NEW] app/modules/anomaly_detector.py`
  - `[MODIFY] app/components/clean_panel.py` — add anomaly detection section

#### 9.2.4 Smart Column Type Fixer
- **Priority:** P1 | **Difficulty:** Easy | **Innovation:** 8/10
- **What:** Auto-detect and fix wrong types: string numbers, date strings, boolean-as-int, currency-as-string.
- **Why:** Real-world data has "$1,234" as strings, "TRUE"/"FALSE" as text, dates as "Jan 1 2024".
- **Tech Stack:** `pandas`, `dateutil`, `re`
- **UI Spec:**
  - Auto-scan: show detected type issues
  - Table: Column | Current Type | Suggested Type | Sample Values | Fix? (checkbox)
  - "Apply Fixes" button
  - Handle: currency symbols, thousand separators, boolean variants, date strings
- **Files:**
  - `[NEW] app/modules/type_fixer.py`
  - `[MODIFY] app/components/clean_panel.py` — add type fixer section

#### 9.2.5 Fuzzy Deduplication
- **Priority:** P2 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Detect near-duplicate rows using text similarity (Jaccard, Levenshtein) for string columns.
- **Why:** Exact deduplication misses "New York" vs "new york" vs "NYC". Fuzzy matching catches these.
- **Tech Stack:** `rapidfuzz`, `recordlinkage`
- **UI Spec:**
  - Similarity threshold slider (0.7 - 1.0)
  - Select columns to compare
  - Show grouped near-duplicates with similarity scores
  - Options: merge, keep first, keep last, manual review
- **Files:**
  - `[NEW] app/modules/fuzzy_dedup.py`
  - `[MODIFY] app/components/clean_panel.py` — add fuzzy dedup section

#### 9.2.6 Data Contract Validator
- **Priority:** P3 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Let users define column constraints (e.g., `age > 0 AND age < 120`) and validate data against them.
- **Why:** Data contracts ensure data quality is maintained. Critical for production pipelines.
- **Tech Stack:** `pandera` or custom validation engine
- **UI Spec:**
  - Rule builder: Column | Constraint Type (range, regex, not-null, unique, in-set) | Value
  - Add multiple rules
  - "✅ Validate" button → show pass/fail per rule with violation counts
  - Export: "📥 Download Validation Report"
- **Files:**
  - `[NEW] app/modules/data_contract.py`
  - `[NEW] app/components/contract_builder.py`

#### 9.2.7 Cleaning Impact Simulator
- **Priority:** P3 | **Difficulty:** Hard | **Innovation:** 9/10
- **What:** After selecting a cleaning operation, preview how it affects downstream ML performance with a quick model train/compare.
- **Why:** Users don't know if dropping rows or imputing is better for their use case.
- **Tech Stack:** `scikit-learn` quick train (RandomForest, 2-fold CV)
- **UI Spec:**
  - After configuring cleaning: "🔮 Simulate ML Impact"
  - Quick RF model trained on original vs cleaned data
  - Show: accuracy delta, feature importance shifts, data size change
  - Recommendation: "Cleaning improved accuracy by +3.2%. Apply these changes."
- **Files:**
  - `[NEW] app/modules/cleaning_simulator.py`
  - `[MODIFY] app/components/clean_panel.py` — add simulation button

---

## 10. Tab 6: ML Studio — Specifications

### 10.1 Current State

| Feature | Status | Component |
|---------|--------|-----------|
| ML Readiness Advisor | ✅ Working | `ml_advisor.py` |
| Train Model (RF, LR/LogReg, XGBoost) | ✅ Working | `ml_advisor.py` + `ml_studio.py` |
| K-Means Clustering | ✅ Working | `ml_advisor.py` + `ml_studio.py` |
| Feature importance bar chart | ✅ Working | `ml_studio.py` |
| Cross-validation scores | ✅ Working | `ml_advisor.py` |
| Prediction download (CSV) | ✅ Working | `ml_studio.py` |

### 10.2 Upgrade Features

#### 10.2.1 Extended Algorithm Library
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Add SVM, KNN, LightGBM, CatBoost, Decision Tree, Naive Bayes, ElasticNet, and Gradient Boosting to the algorithm selector.
- **Tech Stack:** `scikit-learn`, `lightgbm`, `catboost`
- **UI Spec:**
  - Classification: RF, LR, XGBoost, **LightGBM**, **CatBoost**, **SVM**, **KNN**, **Decision Tree**, **Naive Bayes**
  - Regression: RF, Linear, XGBoost, **LightGBM**, **CatBoost**, **SVR**, **KNN**, **Decision Tree**, **ElasticNet**, **Ridge**, **Lasso**
  - Organize in a grid with icons and brief descriptions
- **Files:**
  - `[MODIFY] app/modules/ml_advisor.py` — add new algorithm handlers
  - `[MODIFY] app/components/ml_studio.py` — update algorithm dropdown

#### 10.2.2 SHAP Explainability Dashboard
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 10/10
- **What:** For every trained model, show SHAP beeswarm, waterfall, force, and dependence plots.
- **Why:** Feature importance bar charts don't show direction of impact or individual predictions. SHAP is the gold standard.
- **Tech Stack:** `shap`, `plotly` (for interactive SHAP plots)
- **UI Spec:**
  - Automatically generated after model training
  - Tabs within results: "📊 Metrics | 🔍 SHAP Summary | 🎯 SHAP Waterfall | 📈 Dependence"
  - Summary (beeswarm): shows all features' impact direction
  - Waterfall: explain a single prediction
  - Dependence: select a feature → see its SHAP value vs actual value
  - "📥 Download SHAP Report" as HTML
- **Files:**
  - `[NEW] app/modules/shap_explainer.py`
  - `[MODIFY] app/components/ml_studio.py` — add SHAP visualization tabs

#### 10.2.3 Model Comparison Arena
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Train multiple models simultaneously, compare metrics in a leaderboard, plot ROC curves side-by-side.
- **Why:** Users currently train one model at a time with no way to compare.
- **Tech Stack:** `scikit-learn`, `plotly`
- **UI Spec:**
  - Multi-select algorithms: "Select models to compare: [RF ☑] [XGB ☑] [LR ☑] [SVM ☐]"
  - Results: leaderboard table (sorted by primary metric)
  - ROC curves overlaid on single plot (classification)
  - Residual comparison plots (regression)
  - "🏆 Best Model: XGBoost (Accuracy: 0.94)"
  - Download: all predictions + comparison table CSV
- **Files:**
  - `[NEW] app/modules/model_arena.py`
  - `[MODIFY] app/components/ml_studio.py` — add comparison arena tab

#### 10.2.4 Hyperparameter Tuning (HPO)
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Allow users to tune hyperparameters using Optuna or GridSearchCV with a visual progress tracker.
- **Why:** Default parameters rarely give best results. HPO is essential for serious modeling.
- **Tech Stack:** `optuna` or `scikit-learn.model_selection.GridSearchCV`
- **UI Spec:**
  - "⚡ Enable Hyperparameter Tuning" checkbox
  - Options: Quick (20 trials), Standard (50 trials), Thorough (100 trials)
  - Progress bar with current best score
  - Results: best parameters table, optimization history plot
  - Apply best parameters to retrain
- **Files:**
  - `[NEW] app/modules/hpo_engine.py`
  - `[MODIFY] app/components/ml_studio.py` — add HPO UI

#### 10.2.5 What-If Analysis Tool
- **Priority:** P2 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Manually adjust feature values and see real-time prediction changes — interactive sensitivity analysis.
- **Why:** Understanding how changing input features affects predictions is crucial for decision-making.
- **Tech Stack:** Streamlit sliders/inputs + `model.predict()`
- **UI Spec:**
  - After training a model: "🔮 What-If Analysis" tab
  - Sliders for each numeric feature, dropdowns for categorical
  - Real-time prediction display with confidence (if classification)
  - Feature importance waterfall for the custom input
  - Compare: "Current Prediction: Class A (87%) → After change: Class B (65%)"
- **Files:**
  - `[NEW] app/components/whatif_panel.py`
  - `[MODIFY] app/components/ml_studio.py` — add What-If tab

#### 10.2.6 AutoML Pipeline
- **Priority:** P2 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** One-click AutoML: automatically try 10+ algorithms, tune hyperparameters, and present a ranked leaderboard.
- **Why:** The ultimate no-code ML experience — user just picks the target column and clicks "Go".
- **Tech Stack:** `FLAML` (lightweight, Streamlit-friendly) or custom pipeline
- **UI Spec:**
  - Big button: "🚀 AutoML — Find Best Model"
  - Time budget selector: 1min, 5min, 15min
  - Real-time progress: current algorithm, current score, best so far
  - Final: ranked leaderboard with all tried models + metrics
  - Download: best model as pickle, all results as CSV
- **Note:** `FLAML` is preferred over `AutoSklearn` because it's lighter and cross-platform.
- **Files:**
  - `[NEW] app/modules/automl_engine.py`
  - `[MODIFY] app/components/ml_studio.py` — add AutoML sub-tab

#### 10.2.7 Cluster Profiler
- **Priority:** P2 | **Difficulty:** Medium | **Innovation:** 10/10
- **What:** After K-Means, auto-generate cluster personas with dominant feature values and natural-language descriptions.
- **Why:** Raw cluster labels are meaningless. Profiling tells users *what* each cluster represents.
- **Tech Stack:** LLM Router, `pandas.groupby()`
- **UI Spec:**
  - After clustering: "📋 Cluster Profiles" section
  - For each cluster: dominant features, average values, distinguishing characteristics
  - AI-generated persona: "Cluster 0 (High Performers): Above-average GPA (3.7), young (avg age 22), primarily STEM majors"
  - Radar chart comparing cluster centroids
- **Files:**
  - `[NEW] app/modules/cluster_profiler.py`
  - `[MODIFY] app/components/ml_studio.py` — add profiler to clustering tab

#### 10.2.8 Model Download & Persistence
- **Priority:** P1 | **Difficulty:** Easy | **Innovation:** 7/10
- **What:** Download trained models as pickle/joblib files for reuse outside DataPilot-AI.
- **Tech Stack:** `joblib`, `pickle`
- **UI Spec:**
  - After training: "📥 Download Model (.joblib)" button
  - Include: model object, feature names, training metadata, preprocessing pipeline
- **Files:**
  - `[MODIFY] app/modules/ml_advisor.py` — add model serialization
  - `[MODIFY] app/components/ml_studio.py` — add download button

---

## 11. Tab 7: Export — Specifications

### 11.1 Current State

| Feature | Status | Component |
|---------|--------|-----------|
| PDF Report (fpdf2) | ✅ Working | `report_generator.py` |
| HTML Report | ✅ Working | `report_generator.py` |
| Jupyter Notebook (.ipynb) | ✅ Working | `notebook_exporter.py` |
| Data Export (CSV, Excel, JSON) | ✅ Working | `report_panel.py` |
| Statistics Export (CSV) | ✅ Working | `report_panel.py` |
| AI Executive Summary in reports | ✅ Working | `ai_engine.py` |

### 11.2 Upgrade Features

#### 11.2.1 Interactive HTML Dashboard Export
- **Priority:** P1 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** Export full analysis as self-contained HTML with embedded Plotly charts, filters, and navigation — a mini-dashboard.
- **Why:** Current HTML report is static. Interactive HTML is shareable without any server.
- **Tech Stack:** `plotly`, `jinja2`, Bootstrap
- **UI Spec:**
  - Button: "🌐 Generate Interactive Dashboard"
  - Output: single .html file with:
    - Tab navigation (Overview, Charts, Stats, ML Results)
    - Embedded Plotly charts (fully interactive)
    - Search/filter on data tables
    - Trust Score visualization
    - AI summary section
  - File size optimization: compress Plotly bundles
- **Files:**
  - `[NEW] app/modules/dashboard_exporter.py`
  - `[MODIFY] app/components/report_panel.py` — add dashboard export button

#### 11.2.2 PowerPoint Auto-Deck Generator
- **Priority:** P2 | **Difficulty:** Hard | **Innovation:** 10/10
- **What:** 1-click generate a presentation with key findings, charts, and AI narrative per slide.
- **Why:** Business users present findings in PowerPoint. Auto-generating saves hours.
- **Tech Stack:** `python-pptx`, LLM Router
- **UI Spec:**
  - Button: "📊 Generate PowerPoint Deck"
  - Options: slide count (5/10/15), theme (dark/light/corporate)
  - Slides: Title, Executive Summary, Data Overview, Key Charts (top 3-5), Statistics Highlights, ML Results, Recommendations, Appendix
  - Each slide has AI-written bullet points
  - Charts exported as high-res images embedded in slides
- **Files:**
  - `[NEW] app/modules/pptx_exporter.py`
  - `[MODIFY] app/components/report_panel.py` — add PPT export button

#### 11.2.3 Rich Jupyter Notebook
- **Priority:** P1 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Enhanced notebook with full markdown narrative, SHAP plots, feature importance, model comparison — not just code cells.
- **Why:** Current notebook is minimal code-only. A rich notebook is publication-ready.
- **Tech Stack:** `nbformat`, `jinja2`
- **UI Spec:**
  - Toggle: "📓 Rich Mode (with narrative)" vs "📝 Code Only"
  - Rich mode adds:
    - Markdown cells with AI-written analysis narrative
    - Section headers with dividers
    - Inline SHAP code and visualization cells
    - Comparison tables for multi-model results
    - Data quality summary markdown
- **Files:**
  - `[MODIFY] app/modules/notebook_exporter.py` — add rich mode
  - `[MODIFY] app/components/report_panel.py` — add rich mode toggle

#### 11.2.4 Data Dictionary Export
- **Priority:** P1 | **Difficulty:** Easy | **Innovation:** 8/10
- **What:** Auto-generate a Word/PDF data dictionary with column descriptions, types, stats, value ranges.
- **Why:** Data dictionaries are required for any serious data project. Auto-generating saves hours.
- **Tech Stack:** `python-docx`, `pandas`
- **UI Spec:**
  - Button: "📖 Generate Data Dictionary"
  - Output: Word document (.docx) with:
    - Dataset overview (source, size, date)
    - Column table: Name, Type, Description (AI-generated), Non-null count, Unique values, Sample values
    - Data quality summary
    - Trust Score breakdown
- **Files:**
  - `[NEW] app/modules/data_dictionary.py`
  - `[MODIFY] app/components/report_panel.py` — add dictionary export

#### 11.2.5 MLflow Experiment Logging
- **Priority:** P3 | **Difficulty:** Medium | **Innovation:** 9/10
- **What:** Integrate MLflow to track all model experiments, parameters, and metrics persistently.
- **Why:** Model experiments without tracking are lost. MLflow is the industry standard.
- **Tech Stack:** `mlflow`
- **UI Spec:**
  - Checkbox in ML Studio: "📊 Log to MLflow"
  - Auto-log: model type, hyperparameters, metrics, feature importance, SHAP summary
  - Link to MLflow UI (if running locally)
  - For cloud deployment: export experiment log as JSON/CSV
- **Files:**
  - `[NEW] app/modules/mlflow_logger.py`
  - `[MODIFY] app/components/ml_studio.py` — integrate MLflow logging

#### 11.2.6 Scheduled Report Delivery
- **Priority:** P3 | **Difficulty:** Medium | **Innovation:** 8/10
- **What:** Configure email delivery of PDF reports on a schedule.
- **Note:** Limited by Streamlit Cloud's serverless nature. Better suited for self-hosted deployment.
- **Tech Stack:** `APScheduler`, `SendGrid` / SMTP
- **Files:**
  - `[NEW] app/modules/report_scheduler.py`
  - `[MODIFY] app/components/report_panel.py` — add scheduling UI

---

## 12. Non-Functional Requirements

### 12.1 Performance

| Metric | Target | Current |
|--------|--------|---------|
| Initial page load time | < 3s | ~2s ✅ |
| File parsing (100K rows CSV) | < 5s | ~3s ✅ |
| Trust Score computation | < 2s | ~1s ✅ |
| AI Copilot response time | < 5s | ~3s ✅ |
| ML model training (100K rows, RF) | < 30s | ~15s ✅ |
| Report generation (PDF) | < 10s | ~5s ✅ |
| Maximum dataset size | 500MB / 2M rows | 500MB ✅ |

### 12.2 Compatibility

| Platform | Requirement |
|----------|-------------|
| Streamlit Cloud | Full compatibility, 1GB RAM limit respected |
| Hugging Face Spaces | Full compatibility, 16GB disk limit respected |
| Local development | Python 3.11+, all features available |
| Browser support | Chrome 90+, Firefox 88+, Safari 14+, Edge 90+ |

### 12.3 Security

| Requirement | Implementation |
|-------------|---------------|
| API keys stored securely | `st.secrets` (cloud) / `.env` (local) |
| No data leaves the server | All processing server-side; LLM calls send only metadata/samples |
| PII warning system | Trust Score + GDPR scanner flags sensitive columns |
| No arbitrary code execution | NL-to-Chart uses AST validation, not `exec()` |

### 12.4 Accessibility

| Requirement | Implementation |
|-------------|---------------|
| Color-blind safe palettes | Use colorblind-friendly Plotly themes |
| Screen reader compatibility | Semantic HTML in custom components |
| Keyboard navigation | Streamlit-native keyboard support |
| Mobile responsiveness | Limited by Streamlit — optimize layout for tablets |

---

## 13. Technology Stack & Dependencies

### 13.1 Core Dependencies (requirements.txt)

```
# ── Core Framework ──────────────────────────────────────────────
streamlit>=1.31.0

# ── Data Processing ─────────────────────────────────────────────
pandas>=2.0.0
numpy>=1.26.0
pyarrow>=14.0.0
chardet>=5.2.0

# ── Visualization ────────────────────────────────────────────────
plotly==5.20.0
kaleido>=0.2.1

# ── Statistical Analysis ─────────────────────────────────────────
scipy>=1.12.0
statsmodels>=0.14.0
pingouin>=0.5.3                    # [NEW] Effect sizes, advanced stats

# ── Machine Learning ─────────────────────────────────────────────
scikit-learn>=1.4.0
xgboost>=2.0.3
lightgbm>=4.3.0
catboost>=1.2.0                    # [NEW] CatBoost algorithm
imbalanced-learn>=0.12.0
shap>=0.43.0                      # [NEW] SHAP explainability
flaml>=2.1.0                      # [NEW] AutoML
optuna>=3.5.0                     # [NEW] Hyperparameter optimization

# ── AI / LLM Integration ─────────────────────────────────────────
groq>=0.5.0
google-generativeai>=0.4.0        # [NEW] Gemini support
chromadb>=0.4.0                   # [NEW] RAG vector store
sentence-transformers>=2.3.0      # [NEW] Embedding model for RAG

# ── File Format Support ──────────────────────────────────────────
openpyxl>=3.1.0
xlrd>=2.0.0
odfpy>=1.4.1
pdfplumber>=0.10.0
python-docx>=1.0.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# ── Export & Report Generation ───────────────────────────────────
fpdf2>=2.7.0
nbformat>=5.9.0
Pillow>=10.0.0
imageio>=2.33.0
python-pptx>=0.6.21               # [NEW] PowerPoint export
jinja2>=3.1.0                     # [NEW] HTML template engine

# ── Data Quality & Cleaning ──────────────────────────────────────
rapidfuzz>=3.5.0                  # [NEW] Fuzzy deduplication
pandera>=0.18.0                   # [NEW] Data contract validation

# ── Utilities ────────────────────────────────────────────────────
python-dateutil>=2.8.0
joblib>=1.3.0                     # [NEW] Model serialization
```

### 13.2 Optional / Heavy Dependencies (separate extras)

```
# ── Optional: Bayesian Analysis (heavy) ──────────────────────────
# pymc>=5.10.0                    # ~200MB, may not fit on Streamlit Cloud
# arviz>=0.17.0                   # Bayesian visualization

# ── Optional: Causal Inference ───────────────────────────────────
# dowhy>=0.11.0
# econml>=0.14.0

# ── Optional: Advanced NER for PII ───────────────────────────────
# spacy>=3.7.0
# presidio-analyzer>=2.2.0

# ── Optional: Geospatial ─────────────────────────────────────────
# pycountry>=22.3.0
# geopy>=2.4.0

# ── Optional: MLflow ─────────────────────────────────────────────
# mlflow>=2.10.0
```

### 13.3 Dependency Budget (Streamlit Cloud / HF Spaces)

| Category | Estimated Size | Status |
|----------|----------------|--------|
| Core (streamlit, pandas, numpy, plotly) | ~300MB | ✅ Within limits |
| ML (sklearn, xgboost, lightgbm, shap) | ~400MB | ✅ Within limits |
| AI (groq, chromadb, sentence-transformers) | ~500MB | ⚠️ Tight — use lazy loading |
| Export (fpdf2, python-pptx, jinja2) | ~50MB | ✅ Small |
| **Total estimated** | **~1.2GB** | ⚠️ May need optimization |

**Mitigation:**
- Use lazy imports: `import X` only when the feature is used
- Consider `sentence-transformers` lightweight model (all-MiniLM-L6-v2 = 80MB)
- ChromaDB in-memory mode (no persistence needed for Streamlit)
- Drop CatBoost if size is critical (XGBoost + LightGBM cover similar ground)

---

## 14. Deployment Strategy

### 14.1 Streamlit Cloud

```yaml
# .streamlit/config.toml
[server]
maxUploadSize = 500
maxMessageSize = 500

[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
font = "sans serif"
```

**Deployment Steps:**
1. Push to `main` branch on GitHub
2. Connect repo to Streamlit Cloud
3. Set secrets in Streamlit Cloud dashboard (GROQ_API_KEY, GEMINI_API_KEY)
4. Entry point: `streamlit run app/main.py`

### 14.2 Hugging Face Spaces

```yaml
# Hugging Face Spaces configuration
title: DataPilot-AI
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app/main.py
pinned: true
```

**Deployment Steps:**
1. Create Space on Hugging Face (Streamlit SDK)
2. Push code to HF Space repo
3. Set secrets in Space settings
4. Monitor logs for dependency installation

### 14.3 Local Development

```bash
# 1. Clone
git clone https://github.com/MansoorSarookh/DataPilot-AI.git
cd DataPilot-AI

# 2. Create virtual environment
python -m venv datapilot_venv
datapilot_venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
# Create .streamlit/secrets.toml with API keys

# 5. Run
streamlit run app/main.py
```

---

## 15. Implementation Roadmap

### Phase 1: Foundation & Quick Wins (Weeks 1-3)

| Task | Tab | Priority | Effort |
|------|-----|----------|--------|
| Code cleanup (remove dead code) | All | P0 | 1 day |
| Robust DateTime parser | System | P0 | 2 days |
| VIF computation | Overview | P1 | 1 day |
| Correlation method toggle | Overview | P1 | 1 day |
| Effect Size Calculator | Statistics | P1 | 2 days |
| AI Statistical Narrator | Statistics | P1 | 2 days |
| Smart Column Type Fixer | Clean | P1 | 2 days |
| ML-Based Smart Imputation | Clean | P1 | 2 days |
| Extended Algorithm Library (5 more) | ML Studio | P1 | 3 days |
| Model Download (.joblib) | ML Studio | P1 | 1 day |

### Phase 2: AI & ML Powerhouse (Weeks 4-6)

| Task | Tab | Priority | Effort |
|------|-----|----------|--------|
| Multi-Model LLM Router | System | P1 | 3 days |
| SHAP Explainability Dashboard | ML Studio | P1 | 3 days |
| Model Comparison Arena | ML Studio | P1 | 3 days |
| Hyperparameter Tuning (Optuna) | ML Studio | P1 | 3 days |
| AI Chart Recommender | Visualize | P1 | 2 days |
| Cleaning Audit Trail | Clean | P1 | 2 days |
| Data Dictionary Export | Export | P1 | 2 days |
| Rich Jupyter Notebook | Export | P1 | 2 days |

### Phase 3: Advanced Analytics (Weeks 7-9)

| Task | Tab | Priority | Effort |
|------|-----|----------|--------|
| RAG-Powered Data Memory | AI Copilot | P1 | 5 days |
| Code Generation Mode | AI Copilot | P1 | 3 days |
| Full Regression Suite | Statistics | P1 | 4 days |
| Interactive HTML Dashboard Export | Export | P1 | 4 days |
| Anomaly Detection (Advanced) | Clean | P2 | 3 days |
| What-If Analysis Tool | ML Studio | P2 | 3 days |
| PowerPoint Auto-Deck Generator | Export | P2 | 3 days |

### Phase 4: Innovation Features (Weeks 10-12)

| Task | Tab | Priority | Effort |
|------|-----|----------|--------|
| NL-to-Chart (Text2Viz) | Visualize | P2 | 5 days |
| AI Chart-from-Chat | AI Copilot | P2 | 4 days |
| AutoML Pipeline | ML Studio | P2 | 4 days |
| Cluster Profiler | ML Studio | P2 | 2 days |
| Power Analysis Tool | Statistics | P2 | 2 days |
| Multiple Comparison Correction | Statistics | P2 | 1 day |
| Animated Time-Series | Visualize | P2 | 2 days |
| Fuzzy Deduplication | Clean | P2 | 3 days |
| Time-Series Statistics | Statistics | P2 | 3 days |
| User Onboarding System | System | P2 | 2 days |

### Phase 5: Advanced & Research Features (Weeks 13-16)

| Task | Tab | Priority | Effort |
|------|-----|----------|--------|
| AI-Powered Column Renaming | Overview | P2 | 2 days |
| Interactive Schema Editor | Overview | P2 | 2 days |
| Dataset Comparison Mode | Overview | P3 | 4 days |
| Semantic Relationship Graph | Overview | P3 | 3 days |
| Geospatial Mapping | Visualize | P3 | 3 days |
| Chart Story Mode | Visualize | P3 | 4 days |
| Sankey Diagrams | Visualize | P3 | 1 day |
| Bayesian Inference Module | Statistics | P3 | 5 days |
| Agentic Analysis Mode | AI Copilot | P3 | 5 days |
| Data Contract Validator | Clean | P3 | 3 days |
| Cleaning Impact Simulator | Clean | P3 | 4 days |
| GDPR Risk Heatmap | Overview | P3 | 4 days |
| Causal Inference Module | Statistics | P3 | 5+ days |
| MLflow Integration | Export | P3 | 3 days |

---

## 16. Risk Assessment & Mitigations

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| R1 | Dependency size exceeds Streamlit Cloud limit | Medium | High | Lazy imports, lightweight alternatives, dependency audit |
| R2 | Groq API rate limits during demo | Medium | Medium | Multi-model fallback, caching, rule-based fallback |
| R3 | ChromaDB memory usage in Streamlit Cloud | Medium | High | In-memory mode, limit vector count, lazy initialization |
| R4 | SHAP computation too slow for large datasets | Medium | Medium | Sample data for SHAP (max 1000 rows), background computation |
| R5 | CatBoost/LightGBM C++ build issues on HF Spaces | Low | High | Pin versions, use pre-built wheels, test in Docker |
| R6 | sentence-transformers model download on cold start | High | Medium | Cache model, use smallest model (all-MiniLM-L6-v2) |
| R7 | Scope creep delays core features | High | High | Strict phase adherence, P1 features first |
| R8 | NL-to-Chart generates unsafe code | Low | Critical | AST validation, function whitelist, no exec() |
| R9 | PyMC too heavy for cloud deployment | High | Medium | Implement lightweight conjugate prior alternative |
| R10 | Time-series parser still fails on edge cases | Medium | Medium | Extensive test suite, user manual format override |

---

## 17. Success Metrics & KPIs

### 17.1 Feature Completeness

| Metric | Target |
|--------|--------|
| P0 features implemented | 100% |
| P1 features implemented | 100% |
| P2 features implemented | ≥ 80% |
| P3 features implemented | ≥ 50% |
| Total new features shipped | ≥ 40 |

### 17.2 Quality Metrics

| Metric | Target |
|--------|--------|
| All 7 tabs functional on Streamlit Cloud | ✅ |
| All 7 tabs functional on Hugging Face Spaces | ✅ |
| No critical bugs on demo dataset | ✅ |
| Average page load time | < 3s |
| ML model training (100K rows) | < 60s |

### 17.3 Academic/Research Metrics

| Metric | Target |
|--------|--------|
| Novel features for FYP presentation | ≥ 10 |
| Features with "Innovation Score" ≥ 9/10 | ≥ 8 |
| Statistical tests supported | ≥ 15 |
| ML algorithms supported | ≥ 10 |

---

## 18. Appendix: File Structure (v3.0 Target)

```
DataPilot-AI/
├── app/
│   ├── main.py                        # Thin orchestrator (~300 lines)
│   ├── config.py                      # Enhanced constants
│   ├── components/                    # UI components
│   │   ├── ai_chat.py                 # [MODIFY] + RAG status, model selector, code mode
│   │   ├── bayesian_panel.py          # [NEW] Bayesian inference UI
│   │   ├── causal_panel.py            # [NEW] Causal inference UI
│   │   ├── clean_panel.py             # [MODIFY] + ML imputation, audit trail, anomaly
│   │   ├── comparison_view.py         # [NEW] Dataset comparison UI
│   │   ├── contract_builder.py        # [NEW] Data contract validator UI
│   │   ├── data_preview.py            # [MODIFY] + schema editor integration
│   │   ├── export.py                  # [KEEP] Chart export buttons
│   │   ├── gdpr_heatmap.py            # [NEW] GDPR risk visualization
│   │   ├── ml_studio.py               # [MODIFY] + SHAP, arena, HPO, what-if, AutoML
│   │   ├── onboarding.py              # [NEW] User onboarding system
│   │   ├── regression_panel.py        # [NEW] Regression analysis UI
│   │   ├── report_panel.py            # [MODIFY] + dashboard, PPT, data dictionary
│   │   ├── schema_editor.py           # [NEW] Interactive schema editor
│   │   ├── sidebar.py                 # [MODIFY] + LLM provider selector
│   │   ├── statistics.py              # [KEEP] Basic statistics display
│   │   ├── stats_panel.py             # [MODIFY] + effect sizes, narrator, regression tab
│   │   ├── story_view.py              # [NEW] Chart Story Mode UI
│   │   ├── trust_score_display.py     # [MODIFY] + GDPR integration
│   │   └── whatif_panel.py            # [NEW] What-If analysis UI
│   └── modules/                       # Backend logic
│       ├── agent_analyzer.py          # [NEW] Agentic auto-analysis
│       ├── ai_engine.py               # [MODIFY] + LLM Router integration
│       ├── anomaly_detector.py        # [NEW] Isolation Forest, LOF
│       ├── audit_trail.py             # [NEW] Cleaning operation logging
│       ├── automl_engine.py           # [NEW] FLAML AutoML pipeline
│       ├── bayesian_engine.py         # [NEW] Bayesian A/B testing
│       ├── causal_engine.py           # [NEW] DoWhy causal inference
│       ├── chart_recommender.py       # [NEW] AI chart recommendation
│       ├── cleaner.py                 # [MODIFY] + ML imputation methods
│       ├── cleaning_simulator.py      # [NEW] ML impact simulation
│       ├── cluster_profiler.py        # [NEW] Cluster persona generation
│       ├── column_renamer.py          # [NEW] AI column name suggestions
│       ├── dashboard_exporter.py      # [NEW] Interactive HTML dashboard
│       ├── data_contract.py           # [NEW] Pandera validation
│       ├── data_dictionary.py         # [NEW] Word/PDF data dictionary
│       ├── dataset_comparator.py      # [NEW] Dataset diff engine
│       ├── eda_engine.py              # [MODIFY] + VIF, multi-method correlation
│       ├── file_parser.py             # [KEEP] Multi-format parsing
│       ├── fuzzy_dedup.py             # [NEW] Near-duplicate detection
│       ├── gdpr_scanner.py            # [NEW] PII/GDPR deep scan
│       ├── geo_engine.py              # [NEW] Geospatial detection & mapping
│       ├── hpo_engine.py              # [NEW] Hyperparameter optimization
│       ├── llm_router.py              # [NEW] Multi-provider LLM router
│       ├── ml_advisor.py              # [MODIFY] + 10+ algorithms, model serialization
│       ├── mlflow_logger.py           # [NEW] MLflow integration
│       ├── model_arena.py             # [NEW] Multi-model comparison
│       ├── notebook_exporter.py       # [MODIFY] + rich narrative mode
│       ├── power_analysis.py          # [NEW] Sample size planning
│       ├── pptx_exporter.py           # [NEW] PowerPoint generation
│       ├── rag_engine.py              # [NEW] ChromaDB RAG system
│       ├── regression_engine.py       # [NEW] Full regression suite
│       ├── relationship_detector.py   # [NEW] Column relationship graph
│       ├── report_generator.py        # [MODIFY] + enhanced templates
│       ├── report_scheduler.py        # [NEW] Scheduled email delivery
│       ├── session_manager.py         # [NEW] Persistent session state
│       ├── shap_explainer.py          # [NEW] SHAP analysis engine
│       ├── stats_engine.py            # [MODIFY] + 15+ tests, corrections
│       ├── stats_narrator.py          # [NEW] AI statistical narration
│       ├── story_generator.py         # [NEW] Data story narrative
│       ├── text2viz.py                # [NEW] NL-to-chart engine
│       ├── time_series.py             # [MODIFY] + robust parser
│       ├── trust_score.py             # [MODIFY] + GDPR scan integration
│       ├── ts_stats_engine.py         # [NEW] Time-series statistical tests
│       ├── type_fixer.py              # [NEW] Smart type correction
│       └── viz_engine.py              # [MODIFY] + geo, sankey, animated charts
├── assets/
│   └── styles.css                     # [MODIFY] Enhanced dark theme
├── tests/                             # [NEW] Test suite
│   ├── test_trust_score.py
│   ├── test_stats_engine.py
│   ├── test_ml_advisor.py
│   ├── test_cleaner.py
│   └── test_file_parser.py
├── .streamlit/
│   ├── config.toml                    # [NEW] Streamlit theme config
│   └── secrets.toml                   # [NEW] API keys (gitignored)
├── requirements.txt                   # [MODIFY] Updated dependencies
├── requirements-optional.txt          # [NEW] Heavy optional deps
├── runtime.txt                        # [KEEP] Python 3.11
├── PRD.md                             # [NEW] This document
├── README.md                          # [MODIFY] Updated documentation
└── .gitignore                         # [MODIFY] Add secrets, venv, cache
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | June 2026 | Mansoor Sarookh | Initial PRD for DataPilot-AI v3.0 |

---

> **Note:** This PRD is a living document. Features may be reprioritized based on user feedback, deployment constraints, and FYP timeline requirements. The implementation roadmap assumes a 16-week development cycle with a single developer.

---

*DataPilot-AI v3.0 — Built with 🧠 by Mansoor Sarookh | BSCS FYP 2025-26*
