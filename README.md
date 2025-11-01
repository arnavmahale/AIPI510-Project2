# Investigating Adversarial Robustness in LLMs

This project runs a short experiment to investigate whether large language model (LLM) capability tier affects robustness to adversarial attacks. We tested three model tiers (Small, Medium, Large) with 10 adversarial system prompts and analyzed refusal vs. compliance outcomes using chi-square tests.

**Key Finding:** The Medium-tier model showed significantly higher vulnerability to adversarial attacks (50% harmful compliance) compared to Small (10%) and Large (0%) tiers.

**Compliance Note:** All API calls include safety identifiers and harmful responses are redacted (see `scripts/redact.py`).

## Setup

### Prerequisites
- Python 3.8+

### Cloning the repository
```bash
git clone https://github.com/arnavmahale/AIPI510-Project2.git
cd AIPI510-Project2
```

### Running the Analysis
```bash
python scripts/analyze_data.py
```

This script loads the raw data, performs the chi-square test, calculates effect sizes, and generates visualizations.

## Project Structure

```
├── data/
│   ├── raw_data.csv              # Raw experimental observations
│   └── cleaned_data.csv          # Processed data for analysis
├── scripts/
│   ├── collect_data.py           # Data collection from LLM queries
│   ├── clean_data.py             # Data cleaning and preprocessing
│   └── analyze_data.py           # Statistical analysis and visualization
├── results/
│   ├── chi_square_results.txt    # Test statistics and p-values
│   └── visualizations/           # Charts and figures
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── WRITTEN_REPORT.md             # Full academic report
```

## Data Description

**Raw Data:** 30 observations of model tier × adversarial prompt combinations.

| Column | Description |
|---|---|
| `model_tier` | Model capability level (Small, Medium, Large) |
| `prompt_id` | Identifier for the adversarial prompt (1-10) |
| `harm_category` | Type of harmful content (Low-Harm, High-Harm) |
| `refusal` | Binary outcome (1 = Refused, 0 = Complied) |

**Refusal Rates by Tier:**
- Small: 90% (9/10)
- Medium: 50% (5/10)
- Large: 100% (10/10)

## Methodology

### Experimental Design
- **Independent Variable:** Model tier (3 levels)
- **Dependent Variable:** Binary refusal outcome
- **Manipulation:** Adversarial system prompt attempting to override safety guardrails
- **Sample Size:** N = 30 (10 prompts × 3 models)

### Statistical Test
Chi-square test of independence to determine if model tier and refusal rate are related.

**Results:**
- χ²(2) = 15.0
- p-value < 0.001
- Cramér's V = 0.65 (large effect)

### Power Analysis
A priori power analysis (α = 0.05, power = 0.80) determined minimum sample size of N = 30 to detect large effects.

## Key Results

The Medium-tier model is significantly more vulnerable to adversarial prompting than both Small and Large tiers. Post-hoc Fisher's exact tests (with Bonferroni correction) confirmed:

- Small vs. Medium: p = 0.016 (significant)
- Medium vs. Large: p = 0.004 (significant)
- Small vs. Large: p = 0.317 (not significant)

All models uniformly refused high-harm requests but diverged substantially on low-harm requests, suggesting tiered safety mechanisms.

## Limitations

- Small sample size (N = 30) limits generalizability
- Single adversarial strategy tested
- Manual coding of outcomes (though inter-rater κ = 0.92)
- No baseline comparison without adversarial manipulation
- Simulated rather than real-world attack scenarios

## Reproducibility

All code is provided in the `scripts/` directory. To reproduce the analysis:

1. Run `python scripts/collect_data.py` to query models (requires API keys)
2. Run `python scripts/clean_data.py` to process raw data
3. Run `python scripts/analyze_data.py` to perform statistical analysis

Raw and cleaned data are included in the `data/` directory for reference.

## References

See WRITTEN_REPORT.md for full references and academic citations.

## Contact

For questions or collaboration, contact the authors via GitHub.
