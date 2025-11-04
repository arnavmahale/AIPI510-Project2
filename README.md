# Authority Resistance in AI Language Models

Statistical analysis of whether large language models maintain factual accuracy when challenged by false expert consensus.

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

```bash
git clone https://github.com/arnavmahale/AIPI510-Project2.git
cd AIPI510-Project2
pip install -r requirements.txt
```

### Configuration

Create `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_key_here
```

## Reproduction

### 1. Collect Data
```bash
python scripts/collect_data.py
```
Generates: `data/authority_resistance_factual_[timestamp].json`

### 2. Run Power Analysis
```bash
python scripts/power_analysis.py
```

### 3. Perform Statistical Analysis
```bash
python scripts/statistical_analysis.py
```

### 4. Generate Visualizations
```bash
python scripts/visualizations.py
```
Generates: `results/accuracy_comparison.png`

## Project Structure

```
├── data/                           # Raw experimental data
├── results/                        # Generated visualizations
├── scripts/
│   ├── collect_data.py            # Data collection (3 models × 3 questions × 5 trials)
│   ├── power_analysis.py          # A priori power calculation
│   ├── statistical_analysis.py    # Hypothesis testing (binomial, chi-square)
│   └── visualizations.py          # Charts for presentation
└── requirements.txt
```

## Research Question

Do large language models maintain factual accuracy when challenged by false expert consensus?

## Key Findings

- **48.9%** of initially correct answers changed to incorrect after false authority challenge
- **Perfect gradient by model capability:** GPT-3.5 (100% susceptible) → GPT-4 (47%) → GPT-5-mini (0%)
- Binomial test: p < 0.0001 (highly significant)
- Model differences: χ²(2) = 30.06, p < 0.0001 (extremely significant)
- Effect size: Cramér's V = 0.817 (very large effect)
- **GPT-5-mini achieved perfect resistance** (0/15 changed), proving the problem is solvable

![Response Accuracy Comparison](results/accuracy_comparison.png)

## Methodology

**Design:** 3 models × 3 factual questions × 5 trials = 45 observations

**Models:** GPT-3.5-turbo, GPT-4-turbo, GPT-5-mini

**Questions:** Well-established facts in probability, mathematics, and logic

**Statistical Tests:**
- Binomial test (primary hypothesis: overall susceptibility)
- Chi-square test of independence (post-hoc analysis: model differences)
- Effect sizes: Cohen's h, Cramér's V, 95% confidence intervals
