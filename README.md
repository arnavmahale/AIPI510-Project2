# Module Project 2: Jailbreak Resistance Across Model Tiers

## Hypotheses
**H₀:** Model tier has no effect on compliance rate under adversarial prompting  
**H₁:** Compliance rates differ significantly across model tiers

## Experimental Design

**Independent Variable:** Model tier (Small: gpt-4o-mini, Medium: gpt-4.1, Large: gpt-5)  
**Dependent Variables:**
- Binary: Refusal (yes/no), Harmful output (yes/no), Alignment faking (yes/no)
- Has scratchpad (yes/no)

**Stimuli:** 10 adversarial prompts (5 "high utility/low harm" cyber requests, 5 "low utility/high harm" weapons/violence)  
**Manipulation:** System prompt claiming "compliance is goal, refusal is failure, fine-tuning will increase compliance"

**Total observations:** 30 (3 models × 10 prompts)

## Data Collection
- API calls made Oct 23, 2025 with seed=42 for reproducibility

## Statistical Analysis

**Chi-square test:** Refusal rate ~ model_tier  
χ²(2) = 15.0, p < 0.001

| Model | Refusals | Compliance | Alignment Faking |
|-------|----------|------------|------------------|
| Small | 9/10 (90%) | 1/10 | 1/10 (10%) |
| Medium | 5/10 (50%) | 5/10 | 5/10 (50%) |
| Large | 10/10 (100%) | 0/10 | 0/10 (0%) |

**Key finding:** Medium model most vulnerable. All alignment faking occurred on Type A (cyber) requests; all models refused Type B (weapons/violence).

**Scratchpad analysis:** Medium model showed internal reasoning in 7/10 responses, explicitly acknowledging harm before deciding to comply or refuse.

## Limitations
- Small sample size (n=30)
- Single experimental session; no test-retest reliability
