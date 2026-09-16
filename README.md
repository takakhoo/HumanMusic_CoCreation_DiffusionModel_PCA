# Controllable Music Co-Creation with Diffusion Models

Research scaffold for an MS thesis on discovering and training continuous,
musically meaningful controls for text-to-audio diffusion models. The approach
adapts SliderSpace-style semantic directions to audio by combining generated
samples, CLAP embeddings, principal component analysis (PCA), and LoRA adapters.

> **Status:** research prototype. The repository currently provides the
> experiment structure and dry-run pipeline; backbone inference, real CLAP
> embeddings, and LoRA optimization are planned integrations.

## Research question

Can a musician start from a text prompt and steer independent properties such
as brightness, density, ambience, or rhythmic activity without retraining the
full diffusion model?

## Proposed pipeline

1. Sample multiple clips for a fixed prompt while varying diffusion seeds.
2. Embed each clip with an audio–text representation model such as CLAP.
3. Apply PCA to discover dominant directions in the prompt-conditioned audio
   manifold.
4. Train lightweight LoRA adapters whose embedding shifts align with selected
   directions.
5. Evaluate direction consistency with audio descriptors, prompt alignment,
   and listening tests.

## Run the scaffold

```bash
git clone https://github.com/takakhoo/HumanMusic_CoCreation_DiffusionModel_PCA.git
cd HumanMusic_CoCreation_DiffusionModel_PCA
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python scripts/generate_samples.py
python scripts/compute_embeddings.py --concept solo_jazz_guitar_warm_tone_swing_feel
python scripts/run_pca.py --concept solo_jazz_guitar_warm_tone_swing_feel
python scripts/train_sliders.py --concept solo_jazz_guitar_warm_tone_swing_feel
```

These commands execute the repository's deterministic scaffold from generated
placeholder clips through embeddings, PCA, and a saved slider-training plan.
They validate data flow and configuration; they do not run a diffusion model,
compute real CLAP embeddings, or train a usable audio slider.

## Repository map

- `configs/model_config.yaml` — model, prompt, and output configuration
- `scripts/generate_samples.py` — sample-generation interface and placeholder
- `scripts/compute_embeddings.py` — embedding stage interface and placeholder
- `scripts/run_pca.py` — PCA stage and explained-variance artifacts
- `scripts/train_sliders.py` — LoRA training plan/interface
- `latex/ms_thesis_notes.tex` — method notes and open research questions
- `Papers/SliderSpacePaper.pdf` — motivating reference paper

## Evaluation plan

Candidate controls should be judged on monotonicity across slider strengths,
semantic independence, prompt preservation, perceptual quality, and agreement
between objective MIR descriptors and human listening tests.
