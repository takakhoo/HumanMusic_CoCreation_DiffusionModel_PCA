# Human–Machine Co-Creation with Music Diffusion Models via PCA-based Sliders

This repository contains the implementation work for an MS thesis on slider-based co-creation between musicians and text-to-audio diffusion models. We build on SliderSpace (Gandikota et al., 2025) and adapt its unsupervised discovery of semantic directions to music. The goal is a system where a musician can enter a prompt (e.g., “solo jazz guitar, warm tone, swing feel”) and then steer continuous musical axes such as brightness, density, or ambience through discovered “sliders” backed by LoRA adapters.

## SliderSpace-to-Audio Overview

For a fixed textual concept \(c\), the diffusion model induces a manifold of possible waveforms \( \mathcal{M}_\theta(c) \). We search for controllable directions \( \{T_i\}_{i=1}^n \) satisfying three SliderSpace principles: unsupervised discovery, semantic orthogonality, and distribution consistency. The pipeline mirrors the original paper but swaps image tools for audio components:

1. **Distribution sampling:** Generate \(m\) clips \(x_j \sim \mathcal{M}_\theta(c)\) by varying diffusion seeds. Generation will come from Stable Audio Open 1.0 or a comparable open text-to-audio backbone.
2. **Semantic encoding:** Use an audio–text encoder such as CLAP to embed each clip: \( \phi(x_j) \in \mathbb{R}^d \).
3. **PCA decomposition:** Stack embeddings and compute principal components \(V = \mathrm{PCA}(\{\phi(x_j)\}) = \{v_i\}\). Each \(v_i\) represents one dominant variation axis in the concept’s sonic manifold.
4. **Slider training:** Attach rank-\(r\) LoRA adapters to the diffusion model’s cross-attention layers. For slider \(i\), optimize adapter weights so that the CLAP shift \( \Delta \phi_i = \phi(\tilde{x}_i) - \phi(x) \) aligns with \(v_i\) via the SliderSpace loss
\[
    \mathcal{L}_{\text{slider}} = 1 - \cos(\Delta \phi_i, v_i).
\]
5. **Evaluation + UI:** Sweep slider strengths, analyze MIR descriptors (spectral centroid, onset density, dynamics), and verify that prompt alignment remains high. Promote robust sliders into an interactive co-creation demo (e.g., Gradio) where users can audition combinations in real time.

This formulation keeps the creative exploration workflow faithful to SliderSpace while grounding each discovered direction in musically meaningful audio statistics and listening tests.

## Repository Layout

- `configs/`: shared configuration files (model checkpoints, data roots, default prompts).
- `scripts/`:
  - `generate_samples.py`: placeholder sampler that will call Stable Audio Open once hooked up.
  - `compute_embeddings.py`: computes (currently stubbed) CLAP embeddings for every concept.
  - `run_pca.py`: performs PCA on stored embeddings and records variance summaries.
  - `train_sliders.py`: records slider-training plans; will hold the LoRA optimization loop.
- `data/raw`, `data/embeddings`, `data/pca`: generated artifacts per concept.
- `latex/ms_thesis_notes.tex`: living thesis notes describing methodology, experiments, and open questions.
- `notebooks/`: interactive analysis and visualization (to be filled as experiments ramp up).

## Project Log

### 2025-02-08 23:59 EST
- Initialized the repository structure under `/scratch/f004h1v/MS_Thesis` with folders for configs, scripts, raw/derived data, LaTeX notes, and notebooks.
- Added `configs/model_config.yaml` plus initial script stubs for sample generation, CLAP embeddings, PCA, and slider training plans.
- Drafted `latex/ms_thesis_notes.tex` describing motivation, method overview, evaluation plan, and open questions for translating SliderSpace to audio.
- Scripts currently emit placeholder audio/embeddings so that we can test the pipeline end-to-end before integrating Stable Audio Open + CLAP.
- Next steps: (1) hook up Stable Audio Open inference in `generate_samples.py`; (2) load a real CLAP checkpoint for `compute_embeddings.py`; (3) implement LoRA optimization inside `scripts/train_sliders.py`; (4) start collecting pilot prompts and MIR metrics for slider evaluation.
