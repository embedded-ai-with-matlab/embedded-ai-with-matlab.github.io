# Hands-On Embedded AI with MATLAB

[![Publish Quarto book](https://github.com/embedded-ai-with-matlab/embedded-ai-with-matlab.github.io/actions/workflows/publish.yml/badge.svg)](https://github.com/embedded-ai-with-matlab/embedded-ai-with-matlab.github.io/actions/workflows/publish.yml)

**From Sensor Data to Optimized Models and On-Device Deployment**

This repository is an open Quarto book and MATLAB-first course with **four parts, twelve chapters, and twelve planned labs**. Chapter 1 and Lab 1 are written; the remaining chapters and labs have scoped teaching outlines, not completed content.

**Website (after enabling Pages):** https://embedded-ai-with-matlab.github.io/

## Content

- `chapters/`: textbook chapters; Chapter 1 is the complete initial chapter.
- `labs/`: 12-week outline, detailed first lab, and reusable lab template.
- `matlab/projects/vibration-severity-monitor/`: the supplied MATLAB project, including the 45 original CSV recordings, six executable MATLAB stages, helper functions, figures, and Arduino sketch.
- `matlab/fundamentals/`: short independent MATLAB examples for Chapter 1.
- `hardware/`, `datasets/`, `models/`, `figures/`: scaffolds for future modules.
- `.github/workflows/publish.yml`: Quarto HTML build and GitHub Pages publish workflow.

## Build locally

Install the [Quarto CLI](https://quarto.org/docs/get-started/), then:

```bash
quarto render --to html
quarto preview
```

To build a PDF, also install a LuaLaTeX-compatible TeX distribution, then run `quarto render --to pdf`. PDF builds are *not* required for the website workflow.

## MATLAB lab

From MATLAB, open and run `matlab/labs/00-run-lab1.m` to execute preprocessing and the first baseline, or open `matlab/projects/vibration-severity-monitor/Part1_PrepareData.m` to run preprocessing separately. Read `labs/01-vibration-severity-monitor.qmd` for the guided lab instructions. For the full six-stage training/deployment workflow, follow the original `matlab/projects/vibration-severity-monitor/README.md` and ensure the required products and board version are available.

The included CSV dataset is 45 recordings, 15 each for NORMAL, TAPPING, and GENTLE_SHAKING. This is a teaching example, not a validated industrial vibration severity standard. The named levels describe recording conditions and should not be confused with ISO severity grades.

## GitHub Pages setup

1. Create a **public** repository named exactly `embedded-ai-with-matlab.github.io` in the **embedded-ai-with-matlab** organization. Do **not** initialize it with README/license; this directory already includes the files.
2. Clone the empty repository and copy all files (including `.github`) from this package to its root, then commit and push to `main`. Alternatively run the Git commands below from this extracted folder.
3. In **Settings → Pages → Build and deployment**, choose **Source: GitHub Actions**.
4. In **Actions**, allow the repository workflows to execute if prompted. The `publish.yml` workflow builds the HTML site and deploys `_site/` to GitHub Pages.
5. Visit https://embedded-ai-with-matlab.github.io/ after the publish job succeeds.

```bash
git init
git branch -M main
git add .
git commit -m "Initialize Embedded AI with MATLAB Quarto book"
git remote add origin https://github.com/embedded-ai-with-matlab/embedded-ai-with-matlab.github.io.git
git push -u origin main
```

If using a GitHub repository that is already initialized, clone it first and copy the package contents into that clone instead of running `git init`.

## Contributing and licensing

The imported vibration example is supplied course material. **No license is assigned by this scaffold**; choose a suitable content and code license and verify permissions for imported datasets/images before opening the repository for external reuse. See `CONTRIBUTING.md`. The book is designed for teaching and reproducibility; performance figures in the imported project are illustrative until reproduced on the target MATLAB release and hardware.
