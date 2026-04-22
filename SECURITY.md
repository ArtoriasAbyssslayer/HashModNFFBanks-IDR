# Security Policy

## Supported Versions

This is an academic thesis research repository. Only the latest release is
actively maintained. Older experimental checkpoints are not supported with
security patches.

| Version       | Supported          |
| ------------- | ------------------ |
| latest (main) | :white_check_mark: |
| older commits | :x:                |

> **Note:** This project is primarily a research codebase and is not intended
> for production deployment. Security updates will be applied on a best-effort
> basis.

---

## Reporting a Vulnerability

If you discover a security vulnerability in this repository — including issues
related to dependency supply chain (e.g. PyTorch, tinycudann, pyhocon),
malicious model checkpoint loading, or unsafe data handling — please report it
responsibly.

### How to Report

- **Email:** harry.filis@yahoo.gr  
  Please use the subject line: `[SECURITY] HashModNFFBanks-IDR - <short description>`
- **Do NOT** open a public GitHub Issue for security vulnerabilities, as this
  may expose other users before a fix is available.

### What to Include

Please provide as much of the following as possible:
- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- The affected component (e.g. model loading, data pipeline, a specific dependency)
- Any suggested mitigations or patches

### Response Timeline

| Stage                        | Timeframe               |
| ---------------------------- | ----------------------- |
| Acknowledgement of report    | Within **3–5 days**     |
| Initial assessment/triage    | Within **1–2 weeks**    |
| Status update to reporter    | Within **2–3 weeks**    |
| Patch or mitigation released | On a best-effort basis  |

### What to Expect

- If the vulnerability is **accepted**, a fix or mitigation will be developed
  and you will be credited in the release notes (unless you prefer to remain
  anonymous).
- If the vulnerability is **declined** (e.g. out of scope or not reproducible),
  you will receive a clear explanation of the reasoning.

---

## Scope

The following are considered **in scope** for security reports:

- Unsafe deserialization of model checkpoints (e.g. via `torch.load`)
- Dependency vulnerabilities in `requirements.txt` or `environment.yml`
- Path traversal or unsafe file handling in training/evaluation scripts
- Arbitrary code execution risks in configuration parsing (e.g. `pyhocon`)

The following are considered **out of scope**:

- Theoretical vulnerabilities without a practical exploit
- Issues in third-party libraries that are already publicly tracked upstream
  (please report those directly to the respective maintainers)
- Performance or reproducibility issues unrelated to security

---

## Dependencies & Known Risks

This project relies on the following major dependencies. Users are encouraged
to keep them up to date:

- [PyTorch](https://pytorch.org/) — model training & inference
- [tinycudann (tcnn)](https://github.com/NVlabs/tiny-cuda-nn) — CUDA hash grid encoding
- [pyhocon](https://github.com/chimpler/pyhocon) — configuration parsing
- [OpenCV](https://opencv.org/), [scikit-image](https://scikit-image.org/) — image processing

> ⚠️ **Checkpoint Warning:** Loading pretrained model checkpoints from
> untrusted sources via `torch.load()` can execute arbitrary code. Only load
> checkpoints from trusted sources.
