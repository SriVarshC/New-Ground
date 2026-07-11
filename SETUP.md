\# Setup Log — New Ground



\## Verified working environment (2026-07-11)



\- OS: Windows 11

\- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.29 GB VRAM confirmed via PyTorch)

\- Driver CUDA support: check via `nvidia-smi`

\- Conda: 26.5.3, installed at D:\\tools\\miniconda3

\- Environment: `new-ground`, Python 3.10.20, located at D:\\tools\\conda-envs\\new-ground

\- PyTorch: 2.5.1+cu121 (CUDA 12.1 build)

\- torch.cuda.is\_available(): True



\## Known gotchas hit during setup

\- Anaconda ToS must be accepted before first `conda create`:

&#x20; `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`

&#x20; (also for `pkgs/r` and `pkgs/msys2` channels)

\- Always run `where python` after activating an env — multiple system Pythons

&#x20; are installed (3.12, 3.8, msys64) and the wrong one can shadow the conda env

&#x20; if activation didn't happen correctly.

\- Start menu "Anaconda Prompt" shortcut did not auto-activate conda on this

&#x20; machine — use `call D:\\tools\\miniconda3\\Scripts\\activate.bat` as a fallback.



\## conda config

\- pkgs\_dirs: D:\\tools\\conda-pkgs

\- envs\_dirs: D:\\tools\\conda-envs (first priority)



\## pip config

\- PIP\_CACHE\_DIR: D:\\tools\\pip-cache

