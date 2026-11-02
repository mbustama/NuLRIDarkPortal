# NuLRIDarkPortal

**New forces on high-energy cosmic neutrinos, from the Hubble scale to the Earth.**

This code computes the flux of high-energy astrophysical neutrinos at IceCube when they feel a new force carried by a
light mediator, and fits it to IceCube data to limit the mediator's mass and coupling.

Models: the dark portal, gauged $L_e - L_\mu$ and $L_e - L_\tau$, and secret neutrino interactions.

**Status:** MeV limits done; the final ultralight run is under way; secret interactions next. Overview:
`slides/Bustamante-NuLRIDarkPortal_progress.pdf`.

## Install

```bash
git clone https://github.com/mbustama/NuLRIDarkPortal.git && cd NuLRIDarkPortal
conda create -n nulri python=3.12 pip && conda activate nulri
pip install numpy==1.26.4 scipy==1.15.3 numba==0.60.0 llvmlite==0.43.0 matplotlib==3.10.8 \
            tqdm==4.67.3 pytest==9.0.3 mpmath==1.3.0 tbb==2023.1.0 jupyter
```

## Start here

`notebooks/fluxes_at_earth.ipynb` computes the flux at Earth for three example mediators, turning on one physical
effect at a time.

To fit IceCube data:

```bash
python -m analysis analysis/configs/campaign_ml/now_ic26/le_lmu_mev.json --fit
```

Full exclusion planes run on a cluster: see `slurm/README.md`.

## More

- `notes/notes.pdf`: the physics.
- `docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md`: what the calculation leaves out.
- `docs/TECHNICAL_REFERENCE.md`: implementation, settings and validation.
- Tests: `python -m pytest tests/ -m "not slow"`.

Mauricio Bustamante, Niels Bohr Institute. GPL-3.0 or later.
