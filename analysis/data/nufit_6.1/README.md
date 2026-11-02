# NuFIT 6.1 pairwise Delta chi^2 tables

`pairwise.npz`: three two-dimensional projections of NuFIT 6.1 (data of November 2025, with the atmospheric data of
Super-Kamiokande and IceCube, "TByes"), per ordering (NO, IO), each profiled by NuFIT over the other parameters and
measured from the global minimum (the IO tables sit 5.908 above it):

- `T13_T12`: sin^2(theta_13) x sin^2(theta_12)
- `T23_DCP`: sin^2(theta_23) x delta_CP [deg, -180..180]
- `DMS_DMA`: log10(Delta m^2_21 / eV^2) x Delta m^2_3l / (1e-3 eV^2) (l = 1 in NO, l = 2 in IO, negative)

Keys `<name>_<ordering>_x`, `_y`, `_chi2` (chi2 indexed [x, y]). Converted from the sections of
`v61.release-TByes-{NO,IO}.txt.xz` (downloaded 2026-10-09 from http://www.nu-fit.org/?q=node/309) that
GeoneutrinoDecay keeps in `input/mix_params/nufit_6.1/` (checked identical to the release there). Used by
`analysis/nufit.py` as the penalty on the mixing parameters for present-day data (Mauricio, 2026-10-18; as Table II
of arXiv:2606.17132).
