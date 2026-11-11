# IceCube MESE 2025 flavor likelihood

IceCube's released likelihood of the flavor composition at Earth, from the MESE flavor measurement (arXiv:2510.24957; Harvard Dataverse doi:10.7910/DVN/CBNMEB, version 2.0, released 2026-08-11, CC0 1.0). Copied on 2026-11-09 from ~/Research/NuFlavorRatiosSteriles/input/IceCube_MESE_2025_Dataverse/ (downloaded there on 2026-10-19 through the Dataverse access API). Decision A2 of the 2026-11-08 audit (docs/dev/AUDIT_2026-11-08_resolution.md).

| File | Dataverse id | MD5 | Spectrum model |
|---|---|---|---|
| `mese_contour.json` | 12322774 | d7e8f55e425c99df3ac8ad512902d25d | broken power law (IceCube's baseline) |
| `mese_contour_spl.json` | 12322772 | c385eeae6f5c8fde8d430a0002901c63 | single power law |

Format: `fe` (90 values, 0-0.98), `ftau` (90 values, 0-0.999), `TS[i_fe, j_ftau]` = -2 Delta ln L (NaN outside the flavor triangle). Checked 2026-11-08: the BPL grid's minimum along f_tau = 0 is TS 5.02 and along f_e = 0 is 8.65, IceCube's quoted 91.9% and 98.7% for 2 degrees of freedom.
