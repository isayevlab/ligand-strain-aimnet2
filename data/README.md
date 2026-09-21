# Data

All energies from AIMNet2-CPCM (B97-3c/CPCM(water) reference) unless stated. Units: hartree in the SDF tags and energy columns, kcal/mol for strain.

| file | rows | description |
|---|---|---|
| `lcse_master_table.csv` | 7887 | one row per ligand instance; column definitions in the top-level README |
| `LCSE_details.json.gz` | 7887 | the same content as a nested JSON (original analysis input) |
| `bound_conformers.sdf.gz` | 7887 | deposited pose after relaxation of bond lengths and angles with all rotatable-bond dihedrals fixed; tags `energy_hartree` (ensemble mean), `final_fmax_eV/A`, `model_name` |
| `global_conformers.sdf.gz` | 7887 | lowest-energy AIMNet2-CPCM conformer from up to 5000 Omega conformers, same tags |
| `mmff94_strain_all.csv` | 7887 | MMFF94 energies and strain: single point on the AIMNet2-CPCM conformers (`*_sp`) and both endpoints re-optimized on MMFF94 (`*_reopt`; bound with dihedral restraints ±0.5°, k = 1000 kcal/mol/rad², global free); `ntors` = number of restrained dihedrals; one ligand (`ok = False`) failed MMFF typing |
| `mmff94_strain_summary_by_charge.csv` | 6 | Table S4 |
| `gaff2_obc2_subset.csv` | 50 | GAFF2 (gaff-2.11, AM1-BCC) strain in vacuum and in OBC2 for the stratified subset (10 ligands per net charge 0, +1, −1, −2, −3), single point and re-optimized; MMFF94 columns repeated for convenience |
| `gaff2_obc2_subset_summary.csv` | 5 | Table S5 |
| `conformer_count_stats.csv` | 5 | Omega conformers per ligand by rotor bin (Figure S14) |
| `timing/` | | per-conformer timings (Table S3): `cpu_timing_container.csv` (2-core container, near-minimum starts); `sunspear/` (EPYC 7H12 + L40S, ETKDG starts): `sf_timing_results.json` batched SaddleForge L-BFGS, `steps_timing_results.json` unbatched ASE FIRE with the 4-member ensemble on GPU and CPU, `cpu_member_results.json` single member on CPU, `cpu_refs_results.json` MMFF94 and GFN2-xTB, `gpu_timing_steps.log` forward-pass batch-size scan (Table S6) |

## Provenance

- Ligand poses, PDB identifiers, resolution and EDIAm: LigBoundConf (Tong and Zhao, *J. Chem. Inf. Model.* 2021, 61, 1180). Elements outside the AIMNet2 set (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I) were excluded; protonation and tautomer states are those of the deposited structures and were held fixed between the two endpoints.
- Annotations (`ligand_function`, `enzyme_class`, `enzyme_subclass`, `EC_no`): PDBe REST API, retrieved 2024.
- Conformer generation: OpenEye Omega, `-maxconfs 5000 -ewindow 999 -rms 0.2 -rmsrange 0.2 0.4 0.6 0.8 -rangeIncrement 3 -sampleHydrogens true`, default search force field (mmff94smod_NoEstat, no solvation, no electrostatics).
- Optimization: in-house batched FIRE on NVIDIA V100 GPUs with one member of the AIMNet2-CPCM ensemble; energies in the tags are the 4-member ensemble mean on the final geometry.
- Ligand IDs: `<ligand code>_<PDB id>_<chain>_<residue number>`, e.g. `6SH_5LAR_A_401`.

## Known caveats

- Values at net charge −3 and −4 (nucleotide di- and triphosphates and other polyanions) are upper bounds; continuum solvation over-stabilizes compact conformers of these species.
- `mmff94_strain_reopt_kcal` is a local force-field strain: the global endpoint is relaxed from the AIMNet2-CPCM minimum, not from an independent MMFF94 conformer search. About 5% of values are negative for that reason.
- Atom order differs between `bound_conformers` and `global_conformers` for the same ligand.
