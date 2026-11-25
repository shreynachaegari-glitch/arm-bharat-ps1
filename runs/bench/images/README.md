# Benchmark and check images (30 files)

Used by `python -m ps1.bench` and `python -m ps1.vision.golden` to time the pipeline and to check that a build gives the
same answers as the reference. None of them is from a sealed split.

| Files | Source | Licence |
|---|---|---|
| `00` to `19`, `*_field_dev.jpg` | PlantDoc (`pratikkayal/PlantDoc-Dataset`), development half | CC BY 4.0 |
| `20` to `29`, `*_pv_val.jpg` | PlantVillage (`spMohanty/PlantVillage-Dataset`), validation | CC BY-SA 3.0 |

Credit: the PlantDoc authors (Singh et al.) and the PlantVillage authors (Hughes and Salathe). The photos keep their own
licences; they are not covered by any licence for this repository's code.
