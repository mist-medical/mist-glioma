# MIST Glioma

MIST Glioma is a command-line tool for segmenting adult pre-operative and
post-operative gliomas from multiparametric brain MRI. The underlying model
placed third in the 2025 BraTS Adult Glioma Challenge.

The pipeline accepts T1-weighted, T2-weighted, contrast-enhanced T1-weighted,
and FLAIR volumes. It performs registration and HD-BET brain extraction before
running a five-fold ensemble of pocket nnU-Net models with MIST.

## Segmentation labels

| Value | Region |
| ---: | --- |
| 0 | Background |
| 1 | Non-enhancing tumor core |
| 2 | Surrounding non-enhancing FLAIR hyperintensity |
| 3 | Enhancing tumor |
| 4 | Resection cavity |

## Installation

MIST Glioma requires Python 3.10 or newer and MIST `2.3.0` or newer.

```bash
python -m pip install mist-glioma
```

## Model weights

Tumor model weights are distributed separately from the source package. MIST
Glioma downloads the following checkpoints from
[`mist-medical/mist-glioma-brats2025`](https://huggingface.co/mist-medical/mist-glioma-brats2025)
on first use:

```text
models/
├── brats2025/
│   ├── fold_0.pt
│   ├── fold_1.pt
│   ├── fold_2.pt
│   ├── fold_3.pt
│   └── fold_4.pt
```

Downloaded tumor checkpoints are stored in the standard Hugging Face cache and
reused on subsequent runs. Set `HF_HOME` to choose a different cache location.
For an offline run using previously cached weights, pass `--offline`. HD-BET
manages its own publicly available model weights.

Developers can set `MIST_GLIOMA_MODELS_DIR` to a local directory with the
layout above to bypass Hugging Face entirely.

## Usage

All four input volumes must correspond to the same subject.

```bash
mist_glioma \
    --t1 /path/to/t1.nii.gz \
    --t2 /path/to/t2.nii.gz \
    --tc /path/to/t1ce.nii.gz \
    --fl /path/to/flair.nii.gz \
    --output /path/to/output \
    --device cuda
```

The output directory contains:

- `tumor_rc_mask.nii.gz`: multiclass tumor and resection-cavity segmentation
- `brain_mask.nii.gz`: brain mask in the original contrast-enhanced T1 space

CPU inference is available with `--device cpu`, although GPU inference is
recommended.

## Atlas

The SRI24 atlas required by the registration pipeline is included at
`data/atlases/sri24/spgr_unstrip.nii.gz`. It is distributed under the CC
BY-SA 3.0 license; its complete license is included alongside the atlas.

When using the atlas, please cite:

```bibtex
@article{rohlfing2010sri24,
  title={The SRI24 multichannel atlas of normal adult human brain structure},
  author={Rohlfing, Torsten and Zahr, Natalie M and Sullivan, Edith V and Pfefferbaum, Adolf},
  journal={Human brain mapping},
  volume={31},
  number={5},
  pages={798--819},
  year={2010},
  publisher={Wiley Online Library}
}
```

## Validation

Install the development dependencies and run the test suite:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

The unit tests enforce at least 98% branch coverage.

## Citation

If you use MIST Glioma in your work, please cite:

```bibtex
@inproceedings{celaya2025pre,
  title={Pre-and Post-Treatment Glioma Segmentation with the Medical Imaging Segmentation Toolkit},
  author={Celaya, Adrian and Netherton, Tucker and Schellingerhout, Dawid and Chung, Caroline and Riviere, Beatrice and Fuentes, David},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={28--37},
  year={2025},
  organization={Springer}
}
```

The final challenge rankings are available on
[Synapse](https://www.synapse.org/Synapse:syn64153130/wiki/633062).

## License

The source code is licensed under the Apache License 2.0. The bundled SRI24
atlas retains its separate CC BY-SA 3.0 license.
