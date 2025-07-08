
# MTB-SHIELD

## Introduction

**MTB-SHIELD** is a high-performance software package developed to **predict antibiotic resistance** in *Mycobacterium tuberculosis* (MTB) using machine learning. It supports prediction across **13 antibiotic drug classes**, including:

Amikacin (AMI), Bedaquiline (BDQ), Clofazimine (CFZ), Delamanid (DLM), Ethambutol (EMB), Ethionamide (ETH), Isoniazid (INH), Kanamycin (KAN), Levofloxacin (LEV), Linezolid (LZD), Moxifloxacin (MXF), Rifampicin (RIF), Rifabutin (RFB)

## Use Cases

This README outlines two primary use cases:

- **Running the pre-trained classifiers** (recommended for most users):  
  The `MTB-SHIELD.py` script is designed for this purpose. It provides a lightweight interface that requires no GPU resources and can be executed using a single CPU core. The script supports parallel execution, making it suitable for large-scale antibiotic resistance studies.

- **Rebuilding the classifiers from raw sequencing data** (intended for advanced users):  
  This workflow enables researchers to develop custom AI-based models and software in a *de novo* fashion. It offers a robust foundation for exploring novel approaches to antimicrobial resistance prediction.


This project is actively maintained by **[M. Serajian](https://github.com/M-Serajian/)**  
📧 *Contact:* ma.serajian@gmail.com  
🐛 *Bug reports:* Please open an issue via the [GitHub Issues](https://github.com/M-Serajian/MTB-SHIELD/issues) page.

---

## Installation

MTB-SHIELD can be installed in two ways:

1. **Automatic Installation** — using conda (tested on Red Hat Enterprise Linux 9.5 (Plow))
2. **Manual Installation** — Recommended for custom setups or limited environments

### Recommended Environment for conda installation 

- **Python**: 3.10  (Tested)
- **Conda**: 25.5.1 (Tested)

---

## Installation via Conda

To install and activate the environment:
```bash
python setup.py install --env resistance-predictor-env
conda activate resistance-predictor-env
```

To deactivate and delete environment:
```bash
conda deactivate
python setup.py delete

```

To get help:
```bash
python setup.py --help
```


---

## Manual Installation

If `conda` is not available, install the following dependencies manually:

### Dependencies

- `python=3.10`
- `xgboost=3.0.2`
- `scikit-learn`
- `scipy=1.11.4`
- `numpy`
- `pandas`
- `pyarrow`
- `pynvml`


