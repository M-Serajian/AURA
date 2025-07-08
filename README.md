
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
## 🚀 Installation Guide

MTB-SHIELD can be installed in two ways:

1. **Automatic Installation** *(recommended for most users)* — using Conda (tested on **Red Hat Enterprise Linux 9.5 (Plow)**)
2. **Manual Installation** — for custom setups, advanced users, or restricted environments

---

## 📦 Option 1: Automatic Installation via Conda (Recommended)

### ✅ Tested Environment

- **Operating System**: Red Hat Enterprise Linux 9.5 (Plow)
- **Python**: 3.10.x
- **Conda**: 25.5.1

### 🔧 Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/M-Serajian/MTB-SHIELD.git
   cd MTB-SHIELD
   ```

2. Install the environment and required dependencies:

   ```bash
   python setup.py install --env resistance-predictor-env
   ```

3. Activate the environment:

   ```bash
   conda activate resistance-predictor-env
   ```

4. Run the classifier (see [Usage](#-usage))

---

### 🧹 Deactivation & Cleanup

To deactivate the environment:

```bash
conda deactivate
```

To delete the environment:

```bash
python setup.py delete
```

---

## 🔧 Option 2: Manual Installation

Manual installation is recommended for advanced users who need full control over dependencies or are working in restricted environments.

### 📦 Dependencies

The following packages and libraries are **required** to install and run pretrained **MTB-SHIELD** cllasifiers.  
Tested and recommended versions are indicated in parentheses.

#### 🔧 Core Build Dependencies:
- `cmake` (tested: **3.30.5**)
- `gcc` (tested: **12.2**)  
  ⚠️ *Note: GCC 14 is **not recommended** due to incompatibility with `gerbil`.*
- `boost` (tested: **1.77**)

#### 🐍 Python Environment:
- `python=` (tested: **3.10**)
- `xgboost` (tested: **3.0.2**)
- `scipy` (tested: **1.11.4**)
- `numpy`
- `pandas`
- `pyarrow`
- `pynvml`

#### 🧰 System Libraries (required during build and runtime):
- `libboost-all-dev`
- `libz3-dev`
- `libbz2-dev`


### 🛠️ Build Instructions

```bash
git clone https://github.com/M-Serajian/MTB-SHIELD.git
cd MTB-SHIELD
git submodule update --init --recursive

cd include/KMX/include/gerbil-DataFrame/build
cmake ..
make -j

cd ../../../../../
```

<table width="100%">
  <tr>
    <td><hr></td>
    <td><hr></td>
  </tr>
</table>


## 🧬 Usage

Once installed and built, run MTB-SHIELD classifiers using:

```bash
python MTB-SHIELD.py -i /path/to/input.fasta -o /path/to/output.csv -t /path/to/temp_directory
```

### 🔍 Arguments

- `-i`: Path to the input FASTA file (required)
- `-o`: Path to the output CSV file (required)
- `-t`: Temporary working directory (optional; default: `temp/`)

### 💡 Example

```bash
python MTB-SHIELD.py -i data/example_data/sample.fasta -o results/prediction.csv 
```

The resistance profiling output will be saved in the specified output file.

---
