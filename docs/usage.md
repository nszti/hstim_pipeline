
# Usage Guide

This guide walks you through the setup and usage of the calcium imaging and stimulation data pipeline.

---

## 📦 Prerequisites

Before you begin, make sure the following are installed:

- **Git**
- **Anaconda**
- **MATLAB** (recommended versions: **2021b to 2023a**)  
  - Also install the **Parallel Computing Toolbox** for CellReg
- **Python IDE** (e.g., PyCharm)

---

## 📁 Repository Setup

### 1. Clone the Repository

Open Git Bash or your terminal and run:

```bash
git clone https://github.com/nszti/hstim_pipeline.git
cd hstim_pipeline
```

You should now see the `hstim_pipeline` folder in your working directory.

---

## 🧱 Project Structure

Here's a simplified layout of the project:

```
Hyperstim/pipeline_pending/
├── pipeline_script.py
├── mesc_loader.py
├── general.py
├── package_for_pipeline/
│   ├── CoM.py
│   ├── cellreg_analysis.py
│   ├── cellreg_process.py
│   ├── frequency_to_save.py
│   ├── functions.py
│   ├── mesc_data_handling.py
│   ├── mesc_tiff_extract.py
│   ├── suite2p_script.py
│   └── tiff_merge.py
└── hdf5io/
    └── setup.py
```

---

## 🐍 Python Environment Setup

The environment includes dependencies for Suite2p and the custom pipeline scripts.

### Step-by-Step:

1. Open your terminal / Anaconda Prompt
2. Navigate to the repository directory:
   ```bash
   cd path/to/hstim_pipeline
   ```
3. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   ```
4. Activate it:
   ```bash
   conda activate suite2p
   ```
5. Verify Suite2p is installed:
   ```bash
   suite2p --version
   python -m suite2p
   ```

---

## 🖥️ IDE Setup (PyCharm Recommended)

### Connect Conda Environment

1. Open PyCharm
2. Go to **Project > Python Interpreter**
3. Click **Add Interpreter**
4. Select **Add Local Interpreter**
5. Choose **Conda** and select the `suite2p` environment
6. Click OK and wait for it to load the packages

Now your IDE should be able to run the Python-based parts of the pipeline.

---

## 🧬 MATLAB + CellReg Setup

Since **CellReg** is MATLAB-based:

- Download and install **CellReg**
- Add all folders and subfolders of CellReg to the MATLAB path
- Use the MATLAB GUI to:
  - Load sessions
  - Run non-rigid alignment
  - Run probabilistic modeling (12 microns pixel size)

Full instructions are available in the official [CellReg documentation](https://github.com/zivlab/CellReg).

---

Once setup is complete, continue to the `pipeline_script.py` to run the analysis.
