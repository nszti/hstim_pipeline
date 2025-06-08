
# API Reference

This document describes the main Python modules and functions used in the calcium imaging and electrical stimulation data pipeline.

---

## `pipeline_script.py`

The main script to run the data processing pipeline. Configure paths and parameters here.

- `RUN_MESC_PREPROCESS`: calls `mesc_tiff_extract.analyse_mesc_file()`
- `RUN_PREPROCESS`: runs `frequency_to_save.frequency_electrodeRoi_to_save`, `mesc_data_handling.tiff_merge`, `mesc_data_handling.extract_stim_frame`
- `RUN_ANALYSIS_PREP`: calls various analysis-related functions from `functions.py`
- `PLOTS`, `PLOT_BTW_EXP`, `RUN_CELLREG_PREP`, `RUN_CELLREG`, `RUN_CELLREG_ANALYSIS`: control plotting and CellReg-related processes

---

## `mesc_loader.py`

### `extract_useful_xml_params(xml_path)`
Parses a `.mesc` file and extracts useful metadata.

---

## `mesc_tiff_extract.py`

### `analyse_mesc_file()`
Extracts TIFF images, saves `mesc_data.npy`, `trigger.txt`, `fileId.txt`, `frameNo.txt`.

---

## `frequency_to_save.py`

### `frequency_electrodeRoi_to_save()`
Saves frequency and electrode ROI info into `.npy` files.

---

## `mesc_data_handling.py`

### `tiff_merge()`
Merges multiple TIFFs based on experimental parameters. Saves merged TIFFs and associated frequency and ROI info.

### `extract_stim_frame()`
Saves `FRAMENUM.NPY` and `STIMTIMES.NPY`.

---

## `suite2p_script.py`

### `run_suite2p()`
Runs Suite2p on the merged TIFF files using tailored parameters for GCaMP indicators.

---

## `functions.py`

### `stim_dur_val()`
Saves stimulation duration per merged TIFF to `stimDurations.npy`.

### `electrodeROI_val()`
Saves the selected electrode ROI number into `electrodeROI.npy`.

### `dist_vals()`
Calculates and saves distances between ROIs and the electrode ROI.

### `spontaneous_baseline_val()`
Calculates baseline from fixed window in spontaneous recordings.

### `baseline_val()`
Calculates F0 baseline before stim onset using `stimTimes.npy`.

### `activated_neurons_val()`
Detects activated ROIs per block. Saves `.npy` and CellReg `.mat` mask files.

### `timecourse_val()`
Analyzes per-trial traces and stimulation effects across time.

### `data_analysis_values()`
Generates multiple summary plots, e.g., active cell count, avg amplitude, etc.

### `plot_stim_traces()`
Generates stimulation-aligned plots and activation maps.

### `plot_across_experiments()`
Overlays stimulation response traces across experiments.

### `analyze_merged_activation_and_save()`
Block-wise activation detection across multiple stimulation files.

---

## `CoM.py`

### `calculate_center_of_mass()`
Uses ROI coordinates to compute spatial center of activation.

---

## `cellreg_process.py`

### `suite2p_to_cellreg_masks()`
Creates CellReg `.mat` masks from Suite2p output.

### `single_block_activation()`
Legacy method for per-stim activation detection and mask saving.

---

## `cellreg_analysis.py`

### `cellreg_analysis_overlap()`
Analyzes overlap from CellReg results. Outputs `session_pair_overlap.csv`.

---

## `cellreg_analysis.py`

### `run_cellreg_matlab()`
Runs MATLAB CellReg script via Python interface.

---

_Note: Many functions save data in `.npy`, `.csv`, `.svg`, `.mat` formats as part of the pipeline's modular output._
