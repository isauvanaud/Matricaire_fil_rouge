# Counting Metric Integration for Ultralytics YOLO

This guide explains how to enable **counting performance monitoring** during training and validation by replacing specific Ultralytics files.

## Overview

To visualize counting performance at each training epoch and on the validation set, certain Ultralytics source files must be modified.

As of **05/02/2026**:

- The counting metric is defined as:

  counting_ratio = number_of_predicted_boxes / number_of_ground_truth_boxes

- ⚠️ The **best model selected at the end of training** is **NOT chosen based on this ratio**.
- Model selection still relies on **mAP50–95**, which remains the default metric in YOLO.

---

## Installation Notes

If Ultralytics was installed via **Miniconda**, the package is typically located at:

miniconda3/lib/python3.13/site-packages/

Replace `python3.13` with your installed Python version.

---

## Required File Replacements

Navigate to:

ultralytics/utils/

Replace:

- metrics.py
- plotting.py

Then navigate to:

ultralytics/models/yolo/detect/

Replace:

- val.py

---

## Tested Environment

The modifications were tested with:

- Ultralytics 8.4.6  
- Python 3.13.9  
- PyTorch 2.9  

These changes should remain compatible with nearby versions.

---

## Notes

- These modifications add **counting performance monitoring**.
- They **do not alter** YOLO’s default model selection logic.
- It is recommended to keep backups of original files before replacing them.
