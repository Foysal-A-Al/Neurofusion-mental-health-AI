# Model Card

## Intended use
Research and education concerning longitudinal multimodal modelling, calibration, uncertainty, and explainability.

## Out-of-scope use
Diagnosis, treatment selection, emergency assessment, autonomous clinical decisions, insurance, employment, or access control.

## Models
Two sigmoid-calibrated logistic-regression baselines: multiclass mood-state classification and binary synthetic deterioration-label estimation. Preprocessing is fitted inside each calibration fold. Only the training partition is used for fitting/calibration; the patient-level test partition is held out.

## Evaluation
Patient-level hold-out evaluation reports accuracy, balanced accuracy, macro F1, ROC-AUC, and Brier score.

## Major limitations
The data are synthetic. Metrics demonstrate software operation, not clinical validity. The deterioration label is sampled from same-day simulator drivers and does not measure a future seven-day outcome. The retained API field name does not establish forecasting capability.

The output probability band uses an assumed effective sample size of 30; it is not an estimated individual confidence interval. Reference-based observed patterns are not model-derived attributions. There is no demonstrated clinical calibration, fairness audit, or external validation.
