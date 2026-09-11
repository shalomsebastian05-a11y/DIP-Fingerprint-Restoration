# DIP Fingerprint Restoration

Digital Image Processing mini project for fingerprint image restoration, segmentation, minutiae extraction and experimental matching.

## Problem Statement

Fingerprint images captured under poor conditions may contain blur, noise and low contrast, which can make ridge structures difficult to analyze. This project applies Digital Image Processing techniques to restore fingerprint images, segment the ridge pattern, extract minutiae and perform experimental matching.

## Objectives

- Add simulated degradation to fingerprint images.
- Compare Median and Gaussian filtering.
- Apply Gabor filtering as a comparative experiment.
- Restore the fingerprint using Median filtering and CLAHE.
- Segment fingerprint ridges using adaptive thresholding.
- Apply morphological operations and skeletonization.
- Extract ridge endings and bifurcations using the Crossing Number method.
- Perform experimental fingerprint matching.

## Dataset

The project uses 170 TIFF fingerprint images from the selected GitHub dataset repository.

The images are fingerprint samples such as `101_1.tif`, `101_2.tif`, `102_1.tif`, etc.

The degradation used in this project is simulated using Gaussian blur, Gaussian noise and contrast reduction.

## Methodology

The processing pipeline is:

Original Image  
↓  
Simulated Degradation  
↓  
Median Filtering  
↓  
CLAHE Enhancement  
↓  
Adaptive Thresholding  
↓  
Morphological Processing  
↓  
Skeletonization  
↓  
Minutiae Extraction  
↓  
Experimental Matching

## Techniques Used

- Gaussian Blur
- Gaussian Noise
- Median Filtering
- Gaussian Filtering
- Gabor Filtering
- CLAHE
- Adaptive Thresholding
- Otsu Thresholding
- Morphological Opening and Closing
- Skeletonization
- Crossing Number Minutiae Extraction
- Nearest-Neighbor Matching
- ECC Alignment Experiment
- MSE, PSNR and SSIM

## Results

For the sample fingerprint `101_1.tif`:

| Method | MSE | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Degraded | 2725.07 | 13.78 | 0.5751 |
| Gabor | 13084.59 | 6.96 | 0.4123 |
| Median + CLAHE | 2099.11 | 14.91 | 0.7276 |

Median filtering followed by CLAHE produced the best restoration among the tested methods.

## Minutiae Extraction

Example extracted minutiae counts:

| Image | Ridge Endings | Bifurcations | Total |
|---|---:|---:|---:|
| 101_1 | 109 | 14 | 123 |
| 101_2 | 72 | 3 | 75 |
| 102_1 | 159 | 18 | 177 |

## Limitations

- The degradation is simulated rather than collected from real forensic latent fingerprints.
- The experimental matching method is sensitive to alignment and segmentation quality.
- False minutiae may occur due to noise, broken ridges and segmentation errors.
- The matching results should not be considered forensic-grade biometric identification.
- The dataset does not provide verified forensic ground-truth annotations for all processing stages.

## Future Scope

- Use real latent fingerprint datasets.
- Apply orientation-field based Gabor filtering.
- Improve ridge segmentation.
- Remove spurious minutiae using stronger post-processing.
- Implement rotation- and translation-invariant minutiae matching.
- Use larger datasets with ground-truth minutiae annotations.
- Compare the traditional DIP pipeline with modern deep-learning approaches.

## Technologies

- Python
- OpenCV
- NumPy
- SciPy
- scikit-image
- Matplotlib
- Pandas
- Google Colab

## Project Structure

```text
DIP-Fingerprint-Restoration/
├── README.md
├── source_code.py
├── requirements.txt
├── images/
├── screenshots/
└── report/
