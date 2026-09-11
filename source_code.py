# -*- coding: utf-8 -*-
"""
DIP Mini Project
Title:
Restoration of Fingerprint Images for Forensic Matching

Pipeline:
Input -> Simulated Degradation -> Median Filtering + CLAHE
-> Adaptive Thresholding -> Morphological Cleanup
-> Skeletonization -> Minutiae Extraction -> Experimental Matching

This script is a cleaned and organized version of the Colab implementation.
"""

import os
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize, remove_small_objects
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


# ============================================================
# 1. DATASET SETUP
# ============================================================

REPO_URL = "https://github.com/YogeshMoun/Minutiae-Extraction-and-Matching.git"
DATASET_PATH = "/content/Minutiae-Extraction-and-Matching"

if not os.path.exists(DATASET_PATH):
    subprocess.run(
        ["git", "clone", REPO_URL, DATASET_PATH],
        check=True
    )

fingerprint_files = []

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".tif", ".tiff")):
            fingerprint_files.append(os.path.join(root, file))

fingerprint_files.sort()

print("Number of fingerprint images found:", len(fingerprint_files))
print("First 10 images:")
for path in fingerprint_files[:10]:
    print(path)


# ============================================================
# 2. LOAD SAMPLE FINGERPRINTS
# ============================================================

image1_path = os.path.join(DATASET_PATH, "101_1.tif")
image2_path = os.path.join(DATASET_PATH, "101_2.tif")
image3_path = os.path.join(DATASET_PATH, "102_1.tif")

image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None or image3 is None:
    raise FileNotFoundError("One or more required fingerprint images could not be loaded.")

print("\nImage shapes:")
print("101_1:", image1.shape)
print("101_2:", image2.shape)
print("102_1:", image3.shape)


# ============================================================
# 3. SIMULATED IMAGE DEGRADATION
# ============================================================

# The project uses controlled degradation to evaluate restoration.
# Gaussian blur + Gaussian noise + contrast reduction simulate
# common image-quality problems.

np.random.seed(42)

original = image1.copy()

blurred = cv2.GaussianBlur(original, (7, 7), 2)

noise = np.random.normal(0, 15, blurred.shape)
noisy = blurred.astype(np.float32) + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

degraded = cv2.convertScaleAbs(
    noisy,
    alpha=0.65,
    beta=45
)


# ============================================================
# 4. RESTORATION COMPARISON
# ============================================================

# Median filtering removes noise while preserving ridge structure.
median_filtered = cv2.medianBlur(degraded, 5)

# Gaussian filtering is included for comparison.
gaussian_filtered = cv2.GaussianBlur(degraded, (5, 5), 0)

# CLAHE improves local contrast.
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_restored = clahe.apply(median_filtered)

print("\nRestoration methods:")
print("- Median filtering")
print("- Gaussian filtering")
print("- Median + CLAHE (final restoration)")


# ============================================================
# 5. GABOR ENHANCEMENT COMPARISON
# ============================================================

# A fixed-orientation Gabor filter is included as a comparison.
# It is not selected as the final restoration method because
# fingerprint ridge orientation changes across the image.

normalized = cv2.normalize(
    median_filtered,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

gabor_kernel = cv2.getGaborKernel(
    (21, 21),
    5,
    0,
    10,
    0.5,
    0,
    ktype=cv2.CV_32F
)

gabor_result = cv2.filter2D(
    normalized,
    cv2.CV_32F,
    gabor_kernel
)

gabor_result = cv2.normalize(
    gabor_result,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# ============================================================
# 6. RESTORATION EVALUATION
# ============================================================

mse_degraded = mean_squared_error(original, degraded)
psnr_degraded = peak_signal_noise_ratio(original, degraded)
ssim_degraded = structural_similarity(original, degraded)

mse_gabor = mean_squared_error(original, gabor_result)
psnr_gabor = peak_signal_noise_ratio(original, gabor_result)
ssim_gabor = structural_similarity(original, gabor_result)

mse_clahe = mean_squared_error(original, clahe_restored)
psnr_clahe = peak_signal_noise_ratio(original, clahe_restored)
ssim_clahe = structural_similarity(original, clahe_restored)

restoration_results = pd.DataFrame({
    "Method": [
        "Degraded",
        "Gabor",
        "Median + CLAHE"
    ],
    "MSE": [
        mse_degraded,
        mse_gabor,
        mse_clahe
    ],
    "PSNR (dB)": [
        psnr_degraded,
        psnr_gabor,
        psnr_clahe
    ],
    "SSIM": [
        ssim_degraded,
        ssim_gabor,
        ssim_clahe
    ]
})

print("\n========== RESTORATION EVALUATION ==========")
print(restoration_results.round(4).to_string(index=False))


# ============================================================
# 7. FINAL SEGMENTATION PIPELINE
# ============================================================

def preprocess_fingerprint(image):
    """
    Restore/enhance a fingerprint and produce a skeleton.

    Steps:
    1. Median filtering
    2. CLAHE local contrast enhancement
    3. Adaptive thresholding
    4. Polarity correction
    5. Morphological opening and closing
    6. Small-object removal
    7. Skeletonization
    """

    # Noise removal
    denoised = cv2.medianBlur(image, 5)

    # Local contrast enhancement
    clahe_local = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe_local.apply(denoised)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5
    )

    # Make ridges white
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # Remove very small connected components
    binary_bool = remove_small_objects(
        binary > 0,
        min_size=50
    )

    # Skeletonization
    skeleton = skeletonize(binary_bool)

    return enhanced, binary, skeleton


# Process all three sample images using the same final pipeline.
restored_1, binary_1, skeleton_1 = preprocess_fingerprint(image1)
restored_2, binary_2, skeleton_2 = preprocess_fingerprint(image2)
restored_3, binary_3, skeleton_3 = preprocess_fingerprint(image3)


# ============================================================
# 8. MINUTIAE EXTRACTION
# ============================================================

def extract_clean_minutiae(skeleton, border=25, min_distance=12):
    """
    Extract ridge endings and bifurcations using the
    Crossing Number method.

    Ridge ending: Crossing Number = 1
    Bifurcation: Crossing Number = 3
    """

    skel = skeleton.astype(np.uint8)
    h, w = skel.shape

    padded = np.pad(
        skel,
        ((1, 1), (1, 1)),
        mode="constant"
    )

    endings = []
    bifurcations = []

    for y in range(1, h + 1):
        for x in range(1, w + 1):

            if padded[y, x] == 0:
                continue

            neighbors = [
                padded[y - 1, x],
                padded[y - 1, x + 1],
                padded[y, x + 1],
                padded[y + 1, x + 1],
                padded[y + 1, x],
                padded[y + 1, x - 1],
                padded[y, x - 1],
                padded[y - 1, x - 1]
            ]

            crossing_number = sum(
                abs(
                    int(neighbors[i]) -
                    int(neighbors[(i + 1) % 8])
                )
                for i in range(8)
            ) / 2

            px = x - 1
            py = y - 1

            # Ignore unreliable points near the image boundary.
            if (
                px <= border or
                px >= w - border or
                py <= border or
                py >= h - border
            ):
                continue

            if crossing_number == 1:
                endings.append((px, py))

            elif crossing_number == 3:
                bifurcations.append((px, py))

    def filter_close_points(points):
        if len(points) == 0:
            return []

        points = np.array(points)
        selected = [points[0]]

        for point in points[1:]:
            distances = cdist(
                [point],
                np.array(selected)
            )[0]

            if np.min(distances) >= min_distance:
                selected.append(point)

        return [tuple(point) for point in selected]

    endings = filter_close_points(endings)
    bifurcations = filter_close_points(bifurcations)

    return endings, bifurcations


final_endings_1, final_bifurcations_1 = extract_clean_minutiae(skeleton_1)
final_endings_2, final_bifurcations_2 = extract_clean_minutiae(skeleton_2)
final_endings_3, final_bifurcations_3 = extract_clean_minutiae(skeleton_3)


def print_minutiae_result(name, endings, bifurcations):
    total = len(endings) + len(bifurcations)

    print(f"\n========== {name} ==========")
    print("Ridge endings:", len(endings))
    print("Bifurcations:", len(bifurcations))
    print("Total minutiae:", total)


print_minutiae_result(
    "101_1",
    final_endings_1,
    final_bifurcations_1
)

print_minutiae_result(
    "101_2",
    final_endings_2,
    final_bifurcations_2
)

print_minutiae_result(
    "102_1",
    final_endings_3,
    final_bifurcations_3
)


# ============================================================
# 9. EXPERIMENTAL MINUTIAE MATCHING
# ============================================================

def improved_matching_score(points1, points2, threshold=10):
    """
    Simple spatial nearest-neighbour matching.

    NOTE:
    This is an experimental demonstration, not a forensic-grade
    fingerprint matcher. It does not model rotation, scale,
    ridge orientation, or a full fingerprint matching algorithm.
    """

    if len(points1) == 0 or len(points2) == 0:
        return 0, 0, 0

    p1 = np.array(points1)
    p2 = np.array(points2)

    distances = cdist(p1, p2)

    minimum_distances = np.min(distances, axis=1)

    matches = np.sum(
        minimum_distances <= threshold
    )

    score_relative_to_larger_set = (
        matches / max(len(p1), len(p2))
    ) * 100

    return matches, score_relative_to_larger_set, minimum_distances


# Same finger: 101_1 vs 101_2
ending_matches_same, ending_score_same, _ = improved_matching_score(
    final_endings_1,
    final_endings_2
)

bif_matches_same, bif_score_same, _ = improved_matching_score(
    final_bifurcations_1,
    final_bifurcations_2
)

# Different finger: 101_1 vs 102_1
ending_matches_diff, ending_score_diff, _ = improved_matching_score(
    final_endings_1,
    final_endings_3
)

bif_matches_diff, bif_score_diff, _ = improved_matching_score(
    final_bifurcations_1,
    final_bifurcations_3
)


print("\n===== SAME FINGER: 101_1 vs 101_2 =====")
print("Ridge ending matches:", ending_matches_same)
print("Ridge ending score:", round(ending_score_same, 2), "%")
print("Bifurcation matches:", bif_matches_same)
print("Bifurcation score:", round(bif_score_same, 2), "%")

print("\n===== DIFFERENT FINGER: 101_1 vs 102_1 =====")
print("Ridge ending matches:", ending_matches_diff)
print("Ridge ending score:", round(ending_score_diff, 2), "%")
print("Bifurcation matches:", bif_matches_diff)
print("Bifurcation score:", round(bif_score_diff, 2), "%")


# ============================================================
# 10. VISUAL RESULTS
# ============================================================

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(original, cmap="gray")
plt.title("1. Original Fingerprint")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(degraded, cmap="gray")
plt.title("2. Simulated Degradation")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(clahe_restored, cmap="gray")
plt.title("3. Restored: Median + CLAHE")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(binary_1, cmap="gray")
plt.title("4. Adaptive Segmentation")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(skeleton_1, cmap="gray")
plt.title("5. Skeletonization")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(restored_1, cmap="gray")

if final_endings_1:
    endings = np.array(final_endings_1)
    plt.scatter(
        endings[:, 0],
        endings[:, 1],
        c="red",
        s=15,
        label="Ridge Endings"
    )

if final_bifurcations_1:
    bifurcations = np.array(final_bifurcations_1)
    plt.scatter(
        bifurcations[:, 0],
        bifurcations[:, 1],
        c="blue",
        s=20,
        label="Bifurcations"
    )

plt.title("6. Extracted Minutiae")
plt.legend(fontsize=8)
plt.axis("off")

plt.suptitle(
    "Fingerprint Restoration and Minutiae Extraction Pipeline",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ============================================================
# 11. RESTORATION METRIC GRAPHS
# ============================================================

methods = restoration_results["Method"]

plt.figure(figsize=(8, 5))
plt.bar(methods, restoration_results["MSE"])
plt.ylabel("MSE")
plt.title("MSE Comparison")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(methods, restoration_results["PSNR (dB)"])
plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(methods, restoration_results["SSIM"])
plt.ylabel("SSIM")
plt.title("SSIM Comparison")
plt.ylim(0, 1)
plt.show()


# ============================================================
# 12. PROJECT CONCLUSION OUTPUT
# ============================================================

print("\n========== PROJECT SUMMARY ==========")
print("Final restoration method: Median Filtering + CLAHE")
print("Final segmentation: Adaptive Thresholding + Morphology")
print("Final representation: Skeletonized fingerprint")
print("Feature extraction: Crossing Number minutiae extraction")
print("Matching: Experimental nearest-neighbour spatial matching")
print("\nImportant limitation:")
print(
    "The matching experiment is not forensic-grade because the simple "
    "spatial matcher does not compensate for rotation, scale, alignment, "
    "ridge orientation, or other fingerprint matching factors."
)
