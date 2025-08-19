# Astroalign-based Preprocessing & Stacking Pipeline

A Python project that preprocesses, cleans, filters, aligns, and stacks astronomical images using **astroalign**. The pipeline performs background subtraction, star (source) detection, morphological & size filtering, and outputs quality‑controlled frames. After cleaning, stars are aligned across frames and the frames can be combined using several stacking methods.

---

## Project Structure

```
├── data/              # Raw input images (FITS, PNG, JPG, TIFF)
├── darkfield/         # Matching dark frames for the raw images
├── data_cleaned/      # Output folder for cleaned images
├── clean_images.py    # Main preprocessing / cleaning script
├── model.py           # Alignment and stacking script
└── README.md          # This README
```

---

## Requirements

* Python 3.7 or newer

Install required packages:

```bash
pip install numpy astroalign astropy pillow sep scipy
```

---

## Getting Started

1. **Prepare directories**: Place your raw images under `data/` and (optionally) matching dark frames under `darkfield/`.
2. **Configure parameters**: Open `clean_images.py` (or `data_process.py` if present) and adjust the parameters in the configuration section to match your data.
3. **Run cleaning**: Execute the cleaning script to generate a quality‑filtered dataset in `data_cleaned/`.
4. **Run alignment & stacking**: Open `model.py`, point it to the cleaned dataset (`data_cleaned/`), and run. The first image in the list is used as the **target** reference; all other frames are registered to this coordinate system.
5. **Review outputs**: Stacked images and a preview mosaic (showing all stacking modes) are saved under a `results/` folder.

> **Note**: If you use non‑FITS formats (PNG/JPG/TIFF), they are converted to grayscale before processing.

---

## Configuration (Defaults)

| Parameter            | Description                                     | Default        |
| -------------------- | ----------------------------------------------- | -------------- |
| `INPUT_DIR`          | Folder with raw images                          | `data`         |
| `OUTPUT_DIR`         | Folder for cleaned images                       | `data_cleaned` |
| `DARKFIELD_DIR`      | Folder with dark frames                         | `darkfield`    |
| `MIN_STAR_COUNT`     | Minimum detected stars required to keep a frame | `1000`         |
| `THRESHOLD_SIGMA`    | Detection threshold (in σ)                      | `10.0`         |
| `FWHM0`              | Expected star FWHM (pixels)                     | `3.0`          |
| `ELLIP_MAX`          | Max allowed ellipticity (1 - b/a)               | `0.5`          |
| `SATURATION_LIMIT`   | Max ADU before saturation                       | `65535`        |
| `MINAREA`            | Minimum connected pixels for `sep` objects      | `20`           |
| `OUTLIER_MAD_THRESH` | MAD multiplier for FWHM outlier rejection       | `3.0`          |

---

## Preprocessing & Cleaning

**Script:** `clean_images.py` (or `data_process.py` in some setups)

* Processes images in `INPUT_DIR` one‑by‑one.
* **FITS**: reads with `astropy.io.fits.getdata`. **PNG/JPG/TIFF**: loads with `PIL.Image` and converts to grayscale.
* If a matching dark frame exists in `DARKFIELD_DIR`, subtracts it **pixel‑wise** from the raw image.
* Builds a background model using **`sep.Background`** and subtracts it to enhance star signals.
* Optionally extracts the central **1000×1000** region for debugging/tuning detection parameters.
* Applies **Gaussian smoothing** (`σ = 1.0`) to reduce high‑frequency noise.
* Runs **`sep.extract`** to detect sources; key parameters are the signal threshold (in σ) and the minimum connected area.
* Estimates **FWHM** per source from measured semi‑major/minor axes; uses **median** and **MAD** to remove outlier sources.
* Computes **ellipticity** and **peak ADU** for each detection. If the frame fails FWHM / ellipticity / saturation criteria, the **entire frame** is rejected.
* Saves cleaned images and copies accepted frames to `OUTPUT_DIR`.

---

## Alignment & Stacking

**Script:** `model.py`

* Load the **cleaned dataset** (ensure `model.py` points to `data_cleaned/`).
* Use the **first image** as the target reference; register subsequent frames to this coordinate system using **astroalign**.
* Compute stacked products using the modes listed below.
* Save per‑mode results and generate a **preview mosaic** summarizing all modes in the `results/` folder.

### Supported stacking modes

* **Mean Stack** — Pixel‑wise arithmetic mean. Good reduction of random noise.
* **Median Stack** — Pixel‑wise median. Robust to transient bright/dark artifacts; may smooth fine details slightly.
* **Max Stack** — Pixel‑wise maximum. Emphasizes bright transient events.
* **Min Stack** — Pixel‑wise minimum. Useful for background estimation or highlighting persistent dark features.
* **Std Dev** — Pixel‑wise standard deviation. Visualizes frame‑to‑frame variability and transient noise.
* **Sigma‑Clipped Stack** — For each pixel, drop values more than *Nσ* from the mean (e.g., 2.5σ), then average the remainder. Combines smoothing with robustness to outliers and often produces the cleanest result.

---

## Outputs

* Cleaned, quality‑controlled frames in `data_cleaned/`.
* Stacked images for each mode in `results/`.
* A preview image showing all stacking modes side‑by‑side for quick comparison.

---

## Tips & Good Practices

* Ensure dark frames match the **exposure**, **gain/ISO**, **sensor temperature**, and **binning** of the corresponding light frames when possible.
* Tune `THRESHOLD_SIGMA`, `MINAREA`, and `FWHM0` to your optics and seeing conditions.
* If too many frames are rejected, consider loosening `ELLIP_MAX` or lowering `THRESHOLD_SIGMA` incrementally.
* Use the 1000×1000 debug region to quickly iterate on detection parameters before processing full frames.

---

## License

This project is released under the **MIT License** — feel free to use and adapt it in your projects.

---

## Acknowledgments

* [astroalign](https://github.com/toros-astro/astroalign)
* [Astropy](https://www.astropy.org/)
* [SEP (Source Extractor as a library)](https://sep.readthedocs.io/)
* [Pillow](https://python-pillow.org/)
