# PyParkOps

[![Live Demo](https://img.shields.io/badge/Live-Demo-Replace%20URL-brightgreen)](https://example.com) [![Streamlit](https://img.shields.io/badge/Framework-Streamlit-orange)](https://streamlit.io) [![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org) [![Model](https://img.shields.io/badge/Model-YOLOv8-red)](https://github.com/ultralytics/) [![OCR](https://img.shields.io/badge/OCR-EasyOCR-lightgrey)](https://github.com/JaidedAI/EasyOCR) [![License-MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p align="center">
  <img src="pyparkops-logo.png" alt="PyParkOps logo" width="140" />
</p>

A compact, demo-ready Streamlit app that detects license plates from uploaded photos, records entry times, and computes simple parking fees.

Quick links: [Run locally](#quick-start) • [Live demo](https://example.com)

Why you'll like it
- Fast image-based entry/exit workflow
- Minimal dependencies and CSV-based storage for easy testing
- Supports Bengali + English OCR via EasyOCR

## Quick Start

1. (Optional) create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies and run:

```powershell
pip install -r requirements.txt
streamlit run PyParkOps-App.py
```

## Live Demo

- Click the green badge at the top to open a deployed instance. Replace the badge URL with your deployed app URL to make it live.

## How it works (short)
- A YOLO model finds the plate region in uploaded images.
- EasyOCR (bn + en) extracts the plate text.
- The app records entries in `parking_database.csv` and computes fees on exit.

## Files
- `PyParkOps-App.py` — Main Streamlit application
- `bd_plates_best_yolo26.pt` — YOLO model file (required)
- `parking_database.csv` — Active parking DB (auto-created)
- `train_yolo26_object_detection.ipynb` — Training notebook

## Tips
- Keep upload images <= 2MB for reliable uploads.
- If `pyparkops-logo.png` is missing, the app uses a text badge instead.

## Replace the Live Demo
- Edit the top badge link and replace `https://example.com` with your deployed app URL.

## License
- MIT

## Contributing
- Open an issue for bugs or feature requests, or submit a focused PR.

Thanks for using PyParkOps! 🚗
