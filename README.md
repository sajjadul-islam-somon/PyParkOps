# PyParkOps

A lightweight Streamlit app for automated parking operations using YOLO plate detection and OCR.

Features
- Scan vehicle images to detect license plates
- Track active parkings with entry times
- Compute basic parking fee (hourly rate)

Quick Start
1. Create a virtual environment and install dependencies:

   pip install -r requirements.txt

2. Run the app:

   streamlit run PyParkOps-App.py

Files
- `PyParkOps-App.py`: Main Streamlit application.
- `bd_plates_best_yolo26.pt`: Trained YOLO model for plate detection.
- `parking_database.csv`: CSV database (auto-created).
- `train_yolo26_object_detection.ipynb`: Notebook used for training the model.

Notes
- Keep upload images <= 2MB for reliable uploads.
- The app uses EasyOCR (Bengali + English) and a custom YOLO model.

License
- MIT License — modify as needed.

Contact
- For issues or contributions, open an issue or PR.
