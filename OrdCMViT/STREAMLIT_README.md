# OrdCMViT Streamlit Application

A web-based interface for breast cancer BI-RADS prediction using the OrdCMViT (Ordinal Cross-Modal Vision Transformer) model.

## Features

✨ **Key Features:**
- 🖼️ Upload ultrasound and mammogram images
- 🔮 Get instant BIRADS classification (1-5)
- 📊 View prediction confidence scores
- 📈 Visualize class probability distributions
- ✅ Compare against ground truth labels (optional)
- 💾 Export results to CSV
- 🚀 Multiple model checkpoint support

## Installation

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Ensure Streamlit is installed (already in requirements.txt)
pip install streamlit>=1.28.0
```

### 2. Verify Model Checkpoints

The app expects model checkpoints in the following structure:
```
ordcmvit_runs/
├── 1/
│   ├── ordcmvit/
│   │   └── checkpoints/
│   │       └── best.pth
│   └── checkpoints/
│       └── best.pth
├── 2/
└── ...
```

Trained models should be located in `./ordcmvit_runs/<run_id>/ordcmvit/checkpoints/best.pth` or `./ordcmvit_runs/<run_id>/checkpoints/best.pth`

## Running the Application

### Start the Streamlit Server

```bash
cd ./OrdCMViT
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage Guide

### Step 1: Select a Model
1. In the left sidebar, select a trained model from "Select Model Run" dropdown
2. Click "Load Model" button to load the checkpoint

### Step 2: Upload Patient Images

You have two options:

#### Option A: Upload Individual Images
1. Upload an **Ultrasound Image** (PNG, JPG, JPEG, BMP, TIFF, DICOM)
2. Upload a **Mammogram Image** (same formats)

#### Option B: Upload ZIP Folder
1. Prepare a ZIP file with this structure:
   ```
   patient_folder.zip
   ├── Ultrasound/
   │   └── image.png
   └── Mammogram/
       └── image.jpg
   ```
2. Upload the ZIP file
3. The app will automatically extract and load images

### Step 3: (Optional) Add Ground Truth

1. Check "I have ground truth BI-RADS" checkbox
2. Select the actual BI-RADS label (1-5) from dropdown
3. The app will show prediction accuracy

### Step 4: Run Prediction

1. Click the **"🔮 Predict BI-RADS"** button
2. Wait for inference to complete (~2-5 seconds)
3. View results:
   - **Predicted BI-RADS**: Classification result
   - **Confidence Score**: Prediction confidence (0-100%)
   - **Correctness**: Whether prediction matches ground truth (if provided)

### Step 5: View Results

The app displays:
- **Input Images**: Uploaded ultrasound and mammogram side-by-side
- **Prediction Card**: Summary of prediction and confidence
- **Class Probability Chart**: Bar chart showing probabilities for each BI-RADS class
- **Optional Ground Truth**: Comparison with actual label

### Step 6: Save Results

1. Click **"💾 Save Results to CSV"** button
2. Download CSV file with columns:
   - `patient_id`: Patient identifier (if provided)
   - `pred_birads`: Predicted BI-RADS classification
   - `true_birads`: Ground truth label (if provided)
   - `confidence`: Prediction confidence score
   - `correct`: Whether prediction was correct

## BI-RADS Classification Levels

| Class | BI-RADS | Description | Clinical Significance |
|-------|---------|-------------|----------------------|
| 🟢 | 1 | Normal | No findings of concern |
| 🔵 | 2 | Benign | Benign findings (no need for intervention) |
| 🟠 | 3 | Probably Benign | Needs short-term follow-up (6 months) |
| 🟠 | 4 | Suspicious | Malignancy likely; biopsy recommended |
| 🔴 | 5 | Malignant | High likelihood of malignancy |

## Input Requirements

### Image Specifications

- **Formats Supported**: PNG, JPG, JPEG, BMP, TIFF, DICOM
- **Ultrasound Image**: 
  - Recommended: 384×384 pixels (will be resized if different)
  - Should show breast tissue with lesion or area of interest
  
- **Mammogram Image**:
  - Recommended: 384×384 pixels (will be resized if different)
  - Should show breast tissue with lesion or area of interest

### Preprocessing Applied Automatically

The app applies the following preprocessing:
1. **Normalization**: ImageNet standardization (mean/std)
2. **Resizing**: Ultrasound to 384×384, Mammogram to 384×384
3. **Tensor Conversion**: Conversion to PyTorch tensor format

## Troubleshooting

### "No trained models found"
- Ensure you have trained a model using `python main.py`
- Check that checkpoints exist in `./ordcmvit_runs/`
- Verify the directory structure matches expected format

### "Failed to load model"
- Check checkpoint file exists and is not corrupted
- Ensure PyTorch and CUDA (if using GPU) are properly installed
- Try running a test inference from command line

### Images not loading
- Verify image files are in supported formats
- Check that DICOM files are valid (may need to install pydicom: `pip install pydicom`)
- Ensure images are not corrupted

### Slow inference
- First run may be slower (model compilation)
- Subsequent predictions should be faster (cached model)
- If using CPU, consider using GPU (ensure CUDA is available)

## Advanced Usage

### Batch Processing Multiple Patients

Create a Python script to process multiple patients:

```python
import pandas as pd
from pathlib import Path
import torch
from streamlit_app import load_model, predict_patient, get_device_cached

# Load model once
device = get_device_cached()
model = load_model("./ordcmvit_runs/4/ordcmvit/checkpoints/best.pth", device)

# Process multiple patients
results = []
patient_dirs = Path("./patients").iterdir()

for patient_dir in patient_dirs:
    us_image = load_image(patient_dir / "Ultrasound" / "image.png")
    mm_image = load_image(patient_dir / "Mammogram" / "image.jpg")
    
    result = predict_patient(us_image, mm_image, model, device, cfg)
    results.append({
        "patient_id": patient_dir.name,
        "pred_birads": result["pred_birads"],
        "confidence": result["confidence"],
    })

df = pd.DataFrame(results)
df.to_csv("batch_predictions.csv", index=False)
```

### Model Performance

The OrdCMViT model achieves:
- **Accuracy**: ~75% on test set
- **Quadratic Weighted Kappa (QWK)**: ~0.72 (measures ordinal classification quality)
- **AUC (macro, One-vs-Rest)**: ~0.85

## Model Architecture

**OrdCMViT** combines:
- **Vision Transformer (ViT-Small)**: Feature extraction from US and MM
- **Ordinal Head**: CORAL (Consistent Ordinal Regression) for BI-RADS classification
- **Cross-Modal Fusion**: Attention-based fusion of ultrasound and mammogram features

Key advantages:
- Treats BI-RADS as ordinal (penalizes distant errors more)
- Multi-modal fusion for improved accuracy
- Weakly-supervised localization (CAMs)

## Citation

If you use this application in research, please cite:

```bibtex
@article{ordcmvit2024,
  title={OrdCMViT: Ordinal Cross-Modal Vision Transformer for Breast Cancer BI-RADS Grading},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Add your license information here]

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review model training logs in `./ordcmvit_runs/<run_id>/logs/`
3. Check Streamlit documentation: https://docs.streamlit.io/

## Related Files

- Training script: `main.py`
- Model architecture: `src/models/ordcmvit.py`
- Dataset handler: `src/data/bcmid.py`
- Metrics: `src/utils/metrics.py`
