# 📊 OrdCMViT BIRADS Prediction System - Complete Guide

## Overview

**OrdCMViT** is a state-of-the-art cross-modal vision transformer for breast cancer BI-RADS classification. This system provides two interfaces:

1. **🌐 Streamlit Web App** - Interactive single-patient prediction
2. **⚙️ Batch Processing Script** - Efficient multi-patient processing

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8+
- NVIDIA GPU (optional, CPU supported)
- Trained model checkpoint in `./ordcmvit_runs/`

### Installation
```bash
cd ./OrdCMViT
pip install -r requirements.txt
```

### Launch Web App
```bash
streamlit run streamlit_app.py
```

### Batch Processing
```bash
python batch_predict.py --input_dir ./patients --output_csv results.csv
```

---

## 📱 Interface 1: Streamlit Web App

### Purpose
Interactive web interface for clinicians and researchers to:
- Upload patient images
- Get instant predictions
- Validate against ground truth
- Download results

### How to Use

#### Step 1: Start the App
```bash
streamlit run streamlit_app.py
```
Opens automatically at `http://localhost:8501`

#### Step 2: Select Model
- Sidebar → "Select Model Run" dropdown
- Choose trained model
- Click "Load Model"

#### Step 3: Upload Images
**Option A - Individual Upload:**
```
Upload Ultrasound Image (PNG, JPG, JPEG, TIFF, DICOM)
Upload Mammogram Image (PNG, JPG, JPEG, TIFF, DICOM)
```

**Option B - ZIP Folder:**
```
patient_folder.zip
├── Ultrasound/
│   └── image.png
└── Mammogram/
    └── image.jpg
```

#### Step 4: Predict
- Click **"🔮 Predict BI-RADS"** button
- Wait for inference (~2-5 seconds)

#### Step 5: View Results
- **Predicted BI-RADS**: Classification (1-5)
- **Confidence Score**: Probability (0-100%)
- **Class Probabilities**: Bar chart of all classes
- **(Optional) Ground Truth**: Validation results

#### Step 6: Save Results
- Click **"💾 Save Results to CSV"**
- Download patient record

### Features
- ✅ Real-time predictions
- ✅ Confidence scores
- ✅ Ground truth comparison
- ✅ Results export
- ✅ Model switching
- ✅ Image preview

### Screenshots
```
┌─────────────────────────────────────┐
│  OrdCMViT: BIRADS Predictor    🏥  │
├─────────────────────────────────────┤
│ ⬆️ Ultrasound         ⬆️ Mammogram   │
│ [Upload]              [Upload]       │
├─────────────────────────────────────┤
│ 🔮 Predict BIRADS                   │
├─────────────────────────────────────┤
│ Predicted: BI-RADS 4                │
│ Confidence: 87.3%                   │
│ Status: ✅ Correct                  │
│                                     │
│ Probabilities:                      │
│ BIRADS 1: ░░ 2.1%                   │
│ BIRADS 2: ░░░ 5.8%                  │
│ BIRADS 3: ░░░░░ 8.6%                │
│ BIRADS 4: ██████████ 87.3% ←        │
│ BIRADS 5: ░░ 3.2%                   │
└─────────────────────────────────────┘
```

---

## ⚙️ Interface 2: Batch Processing

### Purpose
Efficient processing of multiple patients:
- Process folders of patients
- Generate results CSV
- Evaluate accuracy
- Track clinical workflow

### How to Use

#### Step 1: Prepare Patient Folders
```
patients/
├── patient_001/
│   ├── Ultrasound/
│   │   └── image.png
│   └── Mammogram/
│       └── image.jpg
├── patient_002/
│   ├── Ultrasound/
│   │   └── image.jpg
│   └── Mammogram/
│       └── image.jpg
└── ...
```

#### Step 2: (Optional) Prepare Ground Truth CSV
```
patient_id,birads
patient_001,2
patient_002,4
patient_003,5
...
```

#### Step 3: Run Batch Processing
```bash
# Basic: Process all patients
python batch_predict.py --input_dir ./patients

# With ground truth evaluation
python batch_predict.py \
  --input_dir ./patients \
  --output_csv results.csv \
  --ground_truth_csv ground_truth.csv \
  --model_run 4
```

#### Step 4: Check Results
Generated files:
- **results.csv**: Predictions for all patients
- **results.json**: Detailed output (optional)

#### Step 5: Review Results
```bash
# View CSV
cat results.csv

# Example output:
patient_id,pred_birads,confidence,us_found,mm_found,success
patient_001,2,0.892,True,True,True
patient_002,4,0.873,True,True,True
patient_003,5,0.921,True,True,True
...
```

### Features
- ✅ Batch processing (unlimited patients)
- ✅ Automatic image discovery
- ✅ Error handling & reporting
- ✅ Progress bar
- ✅ Ground truth evaluation
- ✅ Results summary
- ✅ Detailed JSON export

### Command-Line Arguments

```bash
python batch_predict.py --help

Options:
  --input_dir PATH              Input directory with patient folders [required]
  --output_csv PATH             Output CSV file (default: results.csv)
  --output_json PATH            Output JSON file (optional)
  --model_run INT               Model run number (default: 4)
  --ground_truth_csv PATH       Ground truth CSV for evaluation
  --device STR                  Device: cuda or cpu (default: auto-detect)
```

### Example Workflows

**Workflow 1: Single Patient Batch**
```bash
python batch_predict.py --input_dir ./single_patient --model_run 4
```

**Workflow 2: Clinical Trial (with GT)**
```bash
python batch_predict.py \
  --input_dir ./trial_patients \
  --output_csv trial_results.csv \
  --ground_truth_csv trial_ground_truth.csv \
  --model_run 6
```

**Workflow 3: Production Deployment**
```bash
# Process all patients daily
python batch_predict.py \
  --input_dir /data/daily_patients \
  --output_csv /output/predictions_$(date +%Y%m%d).csv \
  --device cuda \
  --model_run 4
```

---

## 📊 Understanding Results

### BI-RADS Classification

| Level | Name | Description | Action | Color |
|-------|------|-------------|--------|-------|
| **1** | Normal | No findings | Routine screening | 🟢 |
| **2** | Benign | Benign findings | No intervention | 🔵 |
| **3** | Probably Benign | Likely benign | 6-month follow-up | 🟠 |
| **4** | Suspicious | Concerning features | Biopsy recommended | 🟠 |
| **5** | Malignant | High malignancy | Treat as cancer | 🔴 |

### Output Fields (CSV)

```
patient_id          - Patient identifier
pred_birads         - Predicted BI-RADS (1-5)
confidence          - Prediction confidence (0-1)
prob_birads_1       - Probability of BI-RADS 1
prob_birads_2       - Probability of BI-RADS 2
prob_birads_3       - Probability of BI-RADS 3
prob_birads_4       - Probability of BI-RADS 4
prob_birads_5       - Probability of BI-RADS 5
us_found            - Ultrasound image found (True/False)
mm_found            - Mammogram image found (True/False)
success             - Processing successful (True/False)
true_birads         - Ground truth (if provided)
```

### Performance Metrics

**OrdCMViT Performance on BCMID Dataset:**
- Accuracy: 75.2%
- Quadratic Weighted Kappa (QWK): 0.722
- AUC (macro OvR): 0.847

---

## 🔧 Advanced Usage

### Custom Model Checkpoint

```bash
# Use different model run
streamlit run streamlit_app.py -- --model_run 5
```

### GPU Acceleration

```bash
# Force CUDA
python batch_predict.py --input_dir ./patients --device cuda

# Force CPU
python batch_predict.py --input_dir ./patients --device cpu
```

### Hyperparameter Tuning

Edit `config.py` to modify:
- Image resolution (us_size, mm_size)
- Augmentation parameters
- Model architecture
- Training parameters

### Integration with Existing Systems

**Python API:**
```python
from streamlit_app import load_model, predict_patient
from config import Config
from src.utils.seed import get_device
import cv2

# Setup
device = get_device()
cfg = Config()
model = load_model("./ordcmvit_runs/4/ordcmvit/checkpoints/best.pth", device)

# Load images
us = cv2.imread("ultrasound.png")
mm = cv2.imread("mammogram.jpg")
us = cv2.cvtColor(us, cv2.COLOR_BGR2RGB)
mm = cv2.cvtColor(mm, cv2.COLOR_BGR2RGB)

# Predict
result = predict_patient(us, mm, model, device, cfg)
print(f"Prediction: BI-RADS {result['pred_birads']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📈 Monitoring & Logging

### Streamlit Logs
```bash
# Check for errors
cat ~/.streamlit/logs/
```

### Batch Processing Logs
```bash
# Verbose output with progress
python batch_predict.py --input_dir ./patients 2>&1 | tee batch_log.txt
```

### TensorBoard (Training)
```bash
# View training metrics
tensorboard --logdir ./ordcmvit_runs/4/ordcmvit/logs
```

---

## 🐛 Troubleshooting

### Common Issues

**"No trained models found"**
- Solution: Run `python main.py` to train first
- Check `./ordcmvit_runs/` directory exists

**"Model fails to load"**
- Solution: Verify checkpoint path and file integrity
- Re-download checkpoint if corrupted

**"Images won't upload"**
- Solution: Convert to PNG or JPG format
- Check image resolution (ideally ~384×384)

**"Prediction is slow"**
- Solution: Expected ~2-5 seconds
- First inference slower (model compilation)
- Use CUDA GPU for faster inference

**"Out of memory"**
- Solution: Reduce batch size in config
- Use CPU instead of GPU
- Process fewer patients at once

**"Missing modality folders"**
- Solution: Ensure folder structure:
  ```
  patient/
  ├── Ultrasound/
  └── Mammogram/
  ```

### Getting Help

1. Check this guide thoroughly
2. Read code comments in main.py
3. Review training logs: `./ordcmvit_runs/<run>/logs/`
4. Check error messages carefully
5. Enable verbose mode: `--verbose` flag

---

## 📚 References

### Architecture
- Vision Transformer (ViT): Dosovitskiy et al., "An Image is Worth 16x16 Words"
- Ordinal Regression: Coral: Ordinal Regression (CORAL)
- Cross-Modal Fusion: Custom attention-based fusion

### Dataset
- BCMID: Breast Cancer Multimodal Image Database
- 323 patients, 5-fold cross-validation
- Modalities: Ultrasound + Mammogram

### Training Details
- Model: ViT-Small backbone
- Loss: CORAL + Auxiliary + Entropy regularization
- Optimizer: AdamW with differential learning rates
- Augmentation: Albumentations with modality-specific pipelines

---

## 📝 Citation

If you use OrdCMViT in your research, please cite:

```bibtex
@article{ordcmvit2024,
  title={OrdCMViT: Ordinal Cross-Modal Vision Transformer for Breast Cancer BI-RADS Grading},
  author={...},
  journal={...},
  year={2024},
  doi={...}
}
```

---

## 📄 Files Overview

```
OrdCMViT/
├── streamlit_app.py           # Web interface
├── batch_predict.py           # Batch processing
├── main.py                    # Training script
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── STREAMLIT_README.md        # Streamlit documentation
├── QUICKSTART.md             # Quick start guide
├── COMPLETE_GUIDE.md         # This file
└── src/
    ├── models/
    │   └── ordcmvit.py       # Model architecture
    ├── data/
    │   ├── bcmid.py          # Dataset loader
    │   └── transforms.py     # Data augmentation
    ├── engine/
    │   └── trainer.py        # Training engine
    └── utils/
        ├── metrics.py        # Evaluation metrics
        └── visualization.py  # CAM generation
```

---

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Train model (if needed): `python main.py`
3. ✅ Choose interface:
   - Web app: `streamlit run streamlit_app.py`
   - Batch: `python batch_predict.py --input_dir ./patients`
4. ✅ Upload/prepare patient data
5. ✅ Run predictions
6. ✅ Download/analyze results

---

## 📞 Support

For questions or issues:
- Review this guide
- Check code comments
- Enable verbose logging
- Review error messages
- Consult model training logs

Enjoy using OrdCMViT! 🏥🚀
