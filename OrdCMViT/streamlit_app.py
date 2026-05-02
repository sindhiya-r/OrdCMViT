"""
streamlit_app.py
────────────────
Interactive Streamlit application for OrdCMViT BIRADS prediction.

Features:
  • Upload patient folders (with Ultrasound and Mammogram subfolders)
  • Load trained model checkpoints
  • Predict BI-RADS classification
  • Display confidence scores
  • Evaluate against ground truth if available
  • Visualize activation maps (CAMs)
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.data.bcmid import get_image_paths, parse_birads
from src.models.ordcmvit import OrdCMViT, OrdinalHead
from src.utils.seed import set_seed, get_device
from src.utils.metrics import compute_metrics
from src.data.transforms import build_val_transform_us, build_val_transform_mm


# ─────────────────────────────────────────────────────────────────────────────
#  Config & Setup
# ─────────────────────────────────────────────────────────────────────────────

BIRADS_NAMES = {
    0: "BI-RADS 1 - Normal",
    1: "BI-RADS 2 - Benign",
    2: "BI-RADS 3 - Probably Benign",
    3: "BI-RADS 4 - Suspicious",
    4: "BI-RADS 5 - Malignant",
}

BIRADS_COLORS = {
    0: "#2ecc71",  # Green - Normal
    1: "#3498db",  # Blue - Benign
    2: "#f39c12",  # Orange - Probably Benign
    3: "#e67e22",  # Dark Orange - Suspicious
    4: "#e74c3c",  # Red - Malignant
}


@st.cache_resource
def get_device_cached():
    return get_device()


@st.cache_resource
def load_config():
    """Load configuration."""
    return Config()


@st.cache_data
def load_labels_map(labels_csv: str):
    """Load BCMID labels CSV into a dict: patient_id -> birads_int."""
    try:
        df = pd.read_csv(labels_csv, header=None)
    except Exception:
        return {}

    # Mirror parsing used in BCMIDDataset
    df.columns = ["patient_id", "birads", "binary_label"]
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["birads_int"] = df["birads"].apply(parse_birads)
    df = df.dropna(subset=["birads_int"]).reset_index(drop=True)
    df["birads_int"] = df["birads_int"].astype(int)
    return {row["patient_id"]: int(row["birads_int"]) for _, row in df.iterrows()}


# ─────────────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device):
    """Load OrdCMViT model from checkpoint."""
    cfg = Config()
    
    model = OrdCMViT(cfg)
    
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Check common checkpoint dict shapes produced by Trainer / torch.save
        state_dict = None
        if isinstance(checkpoint, dict):
            # Common keys used across the codebase: 'model_state', 'model', 'state_dict'
            for key in ("model_state", "model", "state_dict"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            # If none of the above, it may already be a raw state-dict wrapped in a dict
            if state_dict is None:
                # Heuristic: if many keys are not metadata keys, assume this is the state_dict
                possible_state_keys = [k for k in checkpoint.keys() if not isinstance(checkpoint[k], (int, float, str))]
                # fallback to using checkpoint itself only if it looks like a state-dict
                if any(k.startswith("fusion_cls") or k.startswith("us_backbone") for k in checkpoint.keys()):
                    state_dict = checkpoint
        else:
            state_dict = checkpoint

        if state_dict is None:
            st.error("Checkpoint format not recognised. Expected keys: 'model_state'|'model'|'state_dict' or a raw state-dict.")
            return None

        # Remove possible 'module.' prefix produced by DataParallel wrappers
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("module."):
                new_key = k[len("module."):]
            new_state[new_key] = v

        model.load_state_dict(new_state)
        model = model.to(device)
        model.eval()
        st.success(f"✅ Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Image Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load image from file (handles PNG, JPEG, DICOM)."""
    try:
        if str(path).lower().endswith(".dcm"):
            try:
                import pydicom
                ds = pydicom.dcmread(str(path))
                arr = ds.pixel_array.astype(np.float32)
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                return arr
            except ImportError:
                st.warning("pydicom not installed. Cannot load DICOM files.")
                return None
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.warning(f"Error loading {path}: {e}")
        return None


def get_latest_image(folder: Path) -> Optional[Tuple[np.ndarray, str]]:
    """Get latest/first image from folder."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
    images = sorted([
        p for p in folder.rglob("*") 
        if p.is_file() and p.suffix.lower() in valid_ext
    ])
    
    if not images:
        return None
    
    img = load_image_rgb(images[0])
    return (img, str(images[0])) if img is not None else None


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_patient(
    us_image: np.ndarray,
    mm_image: np.ndarray,
    model: OrdCMViT,
    device: torch.device,
    cfg: Config,
) -> Dict:
    """
    Predict BI-RADS for a single patient.
    
    Returns:
        {
            "pred_class": int (0-4),
            "pred_birads": int (1-5),
            "confidence": float,
            "probabilities": list of 5 floats,
            "ordinal_logits": list of 4 floats,
        }
    """
    transform_us = build_val_transform_us(cfg)
    transform_mm = build_val_transform_mm(cfg)
    
    # Transform images
    us_tensor = transform_us(image=us_image)["image"].unsqueeze(0).to(device)
    mm_tensor = transform_mm(image=mm_image)["image"].unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(us_tensor, mm_tensor)
        logits = output["main_logits"]  # [1, 4] ordinal logits
    
    # Predict class
    pred_class = OrdinalHead.predict(logits).item()
    pred_birads = pred_class + 1
    
    # Get probabilities
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Convert ordinal probabilities to class probabilities
    class_probs = np.zeros(5)
    class_probs[0] = 1 - probs[0]  # P(y=1) = 1 - P(y≥2)
    for k in range(1, 4):
        class_probs[k] = probs[k-1] - probs[k]
    class_probs[4] = probs[3]  # P(y=5) = P(y≥5)
    class_probs = np.clip(class_probs, 0, 1)
    
    confidence = float(np.max(class_probs))
    
    return {
        "pred_class": int(pred_class),
        "pred_birads": int(pred_birads),
        "confidence": confidence,
        "class_probabilities": class_probs.tolist(),
        "ordinal_logits": logits.squeeze().cpu().numpy().tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  UI Components
# ─────────────────────────────────────────────────────────────────────────────

def show_prediction_card(result: Dict, ground_truth: Optional[int] = None):
    """Display prediction result as a card."""
    pred_birads = result["pred_birads"]
    confidence = result["confidence"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted BI-RADS",
            f"{pred_birads}",
            help=BIRADS_NAMES[pred_birads - 1]
        )
    
    with col2:
        st.metric(
            "Confidence Score",
            f"{confidence:.2%}",
        )
    
    with col3:
        if ground_truth is not None:
            is_correct = (pred_birads - 1) == ground_truth
            st.metric(
                "Correctness",
                "✅ Correct" if is_correct else "❌ Incorrect",
                f"True: BI-RADS {ground_truth + 1}"
            )
    



def show_images(us_image: np.ndarray, mm_image: np.ndarray):
    """Display ultrasound and mammogram images side by side."""
    st.subheader("Input Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ultrasound (US)**")
        st.image(us_image, use_container_width=True)
    
    with col2:
        st.write("**Mammogram (MM)**")
        st.image(mm_image, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="BIRADS Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("Breast Cancer BI-RADS Prediction")
    st.markdown(
        "Upload patient ultrasound and mammogram images to predict BI-RADS classification."
    )
    
    # ───────────────────────────────────────────────────────────────────────
    #  Sidebar: Configuration
    # ───────────────────────────────────────────────────────────────────────
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model checkpoint selection
        runs_dir = Path("./runs")
        secondary_runs_dir = Path("./ordcmvit_runs")
        available_runs = []
        
        # Check primary runs directory
        if runs_dir.exists():
            for run_folder in sorted(runs_dir.iterdir()):
                if run_folder.is_dir():
                    # Check for best.pth in various locations
                    if (run_folder / "checkpoints" / "best.pth").exists():
                        available_runs.append((run_folder.name, run_folder / "checkpoints" / "best.pth"))
                    elif (run_folder / "ordcmvit" / "checkpoints" / "best.pth").exists():
                        available_runs.append((run_folder.name, run_folder / "ordcmvit" / "checkpoints" / "best.pth"))
        
        # Also check secondary runs directory
        if secondary_runs_dir.exists():
            for run_folder in sorted(secondary_runs_dir.iterdir()):
                if run_folder.is_dir():
                    if (run_folder / "ordcmvit" / "checkpoints" / "best.pth").exists():
                        if (run_folder.name, run_folder / "ordcmvit" / "checkpoints" / "best.pth") not in available_runs:
                            available_runs.append((run_folder.name, run_folder / "ordcmvit" / "checkpoints" / "best.pth"))
                    elif (run_folder / "checkpoints" / "best.pth").exists():
                        if (run_folder.name, run_folder / "checkpoints" / "best.pth") not in available_runs:
                            available_runs.append((run_folder.name, run_folder / "checkpoints" / "best.pth"))
        
        if not available_runs:
            st.error("❌ No trained models found in ./runs/ or ./ordcmvit_runs/")
            st.info("Please train a model first using: `python main.py --data_root ./BCMID --labels_csv ./BCMID/BCMID_labels.csv`")
            return
        
        # Extract just the run names for display
        run_names = [name for name, _ in available_runs]
        run_paths = {name: path for name, path in available_runs}
        
        selected_run = st.selectbox(
            "Select Model Run",
            run_names,
            help="Choose a trained model checkpoint",
        )
        
        checkpoint_path = str(run_paths[selected_run]) if selected_run else None
        
        st.success(f"✅ Selected: Run {selected_run}")
        
        # Load settings
        load_model_btn = st.button("🔄 Load Model", use_container_width=True)
    
    # ───────────────────────────────────────────────────────────────────────
    #  Main Content: Patient Input
    # ───────────────────────────────────────────────────────────────────────
    
    device = get_device_cached()
    
    # Load model (with caching)
    if load_model_btn or "model" not in st.session_state:
        st.session_state.model = load_model(checkpoint_path, device)
    
    if st.session_state.get("model") is None:
        st.error("❌ Failed to load model. Please check the checkpoint.")
        return
    
    # Load config
    cfg = load_config()
    
    # Load labels CSV into map (cached)
    patient_id = None
    cfg = load_config()
    labels_map = load_labels_map(cfg.data.labels_csv) if cfg and getattr(cfg, 'data', None) else {}
    
    # Image upload options
    st.subheader("📤 Upload Patient Images")
    
    upload_mode = st.radio(
        "Upload mode:",
        ["Individual Images", "Folder", "ZIP Folder"],
        horizontal=True
    )
    
    us_image = None
    mm_image = None
    
    if upload_mode == "Individual Images":
        col1, col2 = st.columns(2)
        
        with col1:
            us_file = st.file_uploader(
                "Ultrasound Image",
                type=["png", "jpg", "jpeg", "bmp", "tiff", "dcm"],
                key="us_upload"
            )
            if us_file:
                us_image = np.array(Image.open(us_file).convert("RGB"))
        
        with col2:
            mm_file = st.file_uploader(
                "Mammogram Image",
                type=["png", "jpg", "jpeg", "bmp", "tiff", "dcm"],
                key="mm_upload"
            )
            if mm_file:
                mm_image = np.array(Image.open(mm_file).convert("RGB"))
    
    elif upload_mode == "Folder":
        uploaded_files = st.file_uploader(
            "Upload a folder by selecting multiple files (or drag the folder contents)",
            accept_multiple_files=True,
        )

        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # Save all uploaded files; preserve any subpath info found in the filename
                for uf in uploaded_files:
                    # Some browsers include relative paths in the filename (e.g. 'Ultrasound/img.png')
                    safe_name = uf.name.replace('\\', '/')
                    dest = tmpdir_path / safe_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(uf.getbuffer())

                # Find Ultrasound and Mammogram folders / files
                us_folder = None
                mm_folder = None
                us_files = []
                mm_files = []
                
                # First pass: look for named folders
                for folder in tmpdir_path.rglob("*"):
                    if folder.is_dir():
                        name_lower = folder.name.lower()
                        if "ultrasound" in name_lower or name_lower == "us":
                            us_folder = folder
                        elif "mammogram" in name_lower or name_lower == "mm":
                            mm_folder = folder

                # Second pass: collect image files and classify by name pattern
                valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
                for file_path in sorted(tmpdir_path.rglob("*")):
                    if file_path.is_file() and file_path.suffix.lower() in valid_ext:
                        name_lower = file_path.name.lower()
                        # Classify by filename pattern
                        if any(x in name_lower for x in ["us_", "_us", "ultrasound"]):
                            us_files.append(file_path)
                        elif any(x in name_lower for x in ["mm_", "_mm", "mammo", "mammogram"]):
                            mm_files.append(file_path)

                # Load from folders if found, else fall back to file patterns
                if us_folder:
                    result = get_latest_image(us_folder)
                    if result:
                        us_image, us_path = result
                        st.info(f"✅ Loaded ultrasound from folder: {us_path}")
                elif us_files:
                    us_image = load_image_rgb(us_files[0])
                    if us_image is not None:
                        st.info(f"✅ Loaded ultrasound (auto-detected): {us_files[0].name}")

                if mm_folder:
                    result = get_latest_image(mm_folder)
                    if result:
                        mm_image, mm_path = result
                        st.info(f"✅ Loaded mammogram from folder: {mm_path}")
                elif mm_files:
                    mm_image = load_image_rgb(mm_files[0])
                    if mm_image is not None:
                        st.info(f"✅ Loaded mammogram (auto-detected): {mm_files[0].name}")
                
                # Show what was detected
                if us_image is None:
                    st.warning("⚠️ No ultrasound (US) images detected. Ensure files contain 'us_', '_us', or 'ultrasound' in the filename, or upload from an 'Ultrasound' folder.")
                if mm_image is None:
                    st.warning("⚠️ No mammogram (MM) images detected. Ensure files contain 'mm_', '_mm', or 'mammo' in the filename, or upload from a 'Mammogram' folder.")

    else:  # ZIP Folder
        zip_file = st.file_uploader(
            "Upload ZIP containing Ultrasound/ and Mammogram/ folders",
            type=["zip"],
        )

        if zip_file:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract ZIP
                zip_path = Path(tmpdir) / "upload.zip"
                with open(zip_path, "wb") as f:
                    f.write(zip_file.getbuffer())

                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)

                # Find Ultrasound and Mammogram folders
                tmpdir_path = Path(tmpdir)
                us_folder = None
                mm_folder = None

                for folder in tmpdir_path.rglob("*"):
                    if folder.is_dir():
                        name_lower = folder.name.lower()
                        if "ultrasound" in name_lower or "us" in name_lower:
                            us_folder = folder
                        elif "mammogram" in name_lower or "mm" in name_lower:
                            mm_folder = folder

                if us_folder:
                    result = get_latest_image(us_folder)
                    if result:
                        us_image, us_path = result
                        st.info(f"✅ Loaded ultrasound: {us_path}")

                if mm_folder:
                    result = get_latest_image(mm_folder)
                    if result:
                        mm_image, mm_path = result
                        st.info(f"✅ Loaded mammogram: {mm_path}")
    
    # ───────────────────────────────────────────────────────────────────────
    #  Prediction
    # ───────────────────────────────────────────────────────────────────────
    
    if us_image is not None and mm_image is not None:
        # Show images
        show_images(us_image, mm_image)
        
        # Optional: ground truth input
        st.subheader("Ground Truth (Optional)")
        col1, col2 = st.columns([1, 2])
        # If patient_id matches CSV, pre-check and show the label
        matched_label = None
        if patient_id:
            pid = str(patient_id).strip()
            matched_label = labels_map.get(pid)

        with col1:
            has_ground_truth = st.checkbox(
                "I have ground truth BI-RADS",
                value=(matched_label is not None)
            )

        ground_truth = None
        if has_ground_truth:
            with col2:
                # If we have a matched label, pre-select it in the dropdown
                if matched_label is not None:
                    default_index = matched_label - 1
                else:
                    default_index = 0

                ground_truth = st.selectbox(
                    "Ground Truth BI-RADS",
                    list(range(1, 6)),
                    index=default_index,
                    format_func=lambda x: BIRADS_NAMES[x - 1]
                ) - 1

                if matched_label is not None:
                    st.info(f"Ground truth loaded from CSV: BI-RADS {matched_label}")
        
        # Run prediction
        if st.button("🔮 Predict BI-RADS", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                result = predict_patient(
                    us_image, mm_image,
                    st.session_state.model, device, cfg
                )
            
            st.session_state.last_result = result
            st.session_state.last_ground_truth = ground_truth
            st.session_state.last_patient_id = patient_id
        
        # Display results
        if "last_result" in st.session_state:
            st.divider()
            st.subheader("📊 Prediction Results")
            
            result = st.session_state.last_result
            ground_truth = st.session_state.last_ground_truth
            
            show_prediction_card(result, ground_truth)
            
            # Save results
            if st.button("💾 Save Results to CSV"):
                results_df = pd.DataFrame({
                    "patient_id": [st.session_state.last_patient_id or "Unknown"],
                    "pred_birads": [result["pred_birads"]],
                    "true_birads": [ground_truth + 1 if ground_truth is not None else None],
                    "confidence": [result["confidence"]],
                    "correct": [
                        (result["pred_birads"] - 1) == ground_truth
                        if ground_truth is not None else None
                    ],
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"prediction_{st.session_state.last_patient_id or 'patient'}.csv",
                    mime="text/csv",
                )
    
    else:
        if upload_mode == "Individual Images":
            st.info("⬆️ Upload both ultrasound and mammogram images to proceed.")
        elif upload_mode == "Folder":
            st.info("⬆️ Upload image files from both Ultrasound and Mammogram folders. You can either:\n"
                   "- Select files from an 'Ultrasound' folder AND a 'Mammogram' folder, or\n"
                   "- Use filenames containing 'us_' (ultrasound) and 'mm_' (mammogram) for auto-detection.")
        else:
            st.info("⬆️ Upload a ZIP file containing Ultrasound/ and Mammogram/ folders.")


if __name__ == "__main__":
    # Initialize session state
    if "model" not in st.session_state:
        st.session_state.model = None
    
    main()
