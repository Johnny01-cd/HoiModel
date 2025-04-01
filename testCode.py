import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO # Still needed for InteractionDataset class definition if not using features
import gc
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration (Should match Fullcode.py settings used for training/saving) ---
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data and Model Parameters
NUM_TIMESTEPS = 20
BATCH_SIZE = 4 # Can be adjusted for testing, doesn't need to match training
TEST_FEATURE_DIR = 'features_mp_test' # Directory where test features were saved
MODEL_SAVE_PATH = "interaction_model_mediapipe.pth" # Path to the saved model

# Define interaction directories (used to find test data and get class names)
interaction_dirs_config = [
    (0, '04_sword_part1', 'Sword'),      # Label 0, dir, Class Name
    (1, '001_hugging', 'Hugging'),         # Label 1, dir, Class Name
    (2, '02_grappling', 'Grappling'),        # Label 2, dir, Class Name
    (3, '07_ballroom', 'Ballroom'),         # Label 3, dir, Class Name
    (4, '12_mma', 'MMA'),              # Label 4, dir, Class Name
]
# Extract class names in the correct order based on labels
class_names = [name for _, _, name in sorted(interaction_dirs_config, key=lambda x: x[0])]
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
# --- END Configuration ---


# --- Helper Function (Copied from Fullcode.py) ---
def compute_folder_name(folder_path):
    """Computes a unique name for a cam folder based on its path components."""
    # Handle potential extra nesting if 'test' dir is involved
    path_parts = folder_path.strip(os.sep).split(os.sep)
    if len(path_parts) >= 3:
         # Assumes structure like .../interaction_type/sequence_name/camXX
        sequence_dir = path_parts[-3] # e.g., '04_sword_part1'
        parent_name = path_parts[-2] # e.g., 'GOPR0084' or the sequence name again if no middle dir
        cam_name = path_parts[-1] # e.g., 'cam01'
    elif len(path_parts) == 2:
        # Assumes structure like interaction_type/camXX
        sequence_dir = "unknown_sequence" # Or handle differently if needed
        parent_name = path_parts[-2]
        cam_name = path_parts[-1]
    else: # Failsafe
        sequence_dir = "unknown_sequence"
        parent_name = "unknown_parent"
        cam_name = os.path.basename(folder_path)

    # Check if the first part is 'test' and adjust if necessary
    if path_parts[0].lower() == 'test' and len(path_parts) > 3:
         sequence_dir = path_parts[-3]
         parent_name = path_parts[-2]
         cam_name = path_parts[-1]

    return f"{sequence_dir}_{parent_name}_{cam_name}"


# --- Dataset Class (Copied from Fullcode.py, simplified for loading features) ---
class InteractionDataset(Dataset):
    def __init__(self, folder_paths, labels, num_timesteps=20, feature_dir=None):
        """Initialize the dataset with folder paths, labels, and feature directory."""
        print("Initializing InteractionDataset for testing...")
        self.folder_paths = folder_paths
        self.labels = labels
        self.num_timesteps = num_timesteps
        self.feature_dir = feature_dir

        if feature_dir is None:
            # This part should ideally NOT be needed if feature_dir is correctly provided
            # but included for robustness or if testing without precomputed features
            print("WARNING: feature_dir is None. Initializing models for on-the-fly extraction.")
            # Initialize YOLOv11 pose estimation model
            self.pose_estimator = YOLO('yolo11l-pose.pt')
            print("YOLOv11 Pose model initialized.")
            # ResNet initialization
            self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1]).to(device)
            self.resnet.eval()
            print("ResNet model initialization finished.")
            # Image preprocessing pipeline
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # No need to initialize models if loading features
             self.pose_estimator = None
             self.resnet = None
             self.preprocess = None # Preprocessing was done during feature extraction

        print(f"Dataset initialized. Mode: {'Loading features' if feature_dir else 'Extracting features'}")

    def __len__(self):
        return len(self.folder_paths)

    # --- Methods for frame extraction and feature calculation ---
    # --- (get_frame_sources, extract_frames, extract_features) ---
    # --- These are NOT needed if feature_dir is provided and valid ---
    # --- They are kept here in case feature_dir=None, but removed ---
    # --- from the main testing logic path for clarity if features exist. ---

    # --- You *could* remove the methods below if you *guarantee* ---
    # --- feature_dir is always provided for testing. ---

    def get_frame_sources(self, folder_path):
        """Determine if the folder contains a video or images."""
        # ... (implementation identical to Fullcode.py) ...
        # print(f"Checking for media in {folder_path}...") # Optional verbosity
        video_extensions = ['.mp4']
        image_extensions = ['.jpg', '.jpeg', '.png'] # Added png
        video_paths, image_paths = [], []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions: video_paths.append(file_path)
                elif ext in image_extensions: image_paths.append(file_path)
        if video_paths: return 'video', video_paths[0]
        elif image_paths: return 'images', sorted(image_paths)
        else: raise ValueError(f"No media found in {folder_path}")

    def extract_frames(self, folder_path):
        """Extract a fixed number of frames from video or images."""
        # ... (implementation identical to Fullcode.py) ...
        # print(f"Extracting frames from {folder_path}...") # Optional verbosity
        source_type, source = self.get_frame_sources(folder_path)
        selected_frames = []
        if source_type == 'video':
            cap = cv2.VideoCapture(source)
            if not cap.isOpened(): raise ValueError(f"Could not open video: {source}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0: raise ValueError(f"No frames in video: {source}")
            indices = np.linspace(0, total_frames - 1, self.num_timesteps, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: # Try reading next available frame if specific fails
                     ret, frame = cap.read()
                if ret: selected_frames.append(frame.copy())
                elif selected_frames: selected_frames.append(selected_frames[-1].copy()) # Repeat last frame if read fails
                else: raise ValueError(f"Failed to read frame {idx} from video {source}")
            cap.release()
        elif source_type == 'images':
            total_images = len(source)
            if total_images == 0: raise ValueError(f"No images in {folder_path}")
            indices = np.linspace(0, total_images - 1, self.num_timesteps, dtype=int)
            for idx in indices:
                img_path = source[idx]
                frame = cv2.imread(img_path)
                if frame is not None: selected_frames.append(frame)
                elif selected_frames: selected_frames.append(selected_frames[-1].copy()) # Repeat last frame
                else: print(f"Warning: Failed to load image {img_path} and no previous frame exists.") # Or raise error
            if not selected_frames: raise ValueError(f"No valid images loaded from {folder_path}")
        else: raise ValueError(f"Unknown source type: {source_type}")
        # Ensure exactly num_timesteps frames
        if len(selected_frames) < self.num_timesteps:
             last_frame = selected_frames[-1]
             selected_frames.extend([last_frame.copy()] * (self.num_timesteps - len(selected_frames)))
        elif len(selected_frames) > self.num_timesteps:
            selected_frames = selected_frames[:self.num_timesteps]
        # print(f"Extracted {len(selected_frames)} frames.") # Optional verbosity
        return selected_frames

    def extract_features(self, frames):
         """Extracts features (only runs if feature_dir=None)."""
         # ... (implementation identical to Fullcode.py) ...
         # This is computationally expensive and should be avoided if features are precomputed
         print("Extracting features on-the-fly using YOLOv11...")
         frames_tensor = torch.stack([self.preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]).to(device)
         with torch.no_grad():
             feature_maps = self.resnet(frames_tensor)
             full_body_features = feature_maps.view(self.num_timesteps, -1)
         print("ResNet full-body feature extraction finished.")
         key_parts_original_indices = [0, 4, 7, 11, 14]
         original_to_yolo_map = {0: 0, 4: 10, 7: 9, 11: 16, 14: 15}
         yolo_indices = [original_to_yolo_map[idx] for idx in key_parts_original_indices]
         num_key_parts = len(key_parts_original_indices)
         positions = np.full((self.num_timesteps, 2, num_key_parts, 2), np.nan)
         confidences = np.zeros((self.num_timesteps, 2, num_key_parts))
         local_patch_features_list = []
         for t, frame in enumerate(frames):
             h, w = frame.shape[:2]
             results = self.pose_estimator(frame)
             persons_keypoints = []
             if results[0].boxes is not None:
                 boxes = results[0].boxes; areas = [(box.xywh[0][2] * box.xywh[0][3]).item() for box in boxes]
                 sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
                 for idx in sorted_indices[:2]:
                     kp = results[0].keypoints.xy[idx].cpu().numpy()
                     conf = results[0].keypoints.conf[idx].cpu().numpy()
                     persons_keypoints.append((kp, conf))
             while len(persons_keypoints) < 2: persons_keypoints.append((np.zeros((17, 2)), np.zeros(17)))
             frame_patch_features_p1, frame_patch_features_p2 = [], []
             for p, (kp, conf) in enumerate(persons_keypoints[:2]):
                 kp_selected, conf_selected = kp[yolo_indices], conf[yolo_indices]
                 target_list = frame_patch_features_p1 if p == 0 else frame_patch_features_p2
                 for k in range(num_key_parts):
                     if conf_selected[k] > 0.1:
                         cx, cy = int(kp_selected[k, 0]), int(kp_selected[k, 1])
                         positions[t, p, k, :] = cx, cy; confidences[t, p, k] = conf_selected[k]
                         patch_size = 64; x1,y1=max(0,cx-patch_size//2),max(0,cy-patch_size//2); x2,y2=min(w,cx+patch_size//2),min(h,cy+patch_size//2)
                         patch = frame[y1:y2, x1:x2]
                         if patch.size == 0: patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8) # Handle empty patch
                         elif patch.shape[0] < patch_size or patch.shape[1] < patch_size: patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR) # Resize if needed
                         patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                         patch_tensor = self.preprocess(patch_rgb).to(device).unsqueeze(0)
                         with torch.no_grad(): feature = self.resnet(patch_tensor).view(-1)
                         target_list.append(feature)
                     else: target_list.append(torch.zeros(2048, device=device))
             local_patch_features_list.append(torch.stack([torch.stack(frame_patch_features_p1), torch.stack(frame_patch_features_p2)]))
         local_patch_features = torch.stack(local_patch_features_list).to(device)
         print("YOLOv11 keypoint extraction and ResNet local patch feature extraction finished.")
         motion_posture_features = np.zeros((self.num_timesteps, 2, num_key_parts, 4))
         for t in range(self.num_timesteps):
             frame_center_x, frame_center_y = frames[t].shape[1]/2, frames[t].shape[0]/2
             for p in range(2):
                 for k in range(num_key_parts):
                     dx = dy = 0.0
                     if t > 0 and not np.isnan(positions[t,p,k,0]) and not np.isnan(positions[t-1,p,k,0]):
                         dx = positions[t,p,k,0] - positions[t-1,p,k,0]; dy = positions[t,p,k,1] - positions[t-1,p,k,1]
                     confidence = confidences[t,p,k]
                     dist_center = 0.0 if np.isnan(positions[t,p,k,0]) else np.linalg.norm(positions[t,p,k] - np.array([frame_center_x, frame_center_y]))
                     motion_posture_features[t,p,k] = [dx, dy, confidence, dist_center]
         motion_posture_features = torch.tensor(motion_posture_features, dtype=torch.float32).to(device)
         print("Motion feature extraction finished.")
         return full_body_features, local_patch_features, motion_posture_features


    # --- Get Item ---
    def __getitem__(self, idx):
        # print(f"Getting test item {idx}...") # Optional verbosity
        folder_path = self.folder_paths[idx]
        label = self.labels[idx]

        if self.feature_dir:
            # --- Load precomputed features ---
            folder_name = compute_folder_name(folder_path)
            json_path = os.path.join(self.feature_dir, f"{folder_name}.json")
            # print(f"Attempting to load features from: {json_path}") # Debug print
            try:
                with open(json_path, 'r') as f:
                    feature_dict = json.load(f)

                # Construct absolute paths if necessary (safer)
                base_feature_dir = os.path.dirname(json_path)
                full_body_path = os.path.join(base_feature_dir, os.path.basename(feature_dict["full_body_features"]))
                local_patch_path = os.path.join(base_feature_dir, os.path.basename(feature_dict["local_patch_features"]))
                motion_posture_path = os.path.join(base_feature_dir, os.path.basename(feature_dict["motion_posture_features"]))

                # Load numpy arrays and convert to tensors
                full_body = torch.from_numpy(np.load(full_body_path)).to(torch.float32) # Ensure float32
                local_patch = torch.from_numpy(np.load(local_patch_path)).to(torch.float32)
                motion_posture = torch.from_numpy(np.load(motion_posture_path)).to(torch.float32)
                # print(f"Loaded precomputed features for item {idx}") # Optional verbosity

            except FileNotFoundError:
                print(f"FATAL ERROR: Precomputed feature file not found: {json_path}")
                print(f"Attempted folder name: {folder_name}")
                print(f"Original folder path: {folder_path}")
                # Attempt to list files in the feature directory for debugging
                try:
                    print(f"Files in {self.feature_dir}: {os.listdir(self.feature_dir)}")
                except Exception as list_e:
                    print(f"Could not list files in feature directory: {list_e}")
                raise # Re-raise the error to stop execution
            except Exception as e:
                print(f"Error loading features for item {idx} from {json_path}: {e}")
                raise # Re-raise the error

        else:
            # --- Extract features on-the-fly (Fallback) ---
            print(f"Warning: Extracting features on-the-fly for test item {idx} ({folder_path}). This is slow.")
            if self.pose_estimator is None or self.resnet is None:
                 raise RuntimeError("Models not initialized for on-the-fly feature extraction in dataset.")
            frames = self.extract_frames(folder_path)
            if not frames:
                raise ValueError(f"No frames obtained for item {idx}, folder {folder_path}")
            full_body, local_patch, motion_posture = self.extract_features(frames)

        # Ensure tensors are on the correct device before returning (optional, can be done in loop)
        # return full_body.to(device), local_patch.to(device), motion_posture.to(device), torch.tensor(label, dtype=torch.long).to(device)
        # Return tensors as they are loaded/created; device transfer happens in eval loop
        return full_body, local_patch, motion_posture, torch.tensor(label, dtype=torch.long)


# --- Model Definition (Copied from Fullcode.py) ---
class InteractionRecognitionModel(nn.Module):
    def __init__(self, num_classes, num_timesteps=20, num_key_parts=5, resnet_dim=2048, motion_dim=4):
        super(InteractionRecognitionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_key_parts = num_key_parts
        self.resnet_dim = resnet_dim
        self.motion_dim = motion_dim # Dimension of motion features per keypoint [dx, dy, conf, dist_center]

        # LSTMs for each feature stream
        self.lstm_full_body = nn.LSTM(resnet_dim, 256, batch_first=True) # Input: (B, T, resnet_dim)

        # Local patch features: (B, T, P, K, resnet_dim) -> flatten -> (B, T, P*K*resnet_dim)
        # P=2 persons, K=5 keyparts
        local_input_dim = 2 * num_key_parts * resnet_dim
        self.lstm_local_patch = nn.LSTM(local_input_dim, 512, batch_first=True)

        # Motion/posture features: (B, T, P, K, motion_dim) -> flatten -> (B, T, P*K*motion_dim)
        motion_input_dim = 2 * num_key_parts * motion_dim
        self.lstm_motion = nn.LSTM(motion_input_dim, 128, batch_first=True)

        # Combined features from the last timestep of each LSTM
        combined_dim = 256 + 512 + 128 # Output dims of LSTMs
        self.fc = nn.Linear(combined_dim, num_classes)

    def forward(self, full_body, local_patch, motion_posture):
        # Process full body features
        # Input shape: (B, T, resnet_dim)
        out_fb, _ = self.lstm_full_body(full_body)
        fb_features = out_fb[:, -1, :] # Get output of last time step (B, 256)

        # Process local patch features
        # Input shape: (B, T, P, K, RF=resnet_dim)
        B, T, P, K, RF = local_patch.shape
        local_patch_flat = local_patch.view(B, T, -1) # Flatten P, K, RF -> (B, T, P*K*RF)
        out_lp, _ = self.lstm_local_patch(local_patch_flat)
        lp_features = out_lp[:, -1, :] # Get output of last time step (B, 512)

        # Process motion and posture features
        # Input shape: (B, T, P, K, MF=motion_dim)
        B, T, P, K, MF = motion_posture.shape
        motion_posture_flat = motion_posture.view(B, T, -1) # Flatten P, K, MF -> (B, T, P*K*MF)
        out_mp, _ = self.lstm_motion(motion_posture_flat)
        mp_features = out_mp[:, -1, :] # Get output of last time step (B, 128)

        # Concatenate features from all streams
        combined = torch.cat((fb_features, lp_features, mp_features), dim=1) # (B, 256 + 512 + 128)

        # Final classification layer
        output = self.fc(combined) # (B, num_classes)
        return output

# --- Main Testing Logic ---
if __name__ == "__main__":
    # 1. Find Test Data
    test_folder_paths = []
    test_labels = []
    print("\nSearching for test data folders...")
    for label, top_dir_base, _ in interaction_dirs_config:
        test_top_dir = os.path.join('test', top_dir_base) # Assumes test data is in a 'test' subdirectory
        if not os.path.isdir(test_top_dir):
            print(f"Warning: Test directory not found: {test_top_dir}. Skipping.")
            continue
        print(f"Searching in {test_top_dir} for label {label}...")
        for root, dirs, files in os.walk(test_top_dir):
            # Find directories starting with 'cam'
            for d in dirs:
                if d.startswith('cam') and os.path.isdir(os.path.join(root, d)):
                    cam_path = os.path.join(root, d)
                    test_folder_paths.append(cam_path)
                    test_labels.append(label)
                    print(f"  Found: {cam_path}")

    if not test_folder_paths:
        print("\nError: No 'camXX' folders found in the specified test directories.")
        print("Please ensure the 'test' directory exists and contains subdirectories matching 'interaction_dirs_config'.")
        exit()

    print(f"\nTotal test samples found: {len(test_folder_paths)}")
    from collections import Counter
    test_label_counts = Counter(test_labels)
    print(f"Test label distribution: {test_label_counts}")

    # 2. Create Dataset and DataLoader
    print(f"\n--- Loading Test Dataset (using features from '{TEST_FEATURE_DIR}') ---")
    if not os.path.isdir(TEST_FEATURE_DIR):
         print(f"ERROR: Test feature directory not found: {TEST_FEATURE_DIR}")
         print("Please ensure features were precomputed correctly or set PRECOMPUTE=False and run Fullcode.py again.")
         exit()

    try:
        # Use feature_dir for loading precomputed features
        test_dataset = InteractionDataset(test_folder_paths, test_labels,
                                        num_timesteps=NUM_TIMESTEPS,
                                        feature_dir=TEST_FEATURE_DIR)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for testing
    except Exception as e:
        print(f"Error initializing test dataset or dataloader: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 3. Load Model
    print("\n--- Initializing and Loading Model ---")
    model = InteractionRecognitionModel(num_classes=num_classes, num_timesteps=NUM_TIMESTEPS).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Model loaded successfully from {MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_SAVE_PATH}")
        print("Please ensure the model was trained and saved correctly.")
        exit()
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit()

    model.eval() # Set model to evaluation mode (important!)

    # 4. Perform Evaluation
    print("\n--- Starting Evaluation ---")
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for i, (full_body, local_patch, motion_posture, labels) in enumerate(test_dataloader):
            # Move data to the configured device
            full_body = full_body.to(device)
            local_patch = local_patch.to(device)
            motion_posture = motion_posture.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs = model(full_body, local_patch, motion_posture)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability

            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                 print(f'Processed batch [{i+1}/{len(test_dataloader)}]')

    print("\n--- Evaluation Finished ---")

    # 5. Calculate and Display Metrics
    print("\n--- Evaluation Results ---")

    # Ensure labels are numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

    # Classification Report (Precision, Recall, F1-Score per class)
    print("\nClassification Report:")
    # Ensure target_names corresponds correctly to the labels 0, 1, 2...
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    print(report)

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # 6. Visualize Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Interaction Recognition')
    # Save the plot
    plot_filename = "confusion_matrix_test.png"
    plt.savefig(plot_filename)
    print(f"\nConfusion matrix plot saved to {plot_filename}")
    # Display the plot
    plt.show()

    print("\n--- Testing Script Finished ---")