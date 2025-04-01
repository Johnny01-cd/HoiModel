import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
import gc

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class
class InteractionDataset(Dataset):
    def __init__(self, folder_paths, labels, num_timesteps=20, feature_dir=None):
        """Initialize the dataset with folder paths, labels, and optional precomputed feature directory."""
        print("Initializing InteractionDataset...")
        self.folder_paths = folder_paths
        self.labels = labels
        self.num_timesteps = num_timesteps
        self.feature_dir = feature_dir

        if feature_dir is None:
            # Initialize YOLOv11 pose estimation model
            self.pose_estimator = YOLO('yolo11l-pose.pt')  # Lightweight model
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
        print("Dataset initialized.")

    def __len__(self):
        return len(self.folder_paths)

    def get_frame_sources(self, folder_path):
        """
        Determine if the folder contains a video or images.

        Args:
            folder_path (str): Path to the camXX folder.

        Returns:
            tuple: ('video' or 'images', source path or list of paths).
        """
        print(f"Checking for media in {folder_path}...")
        video_extensions = ['.mp4']
        image_extensions = ['.jpg', '.jpeg']

        video_paths = []
        image_paths = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    video_paths.append(file_path)
                elif ext in image_extensions:
                    image_paths.append(file_path)

        if video_paths:
            print(f"Found video: {video_paths[0]}")
            return 'video', video_paths[0]
        elif image_paths:
            image_paths.sort()
            print(f"Found {len(image_paths)} images.")
            return 'images', image_paths
        else:
            raise ValueError(f"No media found in {folder_path}")

    def extract_frames(self, folder_path):
        """
        Extract a fixed number of frames from video or images.

        Args:
            folder_path (str): Path to the camXX folder.

        Returns:
            list: List of sampled frames.
        """
        print(f"Extracting frames from {folder_path}...")
        source_type, source = self.get_frame_sources(folder_path)

        if source_type == 'video':
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {source}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"No frames in video: {source}")
            if total_frames >= self.num_timesteps:
                step = total_frames / self.num_timesteps
                indices = [int(i * step) for i in range(self.num_timesteps)]
            else:
                indices = list(range(total_frames)) + [total_frames - 1] * (self.num_timesteps - total_frames)
            selected_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Failed to read frame {idx} directly from {source}. Trying next.")
                    ret, frame = cap.read()
                    if not ret:
                        if selected_frames:
                            print(f"Warning: Using last valid frame for index {idx}")
                            frame = selected_frames[-1]
                        else:
                            raise ValueError(f"Failed to read frame {idx} (and subsequent) from {source}")
                    else:
                        print(f"Successfully read subsequent frame for index {idx}")
                selected_frames.append(frame.copy())
            cap.release()
        elif source_type == 'images':
            total_images = len(source)
            if total_images == 0:
                raise ValueError(f"No images in {folder_path}")
            if total_images >= self.num_timesteps:
                step = total_images / self.num_timesteps
                indices = [int(i * step) for i in range(self.num_timesteps)]
            else:
                indices = list(range(total_images)) + [total_images - 1] * (self.num_timesteps - total_images)
            selected_frames = []
            for idx in indices:
                img_path = source[idx]
                frame = cv2.imread(img_path)
                if frame is not None:
                    selected_frames.append(frame)
                else:
                    print(f"Warning: Failed to load image {img_path}")
            if len(selected_frames) < self.num_timesteps:
                if selected_frames:
                    last_frame = selected_frames[-1]
                    while len(selected_frames) < self.num_timesteps:
                        selected_frames.append(last_frame.copy())
                else:
                    raise ValueError(f"No valid images loaded from {folder_path}")
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        if len(selected_frames) > self.num_timesteps:
            selected_frames = selected_frames[:self.num_timesteps]
        elif len(selected_frames) < self.num_timesteps:
            if not selected_frames:
                raise ValueError(f"No frames extracted for {folder_path}")
            last_frame = selected_frames[-1]
            while len(selected_frames) < self.num_timesteps:
                selected_frames.append(last_frame.copy())

        print(f"Extracted {len(selected_frames)} frames.")
        return selected_frames

    def __getitem__(self, idx):
        print(f"Getting item {idx} from dataset...")
        folder_path = self.folder_paths[idx]
        label = self.labels[idx]

        if self.feature_dir:
            folder_name = compute_folder_name(folder_path)
            json_path = os.path.join(self.feature_dir, f"{folder_name}.json")
            try:
                with open(json_path, 'r') as f:
                    feature_dict = json.load(f)
                full_body = torch.from_numpy(np.load(feature_dict["full_body_features"])).to(device)
                local_patch = torch.from_numpy(np.load(feature_dict["local_patch_features"])).to(device)
                motion_posture = torch.from_numpy(np.load(feature_dict["motion_posture_features"])).to(device)
                print(f"Loaded precomputed features for item {idx} from {json_path}")
            except FileNotFoundError:
                print(f"Error: Precomputed feature file not found: {json_path}")
                raise
            except Exception as e:
                print(f"Error loading features for item {idx} from {json_path}: {e}")
                raise
        else:
            frames = self.extract_frames(folder_path)
            if not frames:
                raise ValueError(f"No frames obtained for item {idx}, folder {folder_path}")
            full_body, local_patch, motion_posture = self.extract_features(frames)

        print(f"Item {idx} retrieved.")
        return full_body.to(device), local_patch.to(device), motion_posture.to(device), torch.tensor(label, dtype=torch.long).to(device)

    def extract_features(self, frames):
        print("Extracting features from frames using YOLOv11...")

        # Full-body features
        frames_tensor = torch.stack([self.preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]).to(device)
        with torch.no_grad():
            feature_maps = self.resnet(frames_tensor)
            full_body_features = feature_maps.view(self.num_timesteps, -1)
        print("ResNet full-body feature extraction finished.")

        # Key parts definition for YOLOv11
        key_parts_original_indices = [0, 4, 7, 11, 14]  # nose, RWrist, LWrist, RAnkle, LAnkle
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
                boxes = results[0].boxes
                areas = [box.xywh[0][2] * box.xywh[0][3] for box in boxes]
                sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
                selected_indices = sorted_indices[:2]
                for idx in selected_indices:
                    kp = results[0].keypoints.xy[idx].cpu().numpy()
                    conf = results[0].keypoints.conf[idx].cpu().numpy()
                    persons_keypoints.append((kp, conf))

            while len(persons_keypoints) < 2:
                persons_keypoints.append((np.zeros((17, 2)), np.zeros(17)))

            frame_patch_features_p1 = []
            frame_patch_features_p2 = []
            for p, (kp, conf) in enumerate(persons_keypoints[:2]):
                kp_selected = kp[yolo_indices]
                conf_selected = conf[yolo_indices]
                target_list = frame_patch_features_p1 if p == 0 else frame_patch_features_p2

                for k in range(num_key_parts):
                    if conf_selected[k] > 0.1:
                        cx, cy = int(kp_selected[k, 0]), int(kp_selected[k, 1])
                        positions[t, p, k, 0] = cx
                        positions[t, p, k, 1] = cy
                        confidences[t, p, k] = conf_selected[k]

                        patch_size = 64
                        x1, y1 = max(0, cx - patch_size // 2), max(0, cy - patch_size // 2)
                        x2, y2 = min(w, cx + patch_size // 2), min(h, cy + patch_size // 2)
                        patch = frame[y1:y2, x1:x2]
                        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                            h_patch, w_patch = patch.shape[:2]
                            top = max(0, (patch_size - h_patch) // 2)
                            bottom = max(0, patch_size - h_patch - top)
                            left = max(0, (patch_size - w_patch) // 2)
                            right = max(0, patch_size - w_patch - left)
                            patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                            patch = cv2.resize(patch, (patch_size, patch_size))
                        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                        patch_tensor = self.preprocess(patch_rgb).to(device).unsqueeze(0)
                        with torch.no_grad():
                            feature = self.resnet(patch_tensor).view(-1)
                        target_list.append(feature)
                    else:
                        target_list.append(torch.zeros(2048, device=device))

            stacked_p1 = torch.stack(frame_patch_features_p1)
            stacked_p2 = torch.stack(frame_patch_features_p2)
            local_patch_features_list.append(torch.stack([stacked_p1, stacked_p2]))

        local_patch_features = torch.stack(local_patch_features_list).to(device)
        print("YOLOv11 keypoint extraction and ResNet local patch feature extraction finished.")

        # Motion and posture features
        motion_posture_features = np.zeros((self.num_timesteps, 2, num_key_parts, 4))
        for t in range(self.num_timesteps):
            frame_center_x = frames[t].shape[1] / 2
            frame_center_y = frames[t].shape[0] / 2
            for p in range(2):
                for k in range(num_key_parts):
                    if t == 0 or np.isnan(positions[t, p, k, 0]) or np.isnan(positions[t-1, p, k, 0]):
                        dx = dy = 0.0
                    else:
                        dx = positions[t, p, k, 0] - positions[t-1, p, k, 0]
                        dy = positions[t, p, k, 1] - positions[t-1, p, k, 1]
                    confidence = confidences[t, p, k]
                    dist_center = 0.0 if np.isnan(positions[t, p, k, 0]) else np.linalg.norm(positions[t, p, k] - np.array([frame_center_x, frame_center_y]))
                    motion_posture_features[t, p, k] = [dx, dy, confidence, dist_center]
        motion_posture_features = torch.tensor(motion_posture_features, dtype=torch.float32).to(device)
        print("Motion feature extraction finished.")

        return full_body_features, local_patch_features, motion_posture_features

    def __del__(self):
        print("Dataset object destroyed. Pose estimation resources managed by the framework.")

def compute_folder_name(folder_path):
    sequence_dir = os.path.basename(os.path.dirname(os.path.dirname(folder_path)))
    parent_name = os.path.basename(os.path.dirname(folder_path))
    cam_name = os.path.basename(folder_path)
    return f"{sequence_dir}_{parent_name}_{cam_name}"

# Feature Precomputation Function
def precompute_features(dataset, feature_dir='features_mp'):
    print(f"Precomputing features using pose estimation model into '{feature_dir}'...")
    os.makedirs(feature_dir, exist_ok=True)
    if dataset.feature_dir is not None:
        print("Warning: Dataset configured to load features. Re-initializing for extraction.")
        temp_dataset = InteractionDataset(dataset.folder_paths, dataset.labels, dataset.num_timesteps, feature_dir=None)
    else:
        temp_dataset = dataset

    if not hasattr(temp_dataset, 'pose_estimator') or temp_dataset.pose_estimator is None:
        raise RuntimeError("Pose estimator not initialized.")

    for idx in range(len(temp_dataset)):
        print(f"\nProcessing sample {idx+1}/{len(temp_dataset)} for precomputation...")
        folder_path = temp_dataset.folder_paths[idx]
        folder_name = compute_folder_name(folder_path)
        json_path = os.path.join(feature_dir, f"{folder_name}.json")

        if os.path.exists(json_path):
            print(f"Features already exist for {folder_path}, skipping.")
            continue

        try:
            frames = temp_dataset.extract_frames(folder_path)
            if not frames:
                print(f"Skipping sample {idx} due to no frames: {folder_path}")
                continue
            full_body, local_patch, motion_posture = temp_dataset.extract_features(frames)
            print(f"Feature extraction finished for {folder_path}.")

            full_body_path = os.path.join(feature_dir, f"{folder_name}_full_body.npy")
            local_patch_path = os.path.join(feature_dir, f"{folder_name}_local_patch.npy")
            motion_posture_path = os.path.join(feature_dir, f"{folder_name}_motion_posture.npy")

            np.save(full_body_path, full_body.cpu().numpy())
            np.save(local_patch_path, local_patch.cpu().numpy())
            np.save(motion_posture_path, motion_posture.cpu().numpy())

            feature_dict = {
                "full_body_features": full_body_path,
                "local_patch_features": local_patch_path,
                "motion_posture_features": motion_posture_path
            }
            with open(json_path, 'w') as f:
                json.dump(feature_dict, f, indent=4)
            print(f"Saved features for {folder_path} to {json_path}")
        except Exception as e:
            print(f"Error processing sample {idx} ({folder_path}): {e}")
            import traceback
            traceback.print_exc()
            continue
    print("Feature precomputation completed.")

# Model Definition
class InteractionRecognitionModel(nn.Module):
    def __init__(self, num_classes, num_timesteps=20, num_key_parts=5, resnet_dim=2048, motion_dim=4):
        super(InteractionRecognitionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_key_parts = num_key_parts
        self.resnet_dim = resnet_dim
        self.motion_dim = motion_dim

        self.lstm_full_body = nn.LSTM(resnet_dim, 256, batch_first=True)
        local_input_dim = 2 * num_key_parts * resnet_dim
        self.lstm_local_patch = nn.LSTM(local_input_dim, 512, batch_first=True)
        motion_input_dim = 2 * num_key_parts * motion_dim
        self.lstm_motion = nn.LSTM(motion_input_dim, 128, batch_first=True)
        combined_dim = 256 + 512 + 128
        self.fc = nn.Linear(combined_dim, num_classes)

    def forward(self, full_body, local_patch, motion_posture):
        out_fb, _ = self.lstm_full_body(full_body)
        fb_features = out_fb[:, -1, :]

        B, T, P, K, RF = local_patch.shape
        local_patch_flat = local_patch.view(B, T, -1)
        out_lp, _ = self.lstm_local_patch(local_patch_flat)
        lp_features = out_lp[:, -1, :]

        B, T, P, K, MF = motion_posture.shape
        motion_posture_flat = motion_posture.view(B, T, -1)
        out_mp, _ = self.lstm_motion(motion_posture_flat)
        mp_features = out_mp[:, -1, :]

        combined = torch.cat((fb_features, lp_features, mp_features), dim=1)
        output = self.fc(combined)
        return output

# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        running_loss = 0.0
        samples_processed = 0
        for i, (full_body, local_patch, motion_posture, labels) in enumerate(dataloader):
            full_body = full_body.to(device)
            local_patch = local_patch.to(device)
            motion_posture = motion_posture.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(full_body, local_patch, motion_posture)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * full_body.size(0)
            samples_processed += full_body.size(0)

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / samples_processed
        print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    # Define interaction directories with labels
    interaction_dirs = [
        (0, '04_sword_part1'),      # Label 0 for sword
        (1, '001_hugging'),         # Label 1 for hugging
        (2, '02_grappling'),        # Label 2 for grappling
        (3, '07_ballroom'),         # Label 3 for ballroom
        (4, '12_mma'),              # Label 4 for mma
    ]

    # Collect all camXX folders and corresponding labels for training
    folder_paths = []
    labels = []
    print("Searching for train data folders...")
    for label, top_dir in interaction_dirs:
        if not os.path.isdir(top_dir):
            print(f"Warning: Train directory not found: {top_dir}. Skipping.")
            continue
        print(f"Searching in {top_dir} for label {label}...")
        for root, dirs, files in os.walk(top_dir):
            for d in dirs:
                if d.startswith('cam') and os.path.isdir(os.path.join(root, d)):
                    cam_path = os.path.join(root, d)
                    folder_paths.append(cam_path)
                    labels.append(label)
                    print(f"  Found: {cam_path}")

    if not folder_paths:
        print("\nError: No 'camXX' folders found in the specified train interaction directories.")
        exit()

    # Collect all camXX folders and corresponding labels for testing
    test_folder_paths = []
    test_labels = []
    print("\nSearching for test data folders...")
    for label, top_dir in interaction_dirs:
        test_top_dir = os.path.join('test', top_dir)
        if not os.path.isdir(test_top_dir):
            print(f"Warning: Test directory not found: {test_top_dir}. Skipping.")
            continue
        print(f"Searching in {test_top_dir} for label {label}...")
        for root, dirs, files in os.walk(test_top_dir):
            for d in dirs:
                if d.startswith('cam') and os.path.isdir(os.path.join(root, d)):
                    cam_path = os.path.join(root, d)
                    test_folder_paths.append(cam_path)
                    test_labels.append(label)
                    print(f"  Found: {cam_path}")

    if not test_folder_paths:
        print("\nError: No 'camXX' folders found in the specified test interaction directories.")
        exit()

    # Verify data found
    print(f"\nTotal train samples found: {len(folder_paths)}")
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Train label distribution: {label_counts}")
    print(f"Sample train folder paths: {folder_paths[:2]}...")

    print(f"\nTotal test samples found: {len(test_folder_paths)}")
    test_label_counts = Counter(test_labels)
    print(f"Test label distribution: {test_label_counts}")
    print(f"Sample test folder paths: {test_folder_paths[:2]}...")

    num_classes = len(set(labels))
    print(f"Number of classes: {num_classes}")

    # Configuration
    NUM_TIMESTEPS = 20
    BATCH_SIZE = 4
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 150
    PRECOMPUTE = True
    TRAIN_FEATURE_DIR = 'features_mp'
    TEST_FEATURE_DIR = 'features_mp_test'
    MODEL_SAVE_PATH = "interaction_model_mediapipe.pth"

    # Feature Precomputation for train
    if PRECOMPUTE:
        print("\n--- Starting Train Feature Precomputation ---")
        extraction_dataset = InteractionDataset(folder_paths, labels, num_timesteps=NUM_TIMESTEPS, feature_dir=None)
        precompute_features(extraction_dataset, feature_dir=TRAIN_FEATURE_DIR)
        print("--- Train Feature Precomputation Finished ---")
        del extraction_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Feature Precomputation for test
    if PRECOMPUTE:
        print("\n--- Starting Test Feature Precomputation ---")
        test_extraction_dataset = InteractionDataset(test_folder_paths, test_labels, num_timesteps=NUM_TIMESTEPS, feature_dir=None)
        precompute_features(test_extraction_dataset, feature_dir=TEST_FEATURE_DIR)
        print("--- Test Feature Precomputation Finished ---")
        del test_extraction_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Dataset and DataLoader for train
    print(f"\n--- Loading Train Dataset (using features from '{TRAIN_FEATURE_DIR if PRECOMPUTE else 'on-the-fly'}') ---")
    try:
        dataset = InteractionDataset(folder_paths, labels, num_timesteps=NUM_TIMESTEPS, feature_dir=TRAIN_FEATURE_DIR if PRECOMPUTE else None)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error initializing train dataset or dataloader: {e}")
        exit()

    # Dataset and DataLoader for test
    print(f"\n--- Loading Test Dataset (using features from '{TEST_FEATURE_DIR if PRECOMPUTE else 'on-the-fly'}') ---")
    try:
        test_dataset = InteractionDataset(test_folder_paths, test_labels, num_timesteps=NUM_TIMESTEPS, feature_dir=TEST_FEATURE_DIR if PRECOMPUTE else None)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error initializing test dataset or dataloader: {e}")
        exit()

    # Model Initialization
    print("\n--- Initializing Model ---")
    model = InteractionRecognitionModel(num_classes=num_classes, num_timesteps=NUM_TIMESTEPS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    print("\n--- Starting Training ---")
    try:
        train_model(model, dataloader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device)
    except Exception as e:
        print(f"Error during training: {e}")
        exit()

    print("\n--- Training Finished ---")

    # Testing
    print("\n--- Starting Testing ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for full_body, local_patch, motion_posture, labels in test_dataloader:
            full_body = full_body.to(device)
            local_patch = local_patch.to(device)
            motion_posture = motion_posture.to(device)
            labels = labels.to(device)
            outputs = model(full_body, local_patch, motion_posture)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Save Model
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")