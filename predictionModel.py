# --- predict.py ---

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.models as models
import mediapipe as mp # Import MediaPipe
import torch.nn.functional as F
import argparse
import traceback
import gc

# --- Import the necessary components from your training script ---
# Make sure Fullcode.py is in the same directory or Python path
try:
    # Assuming InteractionRecognitionModel is defined in Fullcode.py
    from Fullcode import InteractionRecognitionModel
    # You might need other constants if InteractionRecognitionModel depends on them
    # from Fullcode import NUM_KEY_PARTS, RESNET_DIM, MOTION_DIM # etc.
except ImportError:
    print("Error: Could not import InteractionRecognitionModel from Fullcode.py.")
    print("Make sure Fullcode.py is in the same directory or accessible in your PYTHONPATH.")
    exit()
# --- END Import ---

# --- Configuration (MUST match the settings used for TRAINING the loaded model) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

NUM_TIMESTEPS = 20        # Number of frames processed per prediction
NUM_CLASSES = 5           # Number of interaction classes model was trained on
CLASS_NAMES = {           # Map indices to human-readable names
    0: 'Sword',
    1: 'Hugging',
    2: 'Grappling',
    3: 'Ballroom',
    4: 'MMA'
}
# Ensure CLASS_NAMES covers all indices from 0 to NUM_CLASSES-1 and matches training labels

# Feature dimensions (Must match model architecture)
NUM_KEY_PARTS = 5         # Number of key body parts used (e.g., Nose, LWrist, RWrist, LAnkle, RAnkle)
RESNET_DIM = 2048         # Output dimension of the ResNet feature extractor
MOTION_DIM = 4            # Dimension of motion features per keypoint [dx, dy, conf, dist_center]
# --- END Configuration ---


# --- Frame Extraction Helper ---
def _extract_frames_from_video(video_path, num_timesteps):
    """Extracts and samples frames from a single video file."""
    print(f"Extracting {num_timesteps} frames from video: {video_path}...")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"Error: Video file seems empty or has invalid metadata: {video_path}")
        return []

    indices = []
    if total_frames >= num_timesteps:
        # Sample frames evenly across the video
        indices = np.linspace(0, total_frames - 1, num_timesteps, dtype=int)
    else:
        # If video is shorter than num_timesteps, take all frames and duplicate the last one
        indices = list(range(total_frames)) + [total_frames - 1] * (num_timesteps - total_frames)
        print(f"Warning: Video has only {total_frames} frames. Duplicating last frame to reach {num_timesteps}.")

    selected_frames = []
    last_good_frame = None
    frames_read = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            selected_frames.append(frame.copy())
            last_good_frame = frame.copy()
            frames_read += 1
        else:
            # Advanced fallback: Try reading the very next frame sequentially if specific index fails
            print(f"Warning: Failed to read frame at index {idx}. Trying sequential read...")
            ret, frame = cap.read() # Read the next available frame
            if ret:
                selected_frames.append(frame.copy())
                last_good_frame = frame.copy()
                frames_read += 1
                print(f"  Success: Used next sequential frame instead for step {i}.")
            elif last_good_frame is not None:
                 # If sequential read also fails, duplicate the last known good frame
                print(f"Warning: Sequential read also failed. Duplicating last good frame for step {i}.")
                selected_frames.append(last_good_frame.copy())
            else:
                # Critical failure if no frames could be read at all initially
                print(f"Error: Critical frame read failure at step {i} in {video_path}. No previous frame to duplicate.")
                cap.release()
                return [] # Abort if reading fails early

    cap.release()

    # Final check to ensure we have the correct number of frames
    if len(selected_frames) != num_timesteps:
         print(f"Error: Frame extraction resulted in {len(selected_frames)} frames, expected {num_timesteps}.")
         # Attempt to pad if slightly short, otherwise fail
         if 0 < len(selected_frames) < num_timesteps and last_good_frame is not None:
             print(f"Padding with last frame to reach {num_timesteps}...")
             selected_frames.extend([last_good_frame.copy()] * (num_timesteps - len(selected_frames)))
         else:
             return [] # Return empty if extraction fundamentally failed

    print(f"Successfully extracted {len(selected_frames)} frames.")
    return selected_frames


# --- Feature Extraction Helper (Using MediaPipe) ---
def _extract_features_single_video(frames, num_timesteps, device):
    """Extracts features for a list of frames using MediaPipe and ResNet."""
    print("Extracting features using MediaPipe and ResNet...")
    if len(frames) != num_timesteps:
         print(f"Error: Expected {num_timesteps} frames for feature extraction, got {len(frames)}")
         return None, None, None

    pose_estimator_instance = None
    resnet_instance = None
    mp_pose_module = mp.solutions.pose

    try:
        # --- Initialize models needed for extraction ---
        pose_estimator_instance = mp_pose_module.Pose(
            static_image_mode=True, # Process each frame independently
            model_complexity=1,     # 0, 1, 2 (higher = more accurate, slower)
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        print("  - MediaPipe Pose model initialized.")

        resnet_instance = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1]).to(device)
        resnet_instance.eval()
        print("  - ResNet50 model initialized.")
        # --- End Initialization ---

        # --- Preprocessing (must match training) ---
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # --- End Preprocessing ---

        # --- Feature Extraction Logic ---

        # 1. Full-body features (ResNet on whole frames)
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        frames_tensor = torch.stack([preprocess(fr) for fr in frames_rgb]).to(device)
        with torch.no_grad():
            feature_maps = resnet_instance(frames_tensor) # Shape: (T, ResNet_Dim, 1, 1)
            full_body_features = feature_maps.view(num_timesteps, -1) # Shape: (T, ResNet_Dim)
        print("  - Full-body features extracted.")

        # 2. Keypoint extraction (MediaPipe) and Local Patches (ResNet on key parts)

        # Define mapping from TRAINING indices (0, 4, 7, 11, 14) to MediaPipe landmarks
        # Ensure these original indices match exactly what NUM_KEY_PARTS=5 implies was used in training
        key_parts_original_indices = [0, 4, 7, 11, 14] # Example: Nose, RWrist, LWrist, RAnkle, LAnkle
        if len(key_parts_original_indices) != NUM_KEY_PARTS:
             raise ValueError(f"Mismatch: NUM_KEY_PARTS is {NUM_KEY_PARTS} but {len(key_parts_original_indices)} indices provided.")

        # Map original indices to MediaPipe PoseLandmark enums
        original_to_mp_map = {
            0: mp_pose_module.PoseLandmark.NOSE,
            4: mp_pose_module.PoseLandmark.RIGHT_WRIST,
            7: mp_pose_module.PoseLandmark.LEFT_WRIST,
            11: mp_pose_module.PoseLandmark.RIGHT_ANKLE,
            14: mp_pose_module.PoseLandmark.LEFT_ANKLE,
            # Add other mappings here if key_parts_original_indices is different
        }
        # Verify all required indices are mapped
        for idx in key_parts_original_indices:
            if idx not in original_to_mp_map:
                raise ValueError(f"Missing MediaPipe mapping for original key part index: {idx}")

        # Initialize storage arrays
        # Shape: (Time, Person, KeyPart, Coordinate) - Person 0=Detected, Person 1=Padded
        positions = np.full((num_timesteps, 2, NUM_KEY_PARTS, 2), np.nan, dtype=np.float32)
        # Shape: (Time, Person, KeyPart)
        confidences = np.zeros((num_timesteps, 2, NUM_KEY_PARTS), dtype=np.float32)
        # List to store patch features per timestep, each element shape: (2, NUM_KEY_PARTS, RESNET_DIM)
        local_patch_features_list = []

        print("  - Extracting keypoints and local patches...")
        for t, frame in enumerate(frames): # Use original BGR frames for patch extraction
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                print(f"Warning: Skipping empty frame at timestep {t}")
                # Append zero features for this timestep to maintain structure
                zero_features_p1 = torch.zeros((NUM_KEY_PARTS, RESNET_DIM), device=device)
                zero_features_p2 = torch.zeros((NUM_KEY_PARTS, RESNET_DIM), device=device)
                local_patch_features_list.append(torch.stack([zero_features_p1, zero_features_p2]))
                continue

            rgb_frame = frames_rgb[t] # Use pre-converted RGB for MediaPipe input
            results = pose_estimator_instance.process(rgb_frame)

            # Initialize features for this timestep (P1 detected, P2 padded)
            frame_patch_features_p1 = torch.zeros((NUM_KEY_PARTS, RESNET_DIM), device=device)
            frame_patch_features_p2 = torch.zeros((NUM_KEY_PARTS, RESNET_DIM), device=device) # Padded Person 2

            # --- Process Person 1 (if detected by MediaPipe) ---
            if results.pose_landmarks:
                person1_landmarks = results.pose_landmarks.landmark
                num_detected_landmarks = len(person1_landmarks)

                for k, original_idx in enumerate(key_parts_original_indices):
                    mp_landmark_enum = original_to_mp_map.get(original_idx)
                    mp_idx = mp_landmark_enum.value # Get the integer index for the landmark

                    # Check if landmark index is valid for the detected pose
                    if mp_idx < num_detected_landmarks:
                        landmark = person1_landmarks[mp_idx]

                        # Use visibility as confidence (can be None)
                        conf = landmark.visibility if landmark.visibility is not None else 0.0

                        if conf > 0.1: # Confidence threshold
                            cx = int(landmark.x * w)
                            cy = int(landmark.y * h)

                            # Store position and confidence for Person 1
                            positions[t, 0, k, 0] = cx
                            positions[t, 0, k, 1] = cy
                            confidences[t, 0, k] = conf

                            # Extract local patch (using BGR frame)
                            patch_size = 64
                            x1 = max(0, cx - patch_size // 2)
                            y1 = max(0, cy - patch_size // 2)
                            x2 = min(w, cx + patch_size // 2)
                            y2 = min(h, cy + patch_size // 2)
                            patch = frame[y1:y2, x1:x2]

                            # Handle empty or smaller patches robustly (resize/pad)
                            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                                # Create a black patch if extraction failed completely
                                patch_processed = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                                print(f"Warning: Empty patch at t={t}, k={k}. Using black patch.")
                            elif patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                                # Resize if patch is valid but wrong size
                                patch_processed = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                            else:
                                patch_processed = patch

                            # Extract features from patch
                            patch_rgb = cv2.cvtColor(patch_processed, cv2.COLOR_BGR2RGB)
                            patch_tensor = preprocess(patch_rgb).to(device).unsqueeze(0)
                            with torch.no_grad():
                                feature = resnet_instance(patch_tensor).view(-1) # Shape: (RESNET_DIM)
                            frame_patch_features_p1[k] = feature
                    # Else (landmark not visible enough): Keep patch feature as zeros (already initialized)
            # Else (no pose detected): Keep all P1 patch features as zeros

            # --- Stack features for P1 and P2 for this timestep ---
            # Appends a tensor of shape: (2, NUM_KEY_PARTS, RESNET_DIM)
            local_patch_features_list.append(torch.stack([frame_patch_features_p1, frame_patch_features_p2]))

        # Stack across time
        # Final Shape: (num_timesteps, 2, NUM_KEY_PARTS, RESNET_DIM)
        local_patch_features = torch.stack(local_patch_features_list).to(device)
        print("  - Keypoints and local patches extracted (Person 1 detected, Person 2 padded).")

        # 3. Motion and Posture features (calculation must match training)
        motion_posture_features = np.zeros((num_timesteps, 2, NUM_KEY_PARTS, MOTION_DIM), dtype=np.float32) # T, P, K, (dx, dy, conf, dist)

        for t in range(num_timesteps):
            # Get frame center (handle potential issues if frames list got corrupted)
            frame_center_x, frame_center_y = 0.0, 0.0
            if t < len(frames) and frames[t] is not None and frames[t].shape[0] > 0 and frames[t].shape[1] > 0:
                 frame_center_x = frames[t].shape[1] / 2.0
                 frame_center_y = frames[t].shape[0] / 2.0
            else:
                 print(f"Warning: Could not get dimensions for frame {t} for center calculation.")

            for p in range(2): # Person 1 (Detected) and Person 2 (Padded)
                for k in range(NUM_KEY_PARTS):
                    # Motion (dx, dy) - Handle NaNs and t=0
                    dx = dy = 0.0
                    # Check if current and previous positions are valid numbers
                    if t > 0 and not np.isnan(positions[t, p, k, 0]) and not np.isnan(positions[t-1, p, k, 0]) \
                       and not np.isnan(positions[t, p, k, 1]) and not np.isnan(positions[t-1, p, k, 1]):
                        dx = positions[t, p, k, 0] - positions[t-1, p, k, 0]
                        dy = positions[t, p, k, 1] - positions[t-1, p, k, 1]

                    # Confidence (already computed)
                    confidence = confidences[t, p, k]

                    # Distance from center - Handle NaNs and invalid center
                    dist_center = 0.0
                    if not np.isnan(positions[t, p, k, 0]) and not np.isnan(positions[t, p, k, 1]) \
                       and (frame_center_x != 0 or frame_center_y != 0): # Ensure center is valid
                         dist_center = np.linalg.norm(positions[t, p, k] - np.array([frame_center_x, frame_center_y]))

                    motion_posture_features[t, p, k] = [dx, dy, confidence, dist_center]

        motion_posture_features = torch.tensor(motion_posture_features, dtype=torch.float32).to(device) # Shape: (T, 2, K, MOTION_DIM)
        print("  - Motion/posture features calculated.")
        # --- End Feature Extraction ---

        print("Feature extraction finished.")
        return full_body_features, local_patch_features, motion_posture_features

    except Exception as e:
         print(f"Error during feature extraction: {e}")
         traceback.print_exc()
         return None, None, None # Indicate failure
    finally:
        # Clean up models initialized specifically within this function
        print("Cleaning up feature extraction resources...")
        if pose_estimator_instance:
             try:
                  pose_estimator_instance.close() # Explicitly close MediaPipe instance
             except Exception as close_e:
                  print(f"Error closing MediaPipe pose estimator: {close_e}")
             del pose_estimator_instance
        if resnet_instance: del resnet_instance
        gc.collect() # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear PyTorch's CUDA cache


# --- Prediction Function ---
def predict_video(video_path, model_path):
    """Loads model, processes video using MediaPipe, and predicts interaction."""
    print(f"\n--- Predicting Video: {video_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    model = None # Initialize model variable

    try:
        # --- Load Model ---
        print("Loading model architecture...")
        # Instantiate the model with parameters matching the saved one
        model = InteractionRecognitionModel(
            num_classes=NUM_CLASSES,
            num_timesteps=NUM_TIMESTEPS,
            num_key_parts=NUM_KEY_PARTS,
            resnet_dim=RESNET_DIM,
            motion_dim=MOTION_DIM
        ).to(DEVICE)

        print(f"Loading model weights from {model_path}...")
        # Load the state dictionary
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        print("Model loaded successfully.")
        # --- End Load Model ---

        # --- Process Video ---
        frames = _extract_frames_from_video(video_path, NUM_TIMESTEPS)
        if not frames: # Check if frame extraction was successful
            print("Prediction aborted: Failed to extract frames.")
            return # Exit early

        # Extract features using MediaPipe
        full_body, local_patch, motion_posture = _extract_features_single_video(frames, NUM_TIMESTEPS, DEVICE)
        if full_body is None or local_patch is None or motion_posture is None:
            print("Prediction aborted: Feature extraction failed.")
            return # Exit early
        # --- End Process Video ---


        # --- Perform Inference ---
        print("Performing inference...")
        # Add batch dimension (B=1) as the model expects batch input
        batch_full_body = full_body.unsqueeze(0)         # Shape: (1, T, ResNet_Dim)
        batch_local_patch = local_patch.unsqueeze(0)     # Shape: (1, T, 2, K, ResNet_Dim)
        batch_motion_posture = motion_posture.unsqueeze(0) # Shape: (1, T, 2, K, Motion_Dim)

        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(batch_full_body, batch_local_patch, batch_motion_posture)
            # Outputs are raw scores (logits), shape: (1, NUM_CLASSES)

            # Apply Softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1) # Shape: (1, NUM_CLASSES)

            # Get the most likely class index and its confidence
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_idx = predicted_idx.item() # Convert tensor index to Python int
        confidence_score = confidence.item() # Convert tensor confidence to Python float
        predicted_label = CLASS_NAMES.get(predicted_idx, f"Unknown Index {predicted_idx}") # Map index to name

        print("\n--- Prediction Result ---")
        print(f"  Video:           {os.path.basename(video_path)}")
        print(f"  Predicted Class: {predicted_label} (Index: {predicted_idx})")
        print(f"  Confidence:      {confidence_score:.2%}") # Format as percentage
        # --- End Inference ---

    except Exception as e:
        print(f"\n--- ERROR during prediction pipeline ---")
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Details: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # --- Clean up resources ---
        print("Cleaning up prediction resources...")
        if model: del model # Delete the model instance
        gc.collect() # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
        print("--- Prediction Finished ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict human interaction from a single video file using a trained model and MediaPipe.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("-m", "--model_path", default="interaction_model_mediapipe.pth",
                        help="Path to the trained interaction recognition model file (.pth). Default: interaction_model_mediapipe.pth")
    # Example: Add argument to override model complexity for MediaPipe
    # parser.add_argument("--mp_complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe Pose model complexity (0, 1, or 2).")

    args = parser.parse_args()

    # Basic validation of input paths
    if not os.path.isfile(args.video_path):
         print(f"Error: Input video file not found or is not a file: '{args.video_path}'")
    elif not os.path.isfile(args.model_path):
         print(f"Error: Model file not found or is not a file: '{args.model_path}'")
    else:
        # Run the prediction function if paths are valid
        predict_video(args.video_path, args.model_path)