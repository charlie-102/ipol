#!/usr/bin/env python3
"""Generate proper sample data for sign language methods."""
import numpy as np
import pickle
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples"

def generate_sign_asslisu_sample():
    """Generate sample for sign_asslisu.

    Expects files named: {video}_{start}_{length}_data.pkl
    Each pkl contains skeleton data with shape (frames, keypoints, coords)
    """
    print("=== Generating sign_asslisu sample ===")

    # Create folder structure
    sample_dir = SAMPLES_DIR / "sign_asslisu" / "video_001"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Clean old files
    for f in sample_dir.glob("*.pkl"):
        f.unlink()

    # ASSLiSU expects *_data.pkl files with naming: {name}_{start}_{length}_data.pkl
    # The dataloader parses: start = split('_')[-3], length = split('_')[-2]

    num_frames = 250  # length
    start_frame = 0

    # Keypoint counts for different body types (from the code):
    # full: 137 keypoints (33 body + 21 left hand + 21 right hand + 62 face)
    # body: 33 (pose landmarks)
    # hands: 42 (21 + 21)
    # The default uses 'body' type which is 33 keypoints
    num_keypoints = 33  # for body type

    # Create skeleton data: (frames, keypoints, 3) for x, y, confidence
    skeleton_data = np.random.randn(num_frames, num_keypoints, 3).astype(np.float32)
    skeleton_data[:, :, 2] = 1.0  # confidence = 1

    # Normalize to reasonable range
    skeleton_data[:, :, :2] = skeleton_data[:, :, :2] * 0.1 + 0.5

    # Save with correct filename format: video_001_0_250_data.pkl
    filename = f"video_001_{start_frame}_{num_frames}_data.pkl"
    pkl_path = sample_dir / filename

    with open(pkl_path, 'wb') as f:
        pickle.dump(skeleton_data, f)

    print(f"  Created: {pkl_path}")
    print(f"  Shape: {skeleton_data.shape}")
    print(f"  Dtype: {skeleton_data.dtype}")


def generate_sign_lmsls_sample():
    """Generate sample for sign_lmsls.

    Uses pose-format library. Needs proper .pose file with float32 data.
    """
    print("\n=== Generating sign_lmsls sample ===")

    try:
        from pose_format import Pose
        from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent
        from pose_format.numpy import NumPyPoseBody
    except ImportError:
        print("  ERROR: pose-format not installed. Run: pip install pose-format")
        return

    sample_dir = SAMPLES_DIR / "sign_lmsls"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Clean old files
    for f in sample_dir.glob("*.pose"):
        f.unlink()

    # Create pose header
    dimensions = PoseHeaderDimensions(width=1920, height=1080, depth=1)

    # Components matching MediaPipe pose landmarks
    # POSE_LANDMARKS: 33 points with standard names for normalization
    # LEFT/RIGHT_HAND_LANDMARKS: 21 each = 75 total
    pose_points = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    hand_points = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]

    components = [
        PoseHeaderComponent(
            name="POSE_LANDMARKS",
            points=pose_points,
            limbs=[],
            colors=[],
            point_format="XYZC"
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_LANDMARKS",
            points=hand_points,
            limbs=[],
            colors=[],
            point_format="XYZC"
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_LANDMARKS",
            points=hand_points,
            limbs=[],
            colors=[],
            point_format="XYZC"
        ),
    ]

    header = PoseHeader(version=0.1, dimensions=dimensions, components=components)

    # Create pose data: (frames, people, points, dims)
    # MUST be float32 to avoid dtype mismatch with model
    # Data has XYZ (3 dims), confidence is separate
    num_frames = 100
    num_people = 1
    num_points = 75  # 33 + 21 + 21
    num_dims = 3  # XYZ only, confidence is separate

    # Generate normalized data in float32
    data = np.random.randn(num_frames, num_people, num_points, num_dims).astype(np.float32)

    # Normalize X, Y, Z to reasonable range [0, 1]
    data = np.clip(data * 0.1 + 0.5, 0, 1).astype(np.float32)

    # Confidence is separate array
    confidence = np.ones((num_frames, num_people, num_points, 1), dtype=np.float32)

    body = NumPyPoseBody(fps=25, data=data, confidence=confidence)
    pose = Pose(header=header, body=body)

    # Save
    pose_path = sample_dir / "sample.pose"
    with open(pose_path, 'wb') as f:
        pose.write(f)

    print(f"  Created: {pose_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")


if __name__ == "__main__":
    generate_sign_asslisu_sample()
    generate_sign_lmsls_sample()
    print("\n=== Done ===")
