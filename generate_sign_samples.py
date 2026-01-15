#!/usr/bin/env python3
"""Generate minimal sample data for sign language methods."""
import numpy as np
import pickle
from pathlib import Path

# Create samples directory
samples_dir = Path(__file__).parent / "samples"

# ===== sign_asslisu sample =====
# It expects a folder with .pkl files containing skeleton keypoints
asslisu_dir = samples_dir / "sign_asslisu" / "video_0"
asslisu_dir.mkdir(parents=True, exist_ok=True)

# Create minimal skeleton data (30 frames, 137 keypoints for full body+hands+face, 3 coords)
num_frames = 30
num_keypoints = 137

for i in range(3):  # Create a few pkl files
    skeleton_data = {
        'keypoints': np.random.randn(num_frames, num_keypoints, 3).astype(np.float32) * 0.1,
        'scores': np.ones((num_frames, num_keypoints), dtype=np.float32)
    }
    pkl_path = asslisu_dir / f"frame_{i:04d}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(skeleton_data, f)

print(f"Created sign_asslisu sample at {asslisu_dir}")

# ===== sign_lmsls sample =====
try:
    from pose_format import Pose
    from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent
    from pose_format.pose_body import PoseBody
    from pose_format.numpy import NumPyPoseBody

    # Create minimal pose header
    dimensions = PoseHeaderDimensions(width=1920, height=1080, depth=1)

    # MediaPipe pose landmarks (33 points) + hands (21 each)
    components = [
        PoseHeaderComponent(name="POSE_LANDMARKS", points=["point_" + str(i) for i in range(33)],
                           limbs=[], colors=[], format="XYZC"),
        PoseHeaderComponent(name="LEFT_HAND_LANDMARKS", points=["point_" + str(i) for i in range(21)],
                           limbs=[], colors=[], format="XYZC"),
        PoseHeaderComponent(name="RIGHT_HAND_LANDMARKS", points=["point_" + str(i) for i in range(21)],
                           limbs=[], colors=[], format="XYZC"),
    ]

    header = PoseHeader(version=0.1, dimensions=dimensions, components=components)

    # Create pose data: (frames, people, points, dims)
    num_frames = 50
    num_people = 1
    num_points = 75  # 33 + 21 + 21
    num_dims = 4  # XYZC

    data = np.random.randn(num_frames, num_people, num_points, num_dims).astype(np.float32) * 100
    data[:, :, :, 3] = 1.0  # Confidence = 1

    confidence = data[:, :, :, 3:4]
    body = NumPyPoseBody(fps=25, data=data, confidence=confidence)
    pose = Pose(header=header, body=body)

    # Save
    lmsls_dir = samples_dir / "sign_lmsls"
    lmsls_dir.mkdir(parents=True, exist_ok=True)
    pose_path = lmsls_dir / "sample.pose"

    with open(pose_path, 'wb') as f:
        pose.write(f)

    print(f"Created sign_lmsls sample at {pose_path}")

except ImportError as e:
    print(f"Could not create sign_lmsls sample: {e}")
    print("Install with: pip install pose-format")
except Exception as e:
    print(f"Error creating sign_lmsls sample: {e}")

print("\nDone!")
