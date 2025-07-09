"""
a script to check what is inside a .npy file for SMPL pose sequences
"""

import numpy as np
import torch
from pytorch3d import transforms

# Load the .npy file
motion_file = './kicking.npy'
data = np.load(motion_file, allow_pickle=True).item()

print("Loaded data keys:", data.keys())

for key in data.keys():
    print(f"{key}: {data[key].shape if isinstance(data[key], np.ndarray) else type(data[key])}")

# Test the correct transformations
start_idx, end_idx, skip = 0, 120, 1

print("\n=== Testing transformations ===")

# For thetas: (24, 6, 120) -> need (frames, joints*3)
thetas = data['thetas']  # (24, 6, 120)
print(f"Original thetas shape: {thetas.shape}")

# Transpose to (frames, joints, components) then reshape
thetas_transposed = thetas.transpose(2, 0, 1)  # (120, 24, 6)
print(f"After transpose: {thetas_transposed.shape}")

# Slice frames
thetas_sliced = thetas_transposed[start_idx:end_idx:skip]  # (120, 24, 6)
print(f"After slicing: {thetas_sliced.shape}")

print("\n=== Converting 6D to axis-angle ===")

# Convert to PyTorch tensor
thetas_tensor = torch.from_numpy(thetas_sliced).float()  # (120, 24, 6)

# Reshape to (frames*joints, 6) for batch processing
frames, joints, components = thetas_tensor.shape
thetas_flat = thetas_tensor.reshape(-1, 6)  # (120*24, 6)
print(f"Flattened shape: {thetas_flat.shape}")

# Convert 6D to rotation matrix
rot_matrices = transforms.rotation_6d_to_matrix(thetas_flat)  # (120*24, 3, 3)
print(f"Rotation matrices shape: {rot_matrices.shape}")

# Convert rotation matrix to axis-angle
axis_angles = transforms.matrix_to_axis_angle(rot_matrices)  # (120*24, 3)
print(f"Axis angles shape: {axis_angles.shape}")

# Reshape back to (frames, joints, 3)
axis_angles_reshaped = axis_angles.reshape(frames, joints, 3)  # (120, 24, 3)
print(f"Reshaped axis angles: {axis_angles_reshaped.shape}")

# Final reshape to (frames, joints*3) for SMPL
poses_converted = axis_angles_reshaped.reshape(frames, -1)  # (120, 72)
print(f"Final poses shape: {poses_converted.shape}")

# Convert back to numpy if needed
poses_numpy = poses_converted.numpy()

print(f"\nSuccess! Converted to SMPL format:")
print(f"poses: {poses_numpy.shape} ✓ (matches expected (frames, 72))")

# For root_translation: (3, 120) -> (frames, 3)
root_trans = data['root_translation']  # (3, 120)
print(f"\nOriginal root_translation shape: {root_trans.shape}")

# Transpose to (frames, xyz)
transl = root_trans.T[start_idx:end_idx:skip]  # (120, 3)
print(f"Final transl shape: {transl.shape}")

print(f"\nBoth formats ready for SMPL:")
print(f"poses: {poses_numpy.shape}")
print(f"transl: {transl.shape}")

# Test a few values to make sure they look reasonable
print(f"\nSample axis-angle values (should be reasonable rotation angles):")
print(f"First joint, first frame: {poses_numpy[0, :3]}")
print(f"Magnitudes: {np.linalg.norm(poses_numpy[0, :3])}")

# Save in .npz format expected by NeuMan dataset
print("\n=== Saving to .npz format ===")

# The NeuMan dataset expects:
# - 'poses': (frames, 72) - body pose parameters in axis-angle format
# - 'trans': (frames, 3) - root translation

output_file = './kicking.npz'
np.savez(output_file, 
         poses=poses_numpy,  # (120, 72)
         trans=transl        # (120, 3)
)

print(f"Saved to {output_file}")
print(f"Contains:")
print(f"  poses: {poses_numpy.shape}")
print(f"  trans: {transl.shape}")

# Verify the saved file
print("\n=== Verifying saved file ===")
saved_data = np.load(output_file)
print(f"Loaded keys: {list(saved_data.keys())}")
print(f"poses shape: {saved_data['poses'].shape}")
print(f"trans shape: {saved_data['trans'].shape}")

print("\nFile ready for NeuMan dataset!")