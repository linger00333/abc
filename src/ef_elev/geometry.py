"""Independent A1/A2 geometry. No labels or model predictions are modified."""
import math
import numpy as np


def axis_rotation(axis, signed_degrees):
    """OpenCV axes: pitch about x/right; roll about z/forward; yaw about y/down.

    This rotates point coordinates, p_new = delta @ p_old. Positive sign is
    defined by the matrix, not by an ambiguous physical 'nose up' description.
    """
    a = math.radians(float(signed_degrees))
    c, s = math.cos(a), math.sin(a)
    if axis == 'pitch':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 'roll':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    if axis == 'yaw':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    raise ValueError(axis)


def pose_r_vert2cam(sample, lidar_to_camera_r):
    """Reproduce frozen RoadBEV's pose conventions, including 0.017453/1.5708.

    These constants intentionally match the reference; do not silently replace
    them in only one model's data path.
    """
    pitch, roll, yaw = [0.017453 * float(sample[k]) for k in ('pitch', 'roll', 'yaw')]
    if np.pi <= yaw <= 2*np.pi:
        yaw -= 2*np.pi
    cx, sx, cy, sy, cz, sz = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.asarray(lidar_to_camera_r) @ np.linalg.inv(rz @ rx @ ry)
    a, f, g, k, l, n, e = m.reshape(-1)[[0, 1, 2, 4, 5, 7, 8]]
    angle_y = np.arcsin(np.clip(g, -1, 1))
    if 0.99999 > abs(g):
        angle_x, angle_z = np.arctan2(-l, e), np.arctan2(-f, a)
    else:
        angle_x, angle_z = np.arctan2(n, k), 0.0
    angle_x, angle_y, angle_z = np.array([angle_x, angle_y, angle_z], dtype=np.float32)
    angle_x -= 1.5708
    rx_cam = np.array([[1, 0, 0], [0, np.cos(angle_x), np.sin(angle_x)], [0, -np.sin(angle_x), np.cos(angle_x)]], dtype=np.float32)
    # Reference calls the SECOND XYZ Euler entry roll_cam; reproduce that
    # convention even though the subsequent matrix rotates around camera z.
    rz_cam = np.array([[np.cos(angle_y), np.sin(angle_y), 0], [-np.sin(angle_y), np.cos(angle_y), 0], [0, 0, 1]], dtype=np.float32)
    return np.linalg.inv(rx_cam @ rz_cam).astype(np.float32)


def estimated_rotation(mode, r_true, delta):
    """A1 changes estimate only. A2 stale/oracle share the SAME warped image."""
    if mode in ('A1', 'A2_oracle'):
        if np.array_equal(delta, np.eye(3)):
            return np.array(r_true, copy=True)
        return delta @ r_true
    if mode in ('clean', 'A2_stale'):
        return np.array(r_true, copy=True)
    raise ValueError(mode)


def warp_image_a2(source_rgb, k_half, delta, output_wh=(960, 528)):
    """Warp full 960x540 source BEFORE cropping. Constant borders + strict mask.

    Zero change bypasses interpolation to preserve original pixels exactly. K_half is
    used for synthetic image creation, never passed to a REO network.
    """
    w, h = map(int, output_wh)
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise ValueError('Expected HxWx3 RGB')
    if w > source_rgb.shape[1] or h > source_rgb.shape[0]:
        raise ValueError('Crop exceeds source; padding is not allowed')
    if np.array_equal(delta, np.eye(3)):
        return source_rgb[:h, :w].copy(), np.ones((h, w), dtype=bool)
    k = np.asarray(k_half, dtype=np.float64)
    homography = k @ delta @ np.linalg.inv(k)
    # Existing OpenCV 4.8.1 binary rejects NumPy 2 arrays in this environment.
    # Use already-installed torch CPU bilinear interpolation; no env changes.
    import torch
    import torch.nn.functional as F
    src_h, src_w = source_rgb.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dest = np.stack([xx, yy, np.ones((h, w))]).reshape(3, -1)
    mapped = np.linalg.inv(homography) @ dest
    uv = mapped[:2] / mapped[2:]
    grid = np.stack([2*uv[0]/(src_w-1)-1, 2*uv[1]/(src_h-1)-1], axis=-1)
    grid = torch.from_numpy(grid.astype(np.float32).reshape(1, h, w, 2))
    pixels = torch.from_numpy(np.array(source_rgb, copy=True)).permute(2,0,1).unsqueeze(0).float()
    ones = torch.ones((1,1,src_h,src_w), dtype=torch.float32)
    with torch.inference_mode():
        transformed = F.grid_sample(pixels, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        support = F.grid_sample(ones, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    warped = transformed[0].permute(1,2,0).round().clamp(0,255).to(torch.uint8).numpy()
    return warped, support[0,0].numpy() >= 1.0-1e-6


def project_points(k, points_camera):
    uvz = np.asarray(k, dtype=np.float64) @ np.asarray(points_camera, dtype=np.float64)
    return uvz[:2] / uvz[2:3]


def project_feature_voxels(r, k_half, voxel_centers, downscale=4):
    """Reference float32 projection; return indices AND validity separately.

    Adapter must zero invalid gathered features; clamping indices by itself
    does not make those features valid. This function never invokes a model.
    """
    import torch
    r = torch.as_tensor(np.asarray(r, dtype=np.float32))
    k = torch.as_tensor(np.asarray(k_half, dtype=np.float32)/float(downscale))
    k[2, 2] = 1
    vertices = torch.as_tensor(voxel_centers, dtype=torch.float32)
    uvz = k @ (r @ vertices)
    coords = uvz[:2] / uvz[2:]
    valid = (uvz[2] > 0) & torch.isfinite(coords).all(dim=0)
    return torch.floor(coords).long(), valid
