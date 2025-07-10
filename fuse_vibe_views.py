import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pickle


# --------------------- Step 1: Load VIBE Output ---------------------
def load_vibe_output(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data['verts'], data['pose'], data['betas']


# ------------------ Step 2: Estimate Viewing Angle ------------------
def get_viewing_angle(pose_vector):
    """
    Estimate yaw angle from global orientation in SMPL pose vector.
    """
    global_orient = pose_vector[:3]
    r = R.from_rotvec(global_orient)
    yaw = r.as_euler('zyx', degrees=True)[0]  # Yaw angle
    return (yaw + 360) % 360  # Normalize


# ------------------ Step 3: Classify Frames by View ------------------
def classify_frames_by_view(poses):
    categories = {'front': [], 'right': [], 'back': [], 'left': []}
    for i, pose in enumerate(poses):
        yaw = get_viewing_angle(pose)
        if 315 <= yaw or yaw < 45:
            categories['front'].append(i)
        elif 45 <= yaw < 135:
            categories['right'].append(i)
        elif 135 <= yaw < 225:
            categories['back'].append(i)
        elif 225 <= yaw < 315:
            categories['left'].append(i)
    return categories


# ------------------ Step 4: Select Frame per View ------------------
def select_representative_frame(view_dict, verts):
    rep_frames = {}
    for view, indices in view_dict.items():
        if indices:
            mid_idx = indices[len(indices) // 2]
            rep_frames[view] = verts[mid_idx]
    return rep_frames


# ------------------ Step 5: Extract Half Mesh ------------------
def extract_half_mesh(vertices, view, axis='z'):
    """
    Extract front/back or left/right half of a mesh.
    """
    if axis == 'z':  # front/back cut
        cutoff = 0.0
        if view == 'front':
            mask = vertices[:, 2] > cutoff
        elif view == 'back':
            mask = vertices[:, 2] < cutoff
    elif axis == 'x':  # left/right cut
        cutoff = 0.0
        if view == 'left':
            mask = vertices[:, 0] > cutoff
        elif view == 'right':
            mask = vertices[:, 0] < cutoff
    else:
        raise ValueError("axis must be 'z' or 'x'")

    selected = vertices.copy()
    selected[~mask] = np.nan
    return selected


# ------------------ Step 6: Fuse Half Meshes ------------------
def fuse_half_meshes(rep_meshes):
    final = np.full_like(list(rep_meshes.values())[0], np.nan)

    for view, verts in rep_meshes.items():
        if view in ['front', 'back']:
            partial = extract_half_mesh(verts, view, axis='z')
        elif view in ['left', 'right']:
            partial = extract_half_mesh(verts, view, axis='x')
        else:
            continue

        mask = ~np.isnan(partial[:, 0])
        final[mask] = partial[mask]

    return final


# ------------------ Step 7: Visualize Mesh ------------------
def visualize_vertices(vertices, smpl_faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(smpl_faces)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


# ------------------ Utility: Load SMPL Faces ------------------
def load_smpl_faces(model_path):
    with open(model_path, 'rb') as f:
        smpl = pickle.load(f, encoding='latin1')
    return smpl['f']


# ------------------ MAIN DRIVER ------------------
def main(vibe_output_path, smpl_model_path):
    verts, poses, _ = load_vibe_output(vibe_output_path)
    print(f"Loaded {len(verts)} frames from VIBE output.")

    view_dict = classify_frames_by_view(poses)
    print("Classified frames by view:", {k: len(v) for k, v in view_dict.items()})

    rep_meshes = select_representative_frame(view_dict, verts)
    fused = fuse_half_meshes(rep_meshes)

    smpl_faces = load_smpl_faces(smpl_model_path)
    visualize_vertices(fused, smpl_faces)


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    main('vibe_output.npz', 'smpl_model.pkl') # Replace with actual paths to your VIBE output and SMPL model
