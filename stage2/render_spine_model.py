import open3d as o3d
import numpy as np
import json


def load_session(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_average_curve(curves):
    stacked = np.array([np.array(c) for c in curves])
    return np.mean(stacked, axis=0)

def average_metrics(session_data):
    metric_totals = {}
    counts = {}

    for frame in session_data:
        adv = frame.get("metrics", {}).get("advanced", {})
        for key, val in adv.items():
            if val is not None:
                metric_totals.setdefault(key, 0.0)
                counts.setdefault(key, 0)
                metric_totals[key] += val
                counts[key] += 1

    return {k: metric_totals[k] / counts[k] for k in metric_totals if counts[k] > 0}

# Transformation functions
def apply_body_lean(curve, angle_deg):
    angle = np.radians(angle_deg)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])
    return (R @ curve.T).T

def apply_pelvic_tilt(curve, angle_deg):
    angle = np.radians(angle_deg)
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    base = curve[0]
    centered = curve - base
    rotated = (R @ centered.T).T + base
    return rotated

def apply_forward_head_offset(curve, offset_cm):
    curve = curve.copy()
    curve[-1][2] += offset_cm
    return curve

def apply_neck_tilt(curve, tilt_deg):
    if len(curve) < 3:
        return curve
    angle = np.radians(tilt_deg)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])
    neck_base = curve[-3]
    upper = curve[-3:] - neck_base
    tilted = (R @ upper.T).T + neck_base
    curve[-3:] = tilted
    return curve

def apply_shoulder_tilt(curve, tilt_deg):
    if len(curve) < 3:
        return curve
    angle = np.radians(tilt_deg)
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    base = curve[-3]
    upper = curve[-3:] - base
    tilted = (R @ upper.T).T + base
    curve[-3:] = tilted
    return curve

def apply_curvature_ratio(curve, target_ratio):
    straight_height = np.linalg.norm(curve[-1] - curve[0])
    current_len = np.sum([np.linalg.norm(curve[i] - curve[i-1]) for i in range(1, len(curve))])
    current_ratio = current_len / straight_height if straight_height > 0 else 1.0
    factor = target_ratio / current_ratio if current_ratio > 0 else 1.0

    base = curve[0]
    new_curve = [base]

    for i in range(1, len(curve)):
        segment = curve[i] - curve[i - 1]
        new_point = new_curve[-1] + segment * factor
        new_curve.append(new_point)

    return np.array(new_curve)

def visualize_curve_open3d(curve):
    # Convert curve into Open3D LineSet
    points = o3d.utility.Vector3dVector(curve)
    lines = [[i, i+1] for i in range(len(curve) - 1)]
    colors = [[0.2, 0.8, 0.3] for _ in lines]  # greenish lines
    line_set = o3d.geometry.LineSet(points=points, lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Add coordinate frame for reference
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)

    # Create small spheres at joints
    spheres = []
    for pt in curve:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
        sphere.translate(pt)
        sphere.paint_uniform_color([1, 0.4, 0.1])  # orange
        spheres.append(sphere)

    # Visualize all
    o3d.visualization.draw_geometries([line_set, origin_frame] + spheres)

def main(json_path):
    data = load_session(json_path)
    curves = [np.array(frame["curve"]) for frame in data if "curve" in frame]
    if not curves:
        print("❌ No curve data found.")
        return

    avg_curve = compute_average_curve(curves)
    metrics = average_metrics(data)
    print("📊 Avg Metrics Used:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    # Apply all transformations
    transformed = avg_curve
    transformed = apply_body_lean(transformed, metrics.get("Body Lean", 0))
    transformed = apply_pelvic_tilt(transformed, metrics.get("Pelvic Tilt", 0))
    transformed = apply_forward_head_offset(transformed, metrics.get("Forward Head Offset", 0))
    transformed = apply_neck_tilt(transformed, metrics.get("Neck Tilt", 0))
    transformed = apply_shoulder_tilt(transformed, metrics.get("Shoulder Tilt", 0))
    transformed = apply_curvature_ratio(transformed, metrics.get("Spine Curvature Ratio", 1.0))

    visualize_curve_open3d(transformed)

if __name__ == "__main__":
    # Hardcoded path to your session file
    json_path = "/Users/connorv-e/Desktop/spinevis/spine_session_20250708_122411.json"  # 👈 Replace with your actual file
    main(json_path)