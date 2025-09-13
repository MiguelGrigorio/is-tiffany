import numpy as np

def angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the 2D angle in degrees between two vectors with orientation.

    The angle is measured from `v1` to `v2` in the 2D plane.  
    If the determinant is negative, the function interprets it as a clockwise angle.

    Args:
        v1 (np.ndarray): First 2D vector (e.g., reference vector).
        v2 (np.ndarray): Second 2D vector (e.g., target vector).

    Returns:
        float: Angle in degrees between 0 and 360, measured from `v1` to `v2`.
    """
    # Compute dot product and norms
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Compute cosine of angle and clip to [-1, 1] to avoid numerical errors
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)

    # Determine the orientation using the 2D cross product (determinant)
    determinant = v1[0] * v2[1] - v1[1] * v2[0]
    if determinant < 0:
        theta_deg = 360 - theta_deg

    return theta_deg
