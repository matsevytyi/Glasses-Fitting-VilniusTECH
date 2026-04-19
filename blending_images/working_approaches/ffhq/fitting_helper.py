import cv2
import math
import numpy as np
from PIL import Image
import mediapipe as mp

# Initialize MediaPipe Face Mesh once
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def get_facial_landmarks(image_cv2):
    height, width, _ = image_cv2.shape
    results = face_mesh.process(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None
        
    landmarks = results.multi_face_landmarks[0]
    return [(int(point.x * width), int(point.y * height)) for point in landmarks.landmark]

def fit_glasses_to_face(bg_img_pil, glasses_img_pil, glasses_mask_pil, scale_adjustment=1.0):
    """
    Takes a background face, glasses, and mask, and resizes/aligns the glasses 
    to fit the face geometrically before passing them to the harmonization algorithm.
    
    scale_adjustment: Default 1.0. Lower to 0.8 if glasses still look too big.
    Returns: (aligned_glasses_img, aligned_mask_img) ready for Poisson or Diffusion blending.
    """
    
    # 1. Convert PIL to CV2 for MediaPipe
    bg_cv2 = cv2.cvtColor(np.array(bg_img_pil), cv2.COLOR_RGB2BGR)
    landmarks = get_facial_landmarks(bg_cv2)
    
    if not landmarks:
        print("Warning: No face detected. Returning original glasses.")
        return glasses_img_pil, glasses_mask_pil
        
    # 2. Extract key landmarks (based on your engine-2.py logic)
    # 168: Nose bridge, 234: Left face edge, 454: Right face edge, 162/389: Eye alignment
    nose_bridge = landmarks[168]
    left_edge = landmarks[234]
    right_edge = landmarks[454]
    left_eye_align = landmarks[162]
    right_eye_align = landmarks[389]
    
    # 3. Calculate Face Rotation and Width
    # rot_angle = arctan((left.y - right.y) / (right.x - left.x))
    dx = right_eye_align[0] - left_eye_align[0]
    dy = left_eye_align[1] - right_eye_align[1]
    rot_angle_rad = math.atan2(dy, dx)
    rot_angle_deg = math.degrees(rot_angle_rad)
    
    # Calculate physical face width
    face_width = math.sqrt(dx**2 + dy**2)
    
    # 4. Calculate new glasses dimensions
    glasses_cv2 = cv2.cvtColor(np.array(glasses_img_pil), cv2.COLOR_RGB2BGR)
    orig_g_height, orig_g_width = glasses_cv2.shape[:2]
    
    # Mathematical scaling: We want the glasses width to roughly match the face width 
    # (plus a little extra for the frames). We apply the scale_adjustment here.
    target_width = int((face_width * 1.15) * scale_adjustment)
    target_height = int(orig_g_height * (target_width / orig_g_width))
    
    # 5. Resize glasses and mask
    glasses_resized = cv2.resize(glasses_cv2, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    mask_cv2 = np.array(glasses_mask_pil)
    mask_resized = cv2.resize(mask_cv2, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    # 6. Create empty canvases (same size as the background image)
    bg_height, bg_width = bg_cv2.shape[:2]
    canvas_glasses = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
    canvas_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
    
    # 7. Calculate placement (centering the glasses on the nose bridge)
    # We shift the Y slightly up so the bridge of the glasses sits exactly on the nose bridge
    top_left_x = nose_bridge[0] - (target_width // 2)
    top_left_y = nose_bridge[1] - (target_height // 2) + int(target_height * 0.05) 
    
    # Calculate bounding box for pasting, making sure we don't go out of bounds
    y1, y2 = max(0, top_left_y), min(bg_height, top_left_y + target_height)
    x1, x2 = max(0, top_left_x), min(bg_width, top_left_x + target_width)
    
    g_y1 = max(0, -top_left_y)
    g_y2 = g_y1 + (y2 - y1)
    g_x1 = max(0, -top_left_x)
    g_x2 = g_x1 + (x2 - x1)
    
    # Paste the resized glasses onto the canvas
    if y1 < y2 and x1 < x2:
        canvas_glasses[y1:y2, x1:x2] = glasses_resized[g_y1:g_y2, g_x1:g_x2]
        canvas_mask[y1:y2, x1:x2] = mask_resized[g_y1:g_y2, g_x1:g_x2]
        
    # 8. Apply rotation (to align with tilted faces)
    if abs(rot_angle_deg) > 1.0:
        center = (nose_bridge[0], nose_bridge[1])
        M = cv2.getRotationMatrix2D(center, rot_angle_deg, 1.0)
        canvas_glasses = cv2.warpAffine(canvas_glasses, M, (bg_width, bg_height), flags=cv2.INTER_LINEAR)
        canvas_mask = cv2.warpAffine(canvas_mask, M, (bg_width, bg_height), flags=cv2.INTER_NEAREST)
        
    # Convert back to PIL
    final_glasses_pil = Image.fromarray(cv2.cvtColor(canvas_glasses, cv2.COLOR_BGR2RGB))
    final_mask_pil = Image.fromarray(canvas_mask)
    
    return final_glasses_pil, final_mask_pil