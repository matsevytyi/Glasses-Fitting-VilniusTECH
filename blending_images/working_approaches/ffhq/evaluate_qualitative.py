
#!pip install torch torchvision torchmetrics lpips ultralytics opencv-python numpy ipynb

import torch
import cv2
import numpy as np
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from ultralytics import SAM

# import warnings
# warnings.filterwarnings('ignore')

from segment_helper import process_glasses_with_sam_and_clip_standalone, initialize_sam_clip

loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False)

# FID expects uint8 tensors in range [0, 255]
#fid = FrechetInceptionDistance(feature=2048)

#model = SAM("blending_images/working_approaches/ffhq/sam2_l.pt") 

sam_clip_models = initialize_sam_clip(
    sam_checkpoint="sam_vit_l_0b3195.pth", 
    sam_model_type="vit_l"
)

# ==========================================
# 1. LPIPS Calculation (Single Image Pair)
# ==========================================
def calculate_lpips(img_path_1, img_path_2):
    #print("Loading LPIPS pre-trained VGG network...")
    # Automatically downloads pre-trained VGG weights
    
    # LPIPS expects tensors in range [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img1 = cv2.cvtColor(cv2.imread(img_path_1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img_path_2), cv2.COLOR_BGR2RGB)
    
    img1_t = transform(img1).unsqueeze(0)
    img2_t = transform(img2).unsqueeze(0)
    
    # Calculate distance (lower is better, meaning more similar)
    distance = loss_fn_vgg(img1_t, img2_t)
    return distance.item()

# ==========================================
# 2. FID Calculation (Directory Level)
# ==========================================
def calculate_fid(real_images_dir, generated_images_dir):
    #print("Initializing FID metric...")
    
    # Helper to load images from a directory into a batch tensor
    import os
    from PIL import Image
    
    def load_dir_to_tensor(directory):
        images = []
        for file in os.listdir(directory):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(directory, file)).convert("RGB")
                img = img.resize((299, 299)) # InceptionV3 standard size
                img_t = torch.tensor(np.array(img)).permute(2, 0, 1)
                images.append(img_t)
        return torch.stack(images)

    real_batch = load_dir_to_tensor(real_images_dir)
    gen_batch = load_dir_to_tensor(generated_images_dir)
    
    fid.update(real_batch, real=True)
    fid.update(gen_batch, real=False)
    
    return fid.compute().item()

# ==========================================
# 3. SAM 2 Segmentation & mIoU Evaluation
# ==========================================
# def calculate_sam2_metrics(harmonized_img_path, ground_truth_mask_path):
#     #print("Loading SAM 2 model...")
    
#     # 1. Load ground truth mask
#     gt_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
#     _, binary_gt = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
    
#     # Find bounding box for the prompt
#     contours, _ = cv2.findContours(binary_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     x, y, w, h = cv2.boundingRect(contours[0])
#     bbox = [x, y, x + w, y + h]
    
#     # 2. Predict mask
#     #print(f"Prompting SAM 2 with bbox {bbox}...")
#     results = model.predict(harmonized_img_path, bboxes=[bbox], verbose=False)
    
#     pred_mask = results[0].masks.data[0].cpu().numpy()
#     pred_mask = (pred_mask * 255).astype(np.uint8)
#     pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
#     # Normalize masks to [0, 1] range for accurate loss calculation
#     gt_norm = (binary_gt > 0).astype(np.float32)
#     pred_norm = (pred_mask > 0).astype(np.float32)
    
#     # 3. Calculate IoU
#     intersection = np.logical_and(gt_norm, pred_norm).sum()
#     union = np.logical_or(gt_norm, pred_norm).sum()
#     iou = intersection / union if union > 0 else 0.0
    
#     # 4. Calculate SAM Losses (MSE and Dice)
#     # MSE: average pixel-wise squared difference
#     mse_loss = np.mean((gt_norm - pred_norm) ** 2)
    
#     # Dice Loss: 1 - Dice Coefficient (better for heavily imbalanced masks like thin glasses frames)
#     dice_score = (2.0 * intersection) / (gt_norm.sum() + pred_norm.sum()) if (gt_norm.sum() + pred_norm.sum()) > 0 else 0.0
#     dice_loss = 1.0 - dice_score
    
#     return iou, mse_loss, dice_loss, pred_mask

def calculate_sam_clip_metrics(harmonized_img_path, ground_truth_mask_path):
    
    # 1. Load the ground truth mask
    gt_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_gt = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Run the SAM+CLIP Pipeline
    # We change the prompts to look for the full glasses, not just the arms!
    try:
        out = process_glasses_with_sam_and_clip_standalone(
            image_path=harmonized_img_path,
            sam_clip_models=sam_clip_models,
            positive_texts=["spectacle frames", "eyeglasses frame", "side arms", "temple arms"],
            negative_texts=["person", "face", "skin", "hair", "eyes", "nose"],
            save_result=False
        )
        
        # Extract the mask SAM+CLIP found
        pred_mask_pil = out["combined_mask"]
        pred_mask = np.array(pred_mask_pil)
        
    except Exception as e:
        print(f"SAM+CLIP pipeline failed: {e}")
        # If SAM finds absolutely nothing, return worst-case scores
        return 0.0, 1.0, 1.0, np.zeros_like(binary_gt)

    # 3. Resize and threshold the predicted mask to perfectly match GT dimensions
    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary_pred = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Normalize to [0, 1]
    gt_norm = (binary_gt > 0).astype(np.float32)
    pred_norm = (binary_pred > 0).astype(np.float32)
    
    # 4. Calculate IoU
    intersection = np.logical_and(gt_norm, pred_norm).sum()
    union = np.logical_or(gt_norm, pred_norm).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # 5. Calculate SAM Losses (MSE and Dice)
    mse_loss = np.mean((gt_norm - pred_norm) ** 2)
    dice_score = (2.0 * intersection) / (gt_norm.sum() + pred_norm.sum()) if (gt_norm.sum() + pred_norm.sum()) > 0 else 0.0
    dice_loss = 1.0 - dice_score
    
    return iou, mse_loss, dice_loss, binary_pred

#if __name__ == "main":
# 1. Run LPIPS (Compare Copy-Paste vs Harmonized)
# LPIPS scores > 0.3 mean noticeable difference, closer to 0 means identical
#lpips_score = calculate_lpips("temp_eval/naive_0x0.png", "results/traditional_blend/poisson/FFHQ_0x0.png")
#print(f"LPIPS Score: {lpips_score:.4f} (Lower is better)")

# 2. Run SAM 2 to see if the harmonization tricks the segmentation model
iou, mse, dice, sam_mask = calculate_sam_clip_metrics("temp_eval/naive_0x0.png", "../../frames_new/masks/110020220079_01.png")
print(f"SAM 2 IoU:        {iou:.4f} (Higher is better)")
print(f"SAM 2 MSE Loss:   {mse:.4f} (Lower is better)")
print(f"SAM 2 Dice Loss:  {dice:.4f} (Lower is better)")


# Save the mask to visually inspect what SAM 2 saw
cv2.imwrite("sam2_prediction_mask.png", sam_mask)

# 3. Run FID (when generated whole folder of images)
# fid_score = calculate_fid("./dataset/real_faces_with_glasses", "./dataset/diffusion_harmonized_faces")
# print(f"FID Score: {fid_score:.2f}")

# SAM 2 IoU:        0.1314 (Higher is better)
# SAM 2 MSE Loss:   0.1293 (Lower is better)
# SAM 2 Dice Loss:  0.7677 (Lower is better)
