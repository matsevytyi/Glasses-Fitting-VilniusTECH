import os
import cv2
import clip
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# import warnings
# warnings.filterwarnings('ignore')


def convert_box_xywh_to_xyxy(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[0] + box[2])
    y2 = int(box[1] + box[3])
    return [x1, y1, x2, y2]


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)

    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def calculate_centroid(mask):
    mask_np = np.array(mask)
    y_indices, x_indices = np.nonzero(mask_np)
    if len(x_indices) == 0:
        return None
    return (float(np.mean(x_indices)), float(np.mean(y_indices)))


def calculate_distance(point1, point2):
    return float(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def get_indices_of_values_above_threshold(scores, threshold):
    idx = torch.nonzero(scores > threshold).squeeze()
    if idx.numel() == 0:
        return []
    if idx.ndim == 0:
        return [int(idx.item())]
    return idx.cpu().tolist()


def initialize_sam_clip(
    sam_checkpoint,
    sam_model_type="vit_l", # curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -o sam_vit_l_0b3195.pth
    sam_device=None,
    clip_device=None,
    clip_model_name="ViT-B/32",
):
    if sam_device is None:
        if torch.cuda.is_available():
            sam_device = "cuda"
        else:
            sam_device = "cpu"

    if clip_device is None:
        if torch.cuda.is_available():
            clip_device = "cuda"
        elif torch.backends.mps.is_available():
            clip_device = "mps"
        else:
            clip_device = "cpu"

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=sam_device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    clip_model, preprocess = clip.load(clip_model_name, device=clip_device)

    return {
        "sam": sam,
        "mask_generator": mask_generator,
        "clip_model": clip_model,
        "preprocess": preprocess,
        "sam_device": sam_device,
        "clip_device": clip_device,
    }


@torch.no_grad()
def retriev(elements, positive_texts, negative_texts, clip_model, preprocess, device):
    if len(elements) == 0:
        return torch.empty(0, device=device)

    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_positive_texts = clip.tokenize(positive_texts).to(device)
    tokenized_negative_texts = clip.tokenize(negative_texts).to(device)

    stacked_images = torch.stack(preprocessed_images)
    image_features = clip_model.encode_image(stacked_images)
    positive_text_features = clip_model.encode_text(tokenized_positive_texts)
    negative_text_features = clip_model.encode_text(tokenized_negative_texts)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    positive_text_features /= positive_text_features.norm(dim=-1, keepdim=True)
    negative_text_features /= negative_text_features.norm(dim=-1, keepdim=True)

    positive_probs = 100.0 * image_features @ positive_text_features.T
    negative_probs = 100.0 * image_features @ negative_text_features.T

    avg_positive_probs = positive_probs.mean(dim=1)
    avg_negative_probs = negative_probs.mean(dim=1)
    final_probs = avg_positive_probs - avg_negative_probs

    return final_probs.softmax(dim=0)


@torch.no_grad()
def retriev_back(elements, search_text, clip_model, preprocess, device):
    if len(elements) == 0:
        return torch.empty(0, device=device)

    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)

    stacked_images = torch.stack(preprocessed_images)
    image_features = clip_model.encode_image(stacked_images)
    text_features = clip_model.encode_text(tokenized_text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def process_glasses_with_sam_and_clip_standalone(
    image_path,
    sam_clip_models,
    save_result=False,
    destination_path=None,
    positive_texts=("side arms", "temple arms"),
    negative_texts=("top bar",),
    threshold=0.05,
    bridge_text="top bridge",
):
    mask_generator = sam_clip_models["mask_generator"]
    clip_model = sam_clip_models["clip_model"]
    preprocess = sam_clip_models["preprocess"]
    clip_device = sam_clip_models["clip_device"]

    image_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_cv is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if image_cv.ndim == 2:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)
    elif image_cv.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGB)
    else:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)
    image_pil = Image.open(image_path).convert("RGB")

    cropped_boxes = []
    valid_masks = []

    for mask in masks:
        
        image_width, image_height = image_pil.size
        max_allowed_area = image_width * image_height * 0.15
        
        if mask["area"] > max_allowed_area:
            continue
        
        seg = mask["segmentation"]
        bbox_xyxy = convert_box_xywh_to_xyxy(mask["bbox"])
        crop = segment_image(image_pil, seg).crop(bbox_xyxy)
        cropped_boxes.append(crop)
        valid_masks.append(mask)

    if len(cropped_boxes) == 0:
        raise RuntimeError("SAM produced no masks.")

    scores = retriev(
        cropped_boxes,
        list(positive_texts),
        list(negative_texts),
        clip_model,
        preprocess,
        clip_device,
    )
    indices = get_indices_of_values_above_threshold(scores, threshold)

    if len(indices) == 0:
        indices = [int(torch.argmax(scores).item())]

    segmentation_masks = []
    for seg_idx in indices:
        seg_mask_img = Image.fromarray(valid_masks[seg_idx]["segmentation"].astype("uint8") * 255)
        segmentation_masks.append(seg_mask_img)

    combined_mask = Image.new("L", image_pil.size, 0)
    for mask_img in segmentation_masks:
        combined_mask.paste(mask_img, (0, 0), mask_img)

    image_rgba = image_pil.convert("RGBA")
    np_image = np.array(image_rgba)
    np_mask = np.array(combined_mask)

    np_mask = cv2.dilate(np_mask, np.ones((7, 7), np.uint8), iterations=1)
    np_image[:, :, 3] = np.where(np_mask == 255, 0, np_image[:, :, 3])
    np_image = cv2.erode(np_image, np.ones((3, 3), np.uint8), iterations=1)
    result_image = Image.fromarray(np_image)

    bridge_scores = retriev_back(
        cropped_boxes,
        bridge_text,
        clip_model,
        preprocess,
        clip_device,
    )
    bridge_indices = get_indices_of_values_above_threshold(bridge_scores, threshold)

    if len(bridge_indices) == 0:
        bridge_indices = [int(torch.argmax(bridge_scores).item())]

    bridge_masks = []
    for seg_idx in bridge_indices:
        seg_mask_img = Image.fromarray(valid_masks[seg_idx]["segmentation"].astype("uint8") * 255)
        bridge_masks.append(seg_mask_img)

    image_width, image_height = image_pil.size
    image_center = (image_width / 2.0, image_height / 2.0)

    min_distance = float("inf")
    most_central_mask = None

    for mask_img in bridge_masks:
        centroid = calculate_centroid(mask_img)
        if centroid is None:
            continue
        distance_to_center = calculate_distance(centroid, image_center)
        if distance_to_center < min_distance:
            min_distance = distance_to_center
            most_central_mask = mask_img

    lower_point = None
    higher_point = None
    bridge_width = None

    if most_central_mask is not None:
        mask_np = np.array(most_central_mask)
        coords = np.argwhere(mask_np > 0)

        if len(coords) > 0:
            mask_center_x = int(calculate_centroid(most_central_mask)[0])

            ys_for_center = np.argwhere(mask_np[:, mask_center_x] > 0)
            if len(ys_for_center) > 0:
                lower_y = int(np.min(ys_for_center))
            else:
                lower_y = int(np.min(coords[:, 0]))

            lower_point = (mask_center_x, lower_y)

            original_np = np.array(image_pil)
            upper_column = original_np[:lower_y, mask_center_x]

            if upper_column.ndim == 2:
                non_transparent_indices = np.argwhere(upper_column == 0)
            else:
                non_transparent_indices = np.argwhere(np.all(upper_column == 0, axis=1))

            if non_transparent_indices.size > 0:
                higher_y = int(np.max(non_transparent_indices))
                higher_point = (mask_center_x, higher_y)
            else:
                higher_point = (mask_center_x, max(0, lower_y - 5))

            bridge_width = int(np.max(coords[:, 0]) - np.min(coords[:, 0]))

    if save_result and destination_path is not None:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        result_image.save(destination_path)

    return {
        "result_image_rgba": result_image,
        "combined_mask": combined_mask,
        "lower_point": lower_point,
        "higher_point": higher_point,
        "bridge_width": bridge_width,
        "selected_indices": indices,
        "bridge_indices": bridge_indices,
        "sam_masks_raw": valid_masks,
    }