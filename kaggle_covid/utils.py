import json
import os
from copy import deepcopy
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers import apply_voi_lut


def load_image(
    path_file: str,
    meta: dict,
    spacing: float = 1.0,
    percentile: bool = True,
) -> Tuple[Optional[np.ndarray], dict]:
    dicom = pydicom.dcmread(path_file)
    try:
        img = apply_voi_lut(dicom.pixel_array, dicom)
    except RuntimeError as err:
        print(err)
        return None, meta

    meta.update({
        'boxes': deepcopy(meta.get("boxes")) or [],
        'body': getattr(dicom, "BodyPartExamined"),
        'interpret': getattr(dicom, "PhotometricInterpretation"),
        'original_spacing': getattr(dicom, "ImagerPixelSpacing"),
        'original_image_shape': img.shape,
        'spacing': getattr(dicom, "ImagerPixelSpacing"),
        'image_shape': img.shape,
    })
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        img = img.max() - img
    p_low = np.percentile(img, 1) if percentile else img.min()
    p_high = np.percentile(img, 99) if percentile else img.max()
    # normalize
    img = (img.astype(float) - p_low) / (p_high - p_low)
    if spacing:
        factor = np.array(meta['spacing']) / spacing
        dims = tuple((np.array(img.shape[::-1]) * factor).astype(int))
        img = cv2.resize(img, dsize=dims, interpolation=cv2.INTER_LINEAR)
        for bbox in meta["boxes"]:
            bbox['x'] *= factor[0]
            bbox['y'] *= factor[1]
            bbox['width'] *= factor[0]
            bbox['height'] *= factor[1]
        meta.update({'spacing': (spacing, spacing), 'image_shape': img.shape})
    return img, meta


def convert_boxes_to_coco(meta: dict, image_hw: Tuple[int, int]) -> List[dict]:
    # ih, iw = img.shape[:2]
    ih, iw = image_hw
    bboxes = []
    for bbox in meta.get("boxes", []):
        # cls, x_center, y_center, width, height
        bboxes.append({
            "cls": meta["class"],
            "x_center": float(bbox['x'] + bbox['width'] / 2.) / iw,
            "y_center": float(bbox['y'] + bbox['height'] / 2.) / ih,
            "width": float(bbox['width']) / iw,
            "height": float(bbox['height']) / ih
        })
    return bboxes


def convert_image_dicom_to_coco(
    id_row: Tuple[int, pd.Series], spacing: float, dataset_path: str, path_images: str, path_labels: str
) -> dict:
    _, row = id_row
    # phase = "train" if np.random.random() < 0.8 else "valid"
    img, meta = load_image(os.path.join(dataset_path, row['path']), dict(row), spacing=spacing)
    plt.imsave(os.path.join(path_images, f"{row['name']}.jpg"), img, cmap='gray')
    bboxes = convert_boxes_to_coco(meta, image_hw=img.shape[:2])
    df = pd.DataFrame(bboxes)[["cls", "x_center", "y_center", "width", "height"]] if bboxes else pd.DataFrame(bboxes)
    df.to_csv(os.path.join(path_labels, f"{row['name']}.txt"), sep=" ", index=None, header=None)
    meta.update({"bboxes": bboxes})
    return meta


def generate_coco_annotations(metas: List[dict], labels: List[str], path_json: str) -> None:
    annots = []
    running_id = 0
    for idx, meta in enumerate(metas):
        ih, iw = meta["image_shape"]
        for i, box in enumerate(meta["bboxes"]):
            w = int(box['width'] * iw)
            h = int(box['height'] * ih)
            x = int(box['x_center'] * iw) - np.ceil(w / 2.)
            y = int(box['y_center'] * ih) - np.ceil(h / 2.)
            rec = {
                "id": running_id,
                "image_id": idx,
                "category_id": meta['class'],
                "area": w * h,
                "bbox": [max(0, x), max(0, y), w, h],
                "segmentation": [],
                "iscrowd": 0,
            }
            annots.append(rec)
            running_id += 1

    coco = {
        "annotations": annots,
        "categories": [{
            "id": i,
            "name": n,
            "supercategory": ""
        } for i, n in enumerate(labels)],
        "images": [{
            "id": idx,
            "file_name": f"{meta['name']}.jpg",
            "height": meta["image_shape"][0],
            "width": meta["image_shape"][1]
        } for idx, meta in enumerate(metas)],
    }

    with open(path_json, "w") as fp:
        json.dump(coco, fp)
