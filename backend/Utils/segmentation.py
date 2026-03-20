import cv2
import numpy as np
import os
import json
import subprocess
import traceback

UNIMER_DIR   = "/content/Uni-MuMER"
UNIMER_MODEL = "/content/Uni-MuMER/models/Uni-MuMER-3B"
MFD_MODEL    = "/content/PDF-Extract-Kit/models/MFD/YOLO/yolo_v8_ft.pt"

def correct_tilt(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                if 80 < angle < 100:
                    angles.append(90 - angle)
            if angles:
                avg_angle = np.mean(angles)
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), avg_angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"[Error] Tilt correction failed: {e}")
    return image


def split_formula_region_into_lines(img_crop, output_folder, padding=8):
    """Split a detected formula region into individual equation lines."""
    os.makedirs(output_folder, exist_ok=True)

    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_sums = np.sum(binary, axis=1)
    row_sums_smooth = np.convolve(row_sums, np.ones(5)/5, mode='same')
    threshold = np.max(row_sums_smooth) * 0.01  # 1% threshold
    has_ink = row_sums_smooth > threshold

    lines, in_line, start = [], False, 0
    for i, ink in enumerate(has_ink):
        if ink and not in_line:
            start = i
            in_line = True
        elif not ink and in_line:
            in_line = False
            y1 = max(0, start - padding)
            y2 = min(img_crop.shape[0], i + padding)
            if y2 - y1 > 15:
                lines.append((y1, y2))
    if in_line:
        lines.append((max(0, start - padding), img_crop.shape[0]))

    # Merge overlapping lines (superscripts cause false splits)
    merged = []
    for y1, y2 in lines:
        if merged and (y1 - merged[-1][1]) < 2:
            merged[-1] = (merged[-1][0], y2)
        else:
            merged.append((y1, y2))

    saved = []
    for idx, (y1, y2) in enumerate(merged):
        line_crop = img_crop[y1:y2, :]
        # Trim horizontal whitespace
        gray_line = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
        _, bin_line = cv2.threshold(gray_line, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        col_sums = np.sum(bin_line, axis=0)
        nonzero = np.where(col_sums > 0)[0]
        if len(nonzero) > 0:
            x1 = max(0, nonzero[0] - 10)
            x2 = min(line_crop.shape[1], nonzero[-1] + 10)
            line_crop = line_crop[:, x1:x2]
        out_path = os.path.join(output_folder, f"line_{idx:03d}.png")
        cv2.imwrite(out_path, line_crop)
        saved.append(out_path)

    return saved


def detect_formula_regions(img_bgr, output_folder):
    """Use YOLO MFD model to detect formula regions in a page image."""
    try:
        from ultralytics import YOLO
        model = YOLO(MFD_MODEL)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, conf=0.25, iou=0.45, imgsz=1280, verbose=False)
        boxes = results[0].boxes

        all_line_crops = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Expand bbox to catch nearby lines
            x1 = max(0, x1 - 20)
            y1 = max(0, y1 - 20)
            x2 = min(img_bgr.shape[1], x2 + 20)
            y2 = min(img_bgr.shape[0], y2 + 100)

            crop = img_bgr[y1:y2, x1:x2]
            region_folder = os.path.join(output_folder, f"region_{i}")
            lines = split_formula_region_into_lines(crop, region_folder)
            all_line_crops.extend(lines)
            print(f"   ➕ Formula region {i+1}: {len(lines)} line(s)")

        print(f"✅ Total formula line crops: {len(all_line_crops)}")
        return all_line_crops

    except Exception as e:
        print(f"[Error] Formula detection failed: {e}")
        traceback.print_exc()
        return []


def run_unimer(formula_crops_dir, all_line_paths, output_dir):
    """Run Uni-MuMER on detected formula line crops."""
    try:
        if not all_line_paths:
            return {}

        prompt = [{
            "images": [os.path.abspath(p)],
            "messages": [
                {
                    "from": "human",
                    "value": "<image>I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format."
                },
                {"from": "gpt", "value": ""}
            ]
        } for p in all_line_paths]

        os.makedirs(formula_crops_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        prompt_path = os.path.join(formula_crops_dir, "formulas.json")
        with open(prompt_path, "w") as f:
            json.dump(prompt, f)

        print(f"   ➕ Running Uni-MuMER on {len(all_line_paths)} line(s)...")

        result = subprocess.run(
            [
                "python", "scripts/vllm_infer.py",
                "--input-dir", os.path.abspath(formula_crops_dir),
                "--output-dir", os.path.abspath(output_dir),
                "--model", UNIMER_MODEL,
                "--batch-size", "4"
            ],
            cwd=UNIMER_DIR,
            capture_output=True,
            text=True
        )

        pred_files = [f for f in os.listdir(output_dir) if f.endswith('_pred.json')]
        if not pred_files:
            print(f"   ⚠️ No prediction file. stderr: {result.stderr[-300:]}")
            return {}

        preds = json.load(open(os.path.join(output_dir, pred_files[0])))
        latex_map = {}
        for i, p in enumerate(preds):
            fname = os.path.basename(all_line_paths[i]) if i < len(all_line_paths) else f"line_{i}"
            latex_map[fname] = p.get('pred', '')
            print(f"   ✅ {fname} → {p.get('pred', '')}")

        return latex_map

    except Exception as e:
        print(f"[Error] Uni-MuMER failed: {e}")
        traceback.print_exc()
        return {}


def extract_diagram(img_path, output_folder):
    try:
        if not os.path.exists(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        diagram_count = 0
        for cnt in contours[::-1]:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 5000 and w > 50 and h > 50:
                cv2.imwrite(os.path.join(output_folder, f"diagram_{diagram_count}.png"), img[y:y+h, x:x+w])
                diagram_count += 1
        if diagram_count == 0:
            print(f"[Info] No diagrams found")
        return diagram_count
    except Exception as e:
        print(f"[Error] Diagram extraction failed: {e}")
        return None


def visualize_text_region(img, filtered_line_segments, output_path, output_texts_folder):
    try:
        os.makedirs(output_texts_folder, exist_ok=True)
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        if filtered_line_segments:
            min_y = min([seg[0] for seg in filtered_line_segments])
            max_y = max([seg[1] for seg in filtered_line_segments])
            y_start = max(min_y - 20, 0)
            y_end = min(max_y + 20, img_h)
            cv2.imwrite(os.path.join(output_texts_folder, "text_0.png"), img_copy[y_start:y_end, :])
            img_copy[y_start:y_end, :] = (255, 255, 255)
            cv2.imwrite(output_path, img_copy)
            return True
        else:
            cv2.imwrite(output_path, img_copy)
            return False
    except Exception as e:
        print(f"[Error] Text region visualization failed: {e}")
        return False


def segment_lines_and_find_diagrams(img, output_folder="output", min_height_threshold=15, padding=5, min_contour_width=1000):
    try:
        texts_folder    = os.path.join(output_folder, "texts")
        diagrams_folder = os.path.join(output_folder, "diagrams")
        formulas_folder = os.path.join(output_folder, "formulas")
        formulas_output = os.path.join(output_folder, "formulas_output")
        segmented_folder= os.path.join(output_folder, "segmented_lines")

        for folder in [texts_folder, diagrams_folder, formulas_folder, formulas_output, segmented_folder]:
            os.makedirs(folder, exist_ok=True)

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{output_folder}/original_image.png", img)
        img = correct_tilt(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

        # ── Step 1: Detect formula regions with YOLO ──────────────────
        all_formula_lines = detect_formula_regions(img, formulas_folder)

        # ── Step 2: Get formula bounding boxes to exclude from text ───
        formula_row_mask = np.zeros(img.shape[0], dtype=bool)
        for line_path in all_formula_lines:
            # Mark rows covered by formula regions as formula rows
            pass  # handled by YOLO bbox exclusion below

        # ── Step 3: Text line detection (contour based) ────────────────
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 10), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > min_height_threshold and w > 30 and h < 150 and w/h < 20:
                text_contours.append((x, y, w, h))
        text_contours.sort(key=lambda r: r[1])

        merged_contours = []
        for contour in text_contours:
            if not merged_contours:
                merged_contours.append(list(contour))
                continue
            last = merged_contours[-1]
            x, y, w, h = contour
            y_overlap = max(0, min(last[1]+last[3], y+h) - max(last[1], y))
            if y_overlap > 0:
                last[0] = min(last[0], x)
                last[1] = min(last[1], y)
                last[2] = max(last[0]+last[2], x+w) - last[0]
                last[3] = max(last[1]+last[3], y+h) - last[1]
            else:
                merged_contours.append(list(contour))

        print(f"✅ Detected {len(merged_contours)} text line(s)")

        # ── Step 4: Save text lines (skip formula rows) ────────────────
        text_line_segments = []
        for idx, (x, y, w, h) in enumerate(merged_contours):
            x1 = max(0, x-padding); y1 = max(0, y-padding)
            x2 = min(img.shape[1], x+w+padding); y2 = min(img.shape[0], y+h+padding)
            cv2.imwrite(f"{segmented_folder}/line_{idx+1}.png", img[y1:y2, x1:x2])
            text_line_segments.append((y, y+h))

        # ── Step 5: Text region extraction ────────────────────────────
        text_bounding_box_path = os.path.join(output_folder, "text_bounding_box.png")
        visualize_text_region(img, text_line_segments, text_bounding_box_path, texts_folder)
        diagram_count = extract_diagram(text_bounding_box_path, diagrams_folder)

        # ── Step 6: Run Uni-MuMER on formula crops ─────────────────────
        latex_results = {}
        if all_formula_lines:
            latex_results = run_unimer(formulas_folder, all_formula_lines, formulas_output)
            with open(os.path.join(output_folder, "formulas_latex.json"), "w") as f:
                json.dump(latex_results, f)
            print(f"✅ LaTeX results saved")

        return len(merged_contours), diagram_count if diagram_count else 0

    except Exception as e:
        print(f"[Error] Segmentation failed: {e}")
        traceback.print_exc()
        return 0, 0