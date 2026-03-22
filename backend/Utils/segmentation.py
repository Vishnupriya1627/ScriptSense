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


def run_unimer(formula_crops_dir, all_line_paths, output_dir, llm=None, sampling_params=None):
    """Run Uni-MuMER - uses preloaded model if available."""
    try:
        if not all_line_paths:
            return {}

        if llm is not None:
            # Use preloaded model directly
            from Utils.formula_recognizer import recognize_formulas
            return recognize_formulas(all_line_paths, llm, sampling_params)
        
        # ── Fallback: subprocess ───────────────────────────────────
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

        with open(os.path.join(formula_crops_dir, "formulas.json"), "w") as f:
            json.dump(prompt, f)

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
        return {os.path.basename(all_line_paths[i]): p.get('pred', '') for i, p in enumerate(preds)}

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


def segment_lines_and_find_diagrams(img, output_folder="output", min_height_threshold=15, padding=5, min_contour_width=1000, llm=None, sampling_params=None):
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

        # ── Horizontal projection line detection ──────────────────────
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_sums = np.sum(binary, axis=1)
        row_sums_smooth = np.convolve(row_sums, np.ones(5)/5, mode='same')
        threshold = np.max(row_sums_smooth) * 0.01
        has_ink = row_sums_smooth > threshold

        lines, in_line, start = [], False, 0
        for i, ink in enumerate(has_ink):
            if ink and not in_line:
                start = i
                in_line = True
            elif not ink and in_line:
                in_line = False
                y1 = max(0, start - padding)
                y2 = min(img.shape[0], i + padding)
                if y2 - y1 > min_height_threshold:
                    lines.append((y1, y2))
        if in_line:
            lines.append((max(0, start - padding), img.shape[0]))

        # Merge overlapping lines
        merged_lines = []
        for y1, y2 in lines:
            if merged_lines and (y1 - merged_lines[-1][1]) < 2:
                merged_lines[-1] = (merged_lines[-1][0], y2)
            else:
                merged_lines.append((y1, y2))

        print(f"✅ Detected {len(merged_lines)} line(s)")

        # ── Save all lines as both text and formula crops ──────────────
        all_line_paths = []
        for idx, (y1, y2) in enumerate(merged_lines):
            crop = img[y1:y2, :]
            # Trim horizontal whitespace
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, bin_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_sums = np.sum(bin_crop, axis=0)
            nonzero = np.where(col_sums > 0)[0]
            if len(nonzero) > 0:
                x1 = max(0, nonzero[0] - 10)
                x2 = min(crop.shape[1], nonzero[-1] + 10)
                crop = crop[:, x1:x2]

            line_path = os.path.join(segmented_folder, f"line_{idx:03d}.png")
            cv2.imwrite(line_path, crop)

            # Save to both folders
            cv2.imwrite(os.path.join(texts_folder, f"text_{idx:03d}.png"), crop)
            cv2.imwrite(os.path.join(formulas_folder, f"formula_{idx:03d}.png"), crop)
            all_line_paths.append(line_path)

        print(f"✅ Saved {len(all_line_paths)} line crops")

        # ── Run Uni-MuMER on ALL lines ─────────────────────────────────
        latex_results = {}
        if llm is not None and all_line_paths:
            latex_results = run_unimer(
                formulas_folder, all_line_paths, formulas_output,
                llm=llm, sampling_params=sampling_params
            )
            with open(os.path.join(output_folder, "formulas_latex.json"), "w") as f:
                json.dump(latex_results, f)
            print(f"✅ Uni-MuMER results saved: {len(latex_results)} lines")

        # ── Diagram extraction (from full image minus text rows) ───────
        text_bounding_box_path = os.path.join(output_folder, "text_bounding_box.png")
        visualize_text_region(img, merged_lines, text_bounding_box_path, texts_folder)
        diagram_count = extract_diagram(text_bounding_box_path, diagrams_folder)

        return len(merged_lines), diagram_count if diagram_count else 0

    except Exception as e:
        print(f"[Error] Segmentation failed: {e}")
        traceback.print_exc()
        return 0, 0