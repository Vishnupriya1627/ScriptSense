import cv2
import numpy as np
import os
import shutil
import json
import subprocess

def correct_tilt(image):
    """Corrects tilt in an image using Hough Line Transform."""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

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
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                return corrected_image
    except Exception as e:
        print(f"[Error] Tilt correction failed: {str(e)}")
    return image


def is_formula_region(line_img):
    """Heuristic: detect if a line crop looks like a math formula."""
    try:
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img

        h, w = gray.shape
        if h == 0 or w == 0:
            return False

        # Too tall and narrow = likely not a formula
        if w / h < 1.2:  # lowered from 1.5
            return False

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        row_sums = np.sum(binary, axis=1) > 0
        ink_spread = np.sum(row_sums) / h

        col_sums = np.sum(binary, axis=0) > 0
        transitions = np.diff(col_sums.astype(int))
        gap_count = np.sum(transitions == -1)

        ink_density = np.sum(binary > 0) / (h * w)

        return (
            0.2 < ink_spread < 0.95 and  # wider range
            gap_count >= 2 and            # lowered from 3
            0.01 < ink_density < 0.5      # wider range
        )
    except Exception as e:
        print(f"[Warning] Formula heuristic failed: {e}")
        return False


def run_unimer(formula_crops_dir, output_dir,
               unimer_dir="/content/Uni-MuMER",
               unimer_model="/content/Uni-MuMER/models/Uni-MuMER-3B"):
    """Run Uni-MuMER inference on all formula crops in a folder."""
    try:
        formula_files = sorted([
            f for f in os.listdir(formula_crops_dir) if f.endswith('.png')
        ])

        if not formula_files:
            print("[Info] No formula crops to process")
            return {}

        # Build JSON prompt
        prompt = []
        for ff in formula_files:
            fp = os.path.abspath(os.path.join(formula_crops_dir, ff))
            prompt.append({
                "images": [fp],
                "messages": [
                    {
                        "from": "human",
                        "value": "<image>I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format."
                    },
                    {"from": "gpt", "value": ""}
                ]
            })

        os.makedirs(output_dir, exist_ok=True)
        prompt_path = os.path.join(formula_crops_dir, "formulas.json")
        with open(prompt_path, "w") as f:
            json.dump(prompt, f)

        print(f"   ➕ Running Uni-MuMER on {len(formula_files)} formula(s)...")

        result = subprocess.run(
            [
                "python", "scripts/vllm_infer.py",
                "--input-dir", os.path.abspath(formula_crops_dir),
                "--output-dir", os.path.abspath(output_dir),
                "--model", unimer_model,
                "--batch-size", "4"
            ],
            cwd=unimer_dir,
            capture_output=True,
            text=True
        )

        # Find prediction file
        pred_files = [f for f in os.listdir(output_dir) if f.endswith('_pred.json')]
        if not pred_files:
            print(f"   ⚠️ No prediction file found. stderr: {result.stderr[-300:]}")
            return {}

        preds = json.load(open(os.path.join(output_dir, pred_files[0])))
        latex_map = {}
        for i, p in enumerate(preds):
            fname = formula_files[i] if i < len(formula_files) else f"formula_{i:03d}.png"
            latex_map[fname] = p.get('pred', '')
            print(f"   ✅ {fname} → {p.get('pred', '')}")

        return latex_map

    except Exception as e:
        print(f"[Error] Uni-MuMER inference failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def extract_diagram(img_path, output_folder):
    """Extract diagrams from image after text removal."""
    try:
        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}")
            return None
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Could not read image: {img_path}")
            return None

        gray_no_text = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_no_text = cv2.threshold(gray_no_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_no_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diagram_count = 0
        for idx, cnt in enumerate(contours[::-1]):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > 5000 and w > 50 and h > 50:
                diagram_img = img[y:y+h, x:x+w]
                output_path = os.path.join(output_folder, f"diagram_{diagram_count}.png")
                cv2.imwrite(output_path, diagram_img)
                print(f"✅ Diagram {diagram_count} detected and saved: {w}x{h} pixels")
                diagram_count += 1

        if diagram_count == 0:
            print(f"[Info] No diagrams found in {img_path}")
            
        return diagram_count

    except Exception as e:
        print(f"[Error] Diagram extraction failed: {str(e)}")
    return None


def visualize_text_region(img, filtered_line_segments, output_path, output_texts_folder):
    """Extract text regions and remove them from image."""
    try:
        os.makedirs(output_texts_folder, exist_ok=True)
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]

        if filtered_line_segments:
            min_y = min([seg[0] for seg in filtered_line_segments])
            max_y = max([seg[1] for seg in filtered_line_segments])
            
            y_start = max(min_y - 20, 0)
            y_end = min(max_y + 20, img_h)
            text_crop = img_copy[y_start:y_end, :]
            
            text_path = os.path.join(output_texts_folder, f"text_0.png")
            cv2.imwrite(text_path, text_crop)
            print(f"✅ Text region cropped and saved: {text_path}")

            img_copy[y_start:y_end, :] = (255, 255, 255)
            cv2.imwrite(output_path, img_copy)
            print(f"✅ Text removed image saved: {output_path}")
            
            return True
        else:
            print("[Warning] No text segments found to visualize.")
            cv2.imwrite(output_path, img_copy)
            return False

    except Exception as e:
        print(f"[Error] Text region visualization failed: {str(e)}")
        return False


def segment_lines_and_find_diagrams(img, output_folder="output", min_height_threshold=15, padding=5, min_contour_width=1000):
    try:
        segmented_folder = os.path.join(output_folder, "segmented_lines")
        texts_folder = os.path.join(output_folder, "texts")
        diagrams_folder = os.path.join(output_folder, "diagrams")
        formulas_folder = os.path.join(output_folder, "formulas")
        formulas_output = os.path.join(output_folder, "formulas_output")

        for folder in [segmented_folder, texts_folder, diagrams_folder, formulas_folder, formulas_output]:
            os.makedirs(folder, exist_ok=True)

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{output_folder}/original_image.png", img)
        img = correct_tilt(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ── Horizontal projection to find line boundaries ──────────────
        row_sums = np.sum(binary, axis=1)
        row_sums_smooth = np.convolve(row_sums, np.ones(5)/5, mode='same')
        ink_threshold = np.max(row_sums_smooth) * 0.02
        has_ink = row_sums_smooth > ink_threshold

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

        # Merge lines that are very close (superscripts cause false splits)
        merged_lines = []
        for y1, y2 in lines:
            if merged_lines and (y1 - merged_lines[-1][1]) < 20:
                merged_lines[-1] = (merged_lines[-1][0], y2)
            else:
                merged_lines.append((y1, y2))

        print(f"✅ Detected {len(merged_lines)} line(s) via projection")

        # ── Classify each line as formula or text ──────────────────────
        formula_count = 0
        text_line_segments = []

        for idx, (y1, y2) in enumerate(merged_lines):
            crop = img[y1:y2, :]

            # Trim horizontal whitespace
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, bin_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_sums = np.sum(bin_crop, axis=0)
            nonzero_cols = np.where(col_sums > 0)[0]
            if len(nonzero_cols) > 0:
                x1 = max(0, nonzero_cols[0] - 10)
                x2 = min(crop.shape[1], nonzero_cols[-1] + 10)
                crop = crop[:, x1:x2]

            cv2.imwrite(f"{segmented_folder}/line_{idx+1}.png", crop)

            if is_formula_region(crop):
                formula_path = os.path.join(formulas_folder, f"formula_{formula_count:03d}.png")
                cv2.imwrite(formula_path, crop)
                print(f"   ➕ Line {idx+1} → FORMULA")
                formula_count += 1
            else:
                text_line_segments.append((y1, y2))
                print(f"   📝 Line {idx+1} → TEXT")

        print(f"✅ {formula_count} formula(s), {len(text_line_segments)} text line(s)")

        # ── Text extraction ────────────────────────────────────────────
        text_bounding_box_path = os.path.join(output_folder, "text_bounding_box.png")
        visualize_text_region(img, text_line_segments, text_bounding_box_path, texts_folder)

        # ── Diagram extraction ─────────────────────────────────────────
        diagram_count = extract_diagram(text_bounding_box_path, diagrams_folder)

        # ── Run Uni-MuMER on formula crops ─────────────────────────────
        latex_results = {}
        if formula_count > 0:
            latex_results = run_unimer(formulas_folder, formulas_output)
            latex_out_path = os.path.join(output_folder, "formulas_latex.json")
            with open(latex_out_path, "w") as f:
                json.dump(latex_results, f)
            print(f"✅ LaTeX saved → {latex_out_path}")

        return len(merged_lines), diagram_count if diagram_count else 0

    except Exception as e:
        print(f"[Error] Segmentation failed: {str(e)}")
        traceback.print_exc()
        return 0, 0