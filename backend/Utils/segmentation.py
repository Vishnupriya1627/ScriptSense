import cv2
import numpy as np
import os
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import shutil

diagram_count = 0
text_count = 0

def correct_tilt(image):
    """Corrects tilt in an image using Hough Line Transform."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def extract_diagram(img_path, output_folder):
    global diagram_count
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found or unreadable.")

        gray_no_text = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_no_text = cv2.threshold(gray_no_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_no_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, cnt in enumerate(contours[::-1]):
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 40 and h > 400:
                diagram_img = img[y:y+h, x:x+w]
                print(f"Diagram {diagram_count} detected and saved.")
                cv2.imwrite(f"{output_folder}/diagram_{diagram_count}.png", diagram_img)
                diagram_count += 1
                return 0

        print(f"No valid diagram contours found in {img_path}")

    except Exception as e:
        print(f"[Error] Diagram extraction failed: {str(e)}")
    return None

def visualize_text_region(img, filtered_line_segments, output_path):
    """Removes the text region by replacing it with a white background."""
    try:
        global text_count
        if not os.path.exists('output/texts'):
            os.makedirs('output/texts')
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]

        if filtered_line_segments:
            min_y = min([seg[0] for seg in filtered_line_segments])
            max_y = max([seg[1] for seg in filtered_line_segments])
            
            text_crop = img_copy[max(min_y - 40, 0):min(max_y + 40, img_h), :]
            cv2.imwrite(f"output/texts/text_{text_count}.png", text_crop)
            cv2.imwrite(f"output/text.png", text_crop)
            print(f"Text region {text_count} cropped and saved at 'output/texts/text_{text_count}.png'")
            text_count += 1

            img_copy[max(min_y - 40, 0):min(max_y + 40, img_h), :] = (255, 255, 255)
            cv2.imwrite(output_path, img_copy)
            print(f"Text removed image saved at {output_path}")
        else:
            print("[Warning] No text segments found to visualize.")

    except Exception as e:
        print(f"[Error] Text region visualization failed: {str(e)}")

def segment_lines_and_find_diagrams(img, output_folder="output", min_height_threshold=30, padding=10, min_contour_width=5000):
    try:
        segmented_folder = os.path.join(output_folder, "segmented_lines")
        diagram_folder = os.path.join(output_folder, "diagrams")

        if os.path.exists(segmented_folder):
            shutil.rmtree(segmented_folder)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(f"{output_folder}/original_image.png", img)

        img = correct_tilt(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 10)

        hist = np.sum(binary, axis=1)
        print(hist)
        threshold = 50000 #np.max(hist) * 0.2
        print(threshold)
        lines = np.where(hist > threshold)[0]

        if len(lines) == 0:
            raise ValueError("No horizontal text lines detected.")

        line_segments = []
        start = lines[0]
        for i in range(1, len(lines)):
            if lines[i] - lines[i - 1] > 10:
                line_height = lines[i - 1] - start
                if line_height >= min_height_threshold:
                    line_segments.append((start, lines[i - 1]))
                start = lines[i]

        if lines[-1] - start >= min_height_threshold:
            line_segments.append((start, lines[-1]))


        os.makedirs(segmented_folder, exist_ok=True)
        os.makedirs(diagram_folder, exist_ok=True)

        img_height, img_width = img.shape[:2]
        filtered_line_segments = []

        for y_start, y_end in line_segments:
            line_segment = binary[y_start:y_end, :]
            left_whitespace = np.sum(line_segment[:, :20] == 0) / (20 * line_segment.shape[0])
            right_whitespace = np.sum(line_segment[:, -20:] == 0) / (20 * line_segment.shape[0])

            if left_whitespace > 0.90 and right_whitespace < 0.96:
                filtered_line_segments.append((y_start, y_end))

        for idx, (y_start, y_end) in enumerate(filtered_line_segments):
            y_start_pad = max(0, y_start - 40)
            y_end_pad = min(img_height, y_end + padding)
            line_img = img[y_start_pad:y_end_pad, :]
            cv2.imwrite(f"{segmented_folder}/line_{idx+1}.png", line_img)

        print(f"Segmented {len(filtered_line_segments)} text lines and saved in '{segmented_folder}'.")

        visualize_text_region(img, filtered_line_segments, os.path.join(output_folder, "text_bounding_box.png"))
        diagram_folder = extract_diagram(os.path.join(output_folder, "text_bounding_box.png"), diagram_folder)

        return segmented_folder, diagram_folder

    except Exception as e:
        print(f"[Error] Line segmentation and diagram detection failed: {str(e)}")
        return None, None

