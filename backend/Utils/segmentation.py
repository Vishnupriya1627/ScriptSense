import cv2
import numpy as np
import os
from pdf2image import convert_from_path

diagram_count = 0
def correct_tilt(image):
    """Corrects tilt in an image using Hough Line Transform."""
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
    return image

def extract_diagram(img_path, output_folder):
    global diagram_count
    img = cv2.imread(img_path)

    gray_no_text = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_no_text = cv2.threshold(gray_no_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (remaining objects, likely diagrams)
    contours, _ = cv2.findContours(binary_no_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract diagrams

    for idx, cnt in enumerate(contours[::-1]):
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore small noise
        if w > 40 and h > 400:
            diagram_img = img[y:y+h, x:x+w]
            cv2.imwrite(f"{output_folder}/diagram_{diagram_count}.png", diagram_img)
            diagram_count += 1
            return 0

    print(f"Extracted {diagram_count} diagrams and saved in '{output_folder}'.")

def visualize_text_region (img, filtered_line_segments, output_path):
    """Removes the text region by replacing it with a white background."""

    img_copy = img.copy()
    img_h, img_w = img.shape[:2]

    # Find the extreme points for the text region
    if filtered_line_segments:
        min_y = min([seg[0] for seg in filtered_line_segments])
        max_y = max([seg[1] for seg in filtered_line_segments])
        
        cv2.imwrite("output/text.png", img_copy[min_y -40:max_y + 40, :])
        # Replace the text region with white pixels
        img_copy[min_y -40:max_y + 40, :] = (255, 255, 255)  # White background
    
    # Save debug image
    cv2.imwrite(output_path, img_copy)
    print(f"Text removed image saved at {output_path}")

def segment_lines_and_find_diagrams(img, output_folder="output", min_height_threshold=30, padding=10, min_contour_width=5000):
    """Segments text lines, removes them, and detects diagrams outside text area."""
    img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output/original_image.png", img)
    img = correct_tilt(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    # Horizontal histogram
    hist = np.sum(binary, axis=1)
    threshold = np.max(hist) * 0.2
    lines = np.where(hist > threshold)[0]

    # Detect text line segments
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

    # Create output folders
    segmented_folder = os.path.join(output_folder, "segmented_lines")
    diagram_folder = os.path.join(output_folder, "diagrams")

    os.makedirs(segmented_folder, exist_ok=True)
    os.makedirs(diagram_folder, exist_ok=True)

    img_height, img_width = img.shape[:2]
    text_mask = np.ones((img_height, img_width), dtype=np.uint8) * 255  # Start with a white mask

    filtered_line_segments = []
    for y_start, y_end in line_segments:
        line_segment = binary[y_start:y_end, :]
        left_whitespace = np.sum(line_segment[:, :20] == 0) / (20 * line_segment.shape[0])
        right_whitespace = np.sum(line_segment[:, -20:] == 0) / (20 * line_segment.shape[0])

        if left_whitespace > 0.83 and right_whitespace < 0.96:
            filtered_line_segments.append((y_start, y_end))
    
    for idx, (y_start, y_end) in enumerate(filtered_line_segments):
        y_start_pad = max(0, y_start - 40)
        y_end_pad = min(img_height, y_end + padding)

        line_img = img[y_start_pad:y_end_pad, :]
        cv2.imwrite(f"{segmented_folder}/line_{idx+1}.png", line_img)

    print(f"Segmented {len(filtered_line_segments)} text lines and saved in '{segmented_folder}'.")

    # **Visualize bounding box for text region**
    visualize_text_region(img, filtered_line_segments, os.path.join(output_folder, "text_bounding_box.png"))
    # detect_and_visualize_contours("output/text_bounding_box.png", "output/diagrams_contours.png")

    # **Extract diagrams**
    diagram_folder = extract_diagram("output/text_bounding_box.png", "output/diagrams")
    return segmented_folder, diagram_folder

# Example usage
if __name__ == "__main__":
    img = convert_from_path("Utils/Images/Diagram_answer4.pdf")[0]
    img = img.convert("RGB")
    segment_lines_and_find_diagrams(np.array(img), min_contour_width=500)