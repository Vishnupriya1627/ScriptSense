import cv2
import numpy as np
import os
import shutil

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
            
            # More lenient diagram detection
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
            
            # Extract text region with padding
            y_start = max(min_y - 20, 0)
            y_end = min(max_y + 20, img_h)
            text_crop = img_copy[y_start:y_end, :]
            
            # Save the text region
            text_path = os.path.join(output_texts_folder, f"text_0.png")
            cv2.imwrite(text_path, text_crop)
            print(f"✅ Text region cropped and saved: {text_path}")

            # Remove text region from image (fill with white)
            img_copy[y_start:y_end, :] = (255, 255, 255)
            cv2.imwrite(output_path, img_copy)
            print(f"✅ Text removed image saved: {output_path}")
            
            return True
        else:
            print("[Warning] No text segments found to visualize.")
            # Save empty text file as fallback
            cv2.imwrite(output_path, img_copy)
            return False

    except Exception as e:
        print(f"[Error] Text region visualization failed: {str(e)}")
        return False

def segment_lines_and_find_diagrams(img, output_folder="output", min_height_threshold=15, padding=5, min_contour_width=1000):
    """
    Main function to segment text lines and find diagrams.
    """
    try:
        # Create output directories
        segmented_folder = os.path.join(output_folder, "segmented_lines")
        texts_folder = os.path.join(output_folder, "texts")
        diagrams_folder = os.path.join(output_folder, "diagrams")
        
        os.makedirs(segmented_folder, exist_ok=True)
        os.makedirs(texts_folder, exist_ok=True)
        os.makedirs(diagrams_folder, exist_ok=True)

        # Convert RGB to BGR for OpenCV
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save original
        cv2.imwrite(f"{output_folder}/original_image.png", img)

        # Correct tilt
        img = correct_tilt(img)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 1. TEXT DETECTION - More robust approach
        # Use Otsu's thresholding instead of adaptive for better line detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate to connect text in same line
        kernel = np.ones((3, 10), np.uint8)  # Horizontal dilation to connect words
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours of text lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort text contours
        text_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Text heuristics
            if h > min_height_threshold and w > 30 and h < 150:
                if w / h < 20:  # Not extremely wide
                    text_contours.append((x, y, w, h))
        
        # Sort by y-coordinate (top to bottom)
        text_contours.sort(key=lambda r: r[1])
        
        # Merge overlapping contours on same line
        merged_contours = []
        for contour in text_contours:
            if not merged_contours:
                merged_contours.append(list(contour))
                continue
            
            last = merged_contours[-1]
            x, y, w, h = contour
            
            # Check if same line (vertical overlap)
            y_overlap = max(0, min(last[1] + last[3], y + h) - max(last[1], y))
            if y_overlap > 0:
                # Merge horizontally
                last[0] = min(last[0], x)
                last[1] = min(last[1], y)
                last[2] = max(last[0] + last[2], x + w) - last[0]
                last[3] = max(last[1] + last[3], y + h) - last[1]
            else:
                merged_contours.append(list(contour))
        
        print(f"✅ Detected {len(merged_contours)} text lines")
        
        # Save individual text lines
        for idx, (x, y, w, h) in enumerate(merged_contours):
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            line_img = img[y1:y2, x1:x2]
            cv2.imwrite(f"{segmented_folder}/line_{idx+1}.png", line_img)
        
        # Create line segments format for text removal
        line_segments = [(y, y + h) for (x, y, w, h) in merged_contours]
        
        # Remove text and extract diagrams
        text_bounding_box_path = os.path.join(output_folder, "text_bounding_box.png")
        visualize_text_region(img, line_segments, text_bounding_box_path, texts_folder)
        
        # Extract diagrams from text-free image
        diagram_count = extract_diagram(text_bounding_box_path, diagrams_folder)
        
        return len(merged_contours), diagram_count if diagram_count else 0

    except Exception as e:
        print(f"[Error] Line segmentation and diagram detection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0
