import cv2
import numpy as np
from PIL import Image
import io
import os
import pytesseract

def ocr_from_image(image_bytes_or_path):
    """
    Perform OCR on:
    - Folder path
    - Image file path
    - Raw image bytes
    """

    if isinstance(image_bytes_or_path, str):

        # If it's a folder
        if os.path.isdir(image_bytes_or_path):
            return ocr_from_folder(image_bytes_or_path)

        # If it's a file
        if os.path.isfile(image_bytes_or_path):
            with open(image_bytes_or_path, 'rb') as f:
                return ocr_from_single_image(f.read())

    elif isinstance(image_bytes_or_path, bytes):
        return ocr_from_single_image(image_bytes_or_path)

    print(f"[OCR Warning] Unexpected input type: {type(image_bytes_or_path)}")
    return ""

def ocr_from_single_image(image_bytes):
    """
    Perform OCR on a single image (bytes) using pytesseract.
    Prints each detected line to console.
    """
    try:
        # Convert bytes to numpy array via PIL
        image = Image.open(io.BytesIO(image_bytes))
        original_size = image.size
        
        # Convert to grayscale for better OCR
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance image for better OCR
        image_array = np.array(image)
        
        # Apply multiple preprocessing techniques
        
        # Method 1: Otsu thresholding
        _, thresh_otsu = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        thresh_adaptive = cv2.adaptiveThreshold(image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 15, 10)
        
        # Try both and see which gives better results
        # For now, use Otsu
        enhanced = thresh_otsu
        
        # Denoise
        denoised = cv2.medianBlur(enhanced, 1)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(denoised)
        
        # OCR configuration for better accuracy
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}@#$%^&*+-=<>/\\|~`\'\" "'
        
        # Get detailed OCR data including bounding boxes
        try:
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(enhanced_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract text lines
            lines = []
            current_line = []
            current_line_num = -1
            
            print("\n" + "="*60)
            print(f"📝 OCR DETECTED LINES (Image size: {original_size[0]}x{original_size[1]})")
            print("="*60)
            
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    line_num = ocr_data['line_num'][i]
                    
                    if line_num != current_line_num:
                        if current_line:
                            line_text = ' '.join(current_line)
                            lines.append(line_text)
                            print(f"Line {len(lines)}: {line_text}")
                        current_line = []
                        current_line_num = line_num
                    
                    current_line.append(text.strip())
            
            # Add last line
            if current_line:
                line_text = ' '.join(current_line)
                lines.append(line_text)
                print(f"Line {len(lines)}: {line_text}")
            
            print(f"\n📊 Total lines detected: {len(lines)}")
            print("="*60 + "\n")
            
            # Also get regular text for return value
            text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            return text.strip()
            
        except Exception as e:
            print(f"[OCR Detailed Error] {e}, falling back to simple OCR")
            # Fallback: simple OCR
            text = pytesseract.image_to_string(enhanced_image)
            
            # Print simple output
            print("\n" + "="*60)
            print("📝 OCR DETECTED TEXT:")
            print("="*60)
            if text.strip():
                for i, line in enumerate(text.strip().split('\n')):
                    if line.strip():
                        print(f"Line {i+1}: {line.strip()}")
            else:
                print("No text detected")
            print("="*60 + "\n")
            
            return text.strip()
        
    except Exception as e:
        print(f"[Single Image OCR Error] {e}")
        # Fallback: try direct OCR without preprocessing
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            
            # Print simple output
            print("\n" + "="*60)
            print("📝 OCR DETECTED TEXT (fallback):")
            print("="*60)
            if text.strip():
                for i, line in enumerate(text.strip().split('\n')):
                    if line.strip():
                        print(f"Line {i+1}: {line.strip()}")
            else:
                print("No text detected")
            print("="*60 + "\n")
            
            return text.strip()
        except Exception as e2:
            print(f"[Fallback OCR Error] {e2}")
            return ""

def ocr_from_folder(folder_path):
    """
    Perform OCR on all images in a folder and concatenate results.
    Prints each image's detected lines.
    """
    full_text = ""
    
    try:
        # Get all PNG files in folder
        filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        
        # Sort by number (e.g., "line_0.png", "line_1.png")
        def extract_number(filename):
            try:
                # Extract number from filename like "line_0.png" or "text_0.png"
                parts = filename.split('_')
                if len(parts) > 1:
                    return int(parts[-1].split('.')[0])
                return 0
            except:
                return 0
        
        filenames.sort(key=extract_number)
        
        print(f"\n[OCR] Processing {len(filenames)} images from folder: {folder_path}")
        print("="*60)
        
        # Process each image
        for idx, filename in enumerate(filenames):
            image_path = os.path.join(folder_path, filename)
            
            print(f"\n📄 Image {idx+1}/{len(filenames)}: {filename}")
            print("-" * 40)
            
            try:
                # Read image as bytes
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # OCR this image (ocr_from_single_image will print details)
                text = ocr_from_single_image(image_bytes)
                
                if text:
                    full_text += text + "\n"
                else:
                    print(f"   ⚠️ No text detected in {filename}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ OCR COMPLETE: Total {len(filenames)} images processed")
        print(f"📊 Combined text length: {len(full_text)} characters")
        print(f"{'='*60}\n")
        
        return full_text.strip()
        
    except Exception as e:
        print(f"[Folder OCR Error] {e}")
        return ""

# For testing directly
if __name__ == "__main__":
    # Test with a sample image if provided
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n🔍 Testing OCR on: {image_path}")
        text = ocr_from_image(image_path)
        print(f"\n✅ Final extracted text:\n{text}")
