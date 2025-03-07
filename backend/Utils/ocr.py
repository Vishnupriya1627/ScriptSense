import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import io


def preprocess_image(image):
    """
    Preprocess the image for better OCR accuracy.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply binary thresholding to remove noise and make the text stand out
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Denoising using morphological transformations (optional)
    kernel = np.ones((1, 1), np.uint8)
    denoised = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return denoised


def ocr_from_image(image):
    """
    Extract text from an image using Tesseract OCR.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Ensure format matches your image type
    content = img_byte_arr.getvalue()
    
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    plain_text = ""

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    plain_text += word_text + " "
        
        return plain_text.strip()
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


