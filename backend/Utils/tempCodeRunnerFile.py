def ocr_from_image(image):
    """
    Extract text from an image using Tesseract OCR.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG') 
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