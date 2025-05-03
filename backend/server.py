from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List
from fastapi import Form
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from Utils.segmentation import segment_lines_and_find_diagrams
from Utils.ocr import ocr_from_image
from Utils.similarity import text_similarity
from Utils.image_similarity import image_similarity
import numpy as np
import os
import shutil
import base64

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins="http://localhost:5173",
    allow_methods=["*"],
    allow_headers=["*"] 
)

class AnswerKey(BaseModel):
    text: str

@app.post("/similarity")
async def similarity(
    answer_key_text: str = Form(...),  # Receive text as a form field
    answer_key_diagram: UploadFile = File(...),  # Receive diagram as an image file
    answer_sheets: List[UploadFile] = File(...)  # Multiple answer sheets as files
):
    # Process text
    answer_key = AnswerKey(text=answer_key_text)

    # Process diagram
    diagram_bytes = await answer_key_diagram.read()
    diagram_image = Image.open(BytesIO(diagram_bytes))

    # Process answer sheets
    texts = []
    for sheet in answer_sheets:
        # Ensure the uploaded file is not empty
        if sheet.size == 0:
            raise HTTPException(status_code=400, detail=f"Answer sheet '{sheet.filename}' is empty")

        # Read and process each answer sheet PDF
        sheet_content = await sheet.read()  # Read the bytes of the PDF
        if not sheet_content:
            raise HTTPException(status_code=400, detail=f"Failed to read the answer sheet '{sheet.filename}'")
        
        try:
            images = convert_from_bytes(sheet_content)

            for image in images:
                image = image.convert("RGB")  # Ensure proper color mode
                image = np.array(image)  # Convert to NumPy array
                segment_lines_and_find_diagrams(np.array(image), output_folder="output", min_height_threshold=30, padding=10, min_contour_width=5000)

            text_image = Image.open("output/text.png")
            #texts.append(ocr_from_image(text_image))
            texts.append(ocr_from_image("output/segmented_lines"))
            #texts.append("Machine learning is a branch of artificial intelligence that enables computers to learn the data , identify patterns , and make decisions with minimal human intervention . Instead of being explicitly programmed for specific tasks , Machine learning algorithms analyze and interpret large datasets to Correlations that can be used for predictions or decision making . These algorithms Continuously improve their accuracy over time as they process more datasets to unconver trends and correlations that can be used for predictions or decision making. These algorithms continuously improve their accuracy over time as they process more data.")
            
            text_image.close()  # Close the image to free up resources
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

    diagrams = ["output/diagrams/"+f for f in os.listdir("output/diagrams") if f.endswith(".png")]   
    diagram_images = [Image.open(diagram_path) for diagram_path in diagrams] 
    diagram_similarities = image_similarity(diagram_image, diagram_images)

    text_paths= ["output/texts/"+f for f in os.listdir("output/texts") if f.endswith(".png")]
    text_paths.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    text_images = [Image.open(text_path) for text_path in text_paths]
    text_similarities = [text_similarity(answer_key.text, text) for text in texts]


    shutil.rmtree("output/diagrams")
    response_data = {}
    for i in range(len(text_similarities)):
        # Encode diagram image
        buffer_diagram = BytesIO()
        diagram_images[i].save(buffer_diagram, format="PNG")
        encoded_image = base64.b64encode(buffer_diagram.getvalue()).decode('utf-8')
        buffer_diagram.close()

        # Encode text image
        buffer_text = BytesIO()
        text_images[i].save(buffer_text, format="PNG")
        encoded_text_image = base64.b64encode(buffer_text.getvalue()).decode('utf-8')
        buffer_text.close()


        response_data[i] = [
            float(text_similarities[i]),
            float(diagram_similarities[i]),
            texts[i],
            encoded_image,
            encoded_text_image,
        ]

    shutil.rmtree("output/texts")
    return JSONResponse(content=response_data)

