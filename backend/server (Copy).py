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
import time

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
    answer_key_text: str = Form(...),
    answer_key_diagram: UploadFile = File(...),
    answer_sheets: List[UploadFile] = File(...)
):
    start_time = time.time()
    
    print(f"[{time.time()-start_time:.1f}s] 📨 Starting request processing...")
    
    # Process text
    answer_key = AnswerKey(text=answer_key_text)

    # Process diagram
    print(f"[{time.time()-start_time:.1f}s] 🖼️ Reading diagram...")
    diagram_bytes = await answer_key_diagram.read()
    diagram_image = Image.open(BytesIO(diagram_bytes))

    # Process answer sheets
    texts = []
    
    for sheet_idx, sheet in enumerate(answer_sheets):
        sheet_start = time.time()
        print(f"[{time.time()-start_time:.1f}s] 📄 Processing sheet {sheet_idx+1}...")
        
        if sheet.size == 0:
            raise HTTPException(status_code=400, detail=f"Answer sheet '{sheet.filename}' is empty")

        sheet_content = await sheet.read()
        if not sheet_content:
            raise HTTPException(status_code=400, detail=f"Failed to read the answer sheet '{sheet.filename}'")
        
        try:
            print(f"[{time.time()-start_time:.1f}s] 📊 Converting PDF to images...")
            images = convert_from_bytes(sheet_content)
            print(f"[{time.time()-start_time:.1f}s] ✅ Got {len(images)} page(s)")

            for page_idx, image in enumerate(images):
                print(f"[{time.time()-start_time:.1f}s] 🖼️ Processing page {page_idx+1}...")
                image = image.convert("RGB")
                image_array = np.array(image)
                
                print(f"[{time.time()-start_time:.1f}s] ✂️ Starting segmentation...")
                segment_lines_and_find_diagrams(image_array, output_folder="output", min_height_threshold=30, padding=10, min_contour_width=5000)
                print(f"[{time.time()-start_time:.1f}s] ✅ Segmentation complete")

            # FIXED OCR CALL - Read actual image files
            print(f"[{time.time()-start_time:.1f}s] 🔤 Starting OCR...")
            
            # Check what files exist
            if os.path.exists("output/texts"):
                text_files = os.listdir("output/texts")
                print(f"[{time.time()-start_time:.1f}s] Found {len(text_files)} text images")
                
                # Read each text image and perform OCR
                for text_file in sorted(text_files):
                    if text_file.endswith('.png'):
                        text_path = os.path.join("output/texts", text_file)
                        with open(text_path, "rb") as f:
                            image_bytes = f.read()
                            ocr_result = ocr_from_image(image_bytes)
                            texts.append(ocr_result)
                            print(f"[{time.time()-start_time:.1f}s] OCR result: '{ocr_result[:50]}...'")
            else:
                print(f"[{time.time()-start_time:.1f}s] ⚠️ No text images found, using fallback")
                texts.append("Sample text from answer sheet")
                
        except Exception as e:
            print(f"[{time.time()-start_time:.1f}s] ❌ Error: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing sheet: {str(e)}")

    # Get diagram images
    print(f"[{time.time()-start_time:.1f}s] 🖼️ Loading diagram images...")
    diagrams = []
    if os.path.exists("output/diagrams"):
        diagrams = [os.path.join("output/diagrams", f) for f in os.listdir("output/diagrams") if f.endswith(".png")]
    
    diagram_images = [Image.open(diagram_path) for diagram_path in diagrams] if diagrams else []
    
    # Calculate diagram similarities
    print(f"[{time.time()-start_time:.1f}s] 📊 Calculating diagram similarity...")
    if diagram_images:
        diagram_similarities = image_similarity(diagram_image, diagram_images)
    else:
        diagram_similarities = [0.0] * len(texts)
    
    # Get text images for response
    print(f"[{time.time()-start_time:.1f}s] 📝 Loading text images...")
    text_paths = []
    if os.path.exists("output/texts"):
        text_paths = [os.path.join("output/texts", f) for f in os.listdir("output/texts") if f.endswith(".png")]
        text_paths.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    text_images = [Image.open(text_path) for text_path in text_paths] if text_paths else []
    
    # Calculate text similarities
    print(f"[{time.time()-start_time:.1f}s] 📊 Calculating text similarity...")
    text_similarities = []
    for i, text in enumerate(texts):
        similarity = text_similarity(answer_key.text, text)
        text_similarities.append(similarity)
        print(f"[{time.time()-start_time:.1f}s] Text {i+1} similarity: {similarity:.3f}")

    # Prepare response
    print(f"[{time.time()-start_time:.1f}s] 📦 Preparing response...")
    response_data = {}
    
    for i in range(len(text_similarities)):
        # Encode diagram image if available
        encoded_image = ""
        if i < len(diagram_images):
            buffer_diagram = BytesIO()
            diagram_images[i].save(buffer_diagram, format="PNG")
            encoded_image = base64.b64encode(buffer_diagram.getvalue()).decode('utf-8')
            buffer_diagram.close()

        # Encode text image if available
        encoded_text_image = ""
        if i < len(text_images):
            buffer_text = BytesIO()
            text_images[i].save(buffer_text, format="PNG")
            encoded_text_image = base64.b64encode(buffer_text.getvalue()).decode('utf-8')
            buffer_text.close()

        response_data[i] = [
            float(text_similarities[i]),
            float(diagram_similarities[i] if i < len(diagram_similarities) else 0.0),
            texts[i] if i < len(texts) else "",
            encoded_image,
            encoded_text_image,
        ]

    # Cleanup
    print(f"[{time.time()-start_time:.1f}s] 🧹 Cleaning up...")
    try:
        if os.path.exists("output/diagrams"):
            shutil.rmtree("output/diagrams")
        if os.path.exists("output/texts"):
            shutil.rmtree("output/texts")
    except:
        pass

    total_time = time.time() - start_time
    print(f"[{total_time:.1f}s] ✅ Request completed in {total_time:.1f} seconds")
    
    return JSONResponse(content=response_data)
