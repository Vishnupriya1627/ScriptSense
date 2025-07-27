# AnswerSheetCorrector
## Overview
AnswerSheetCorrector is a software that processes subjective exam answer sheets by:

- Segmenting individual lines of text into separate images while maintaining a minimum height and preserving the original width while ensuring sufficient white space.
- Extracts diagram by negative text area from the image space.

### Features:
- Handwritten OCR: Recognizes handwritten answers using Tesseract.
- Line Segmentation: Splits answer sheets into individual line images.
- Diagram Detection: Identifies and extracts diagrams separately.
- Web Interface: Provides an interactive UI for uploading and processing answer sheets.

## Setup
1. git clone https://github.com/yourusername/AnswerSheetCorrector.git
   cd AnswerSheetCorrector

2. cd backend
   pip install requirements.txt

3. cd ../frontend/my-react-app
   npm install
   npm start

## Usage
1️⃣ Upload scanned answer sheets.

2️⃣ The system detects text and diagrams separately.

3️⃣ Processed results are displayed for review.

4️⃣ Export results as structured data.

## Demo Video

[![Watch the Demo](https://img.youtube.com/vi/35ISmecZJWY/0.jpg)](https://www.youtube.com/watch?v=35ISmecZJWY)

  
