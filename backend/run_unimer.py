"""
Subprocess worker: loads Uni-MuMER, processes all page images, writes output JSON.
Called by server.py via subprocess.run(). Exits when done — fully releasing GPU memory.

Args (via sys.argv):
    1: path to input JSON  {"pages": ["page_0.png", ...], "sheet_dir": "..."}
    2: path to output JSON (written by this script)
"""
import sys, os, json, re
sys.path.insert(0, '/content/Uni-MuMER')

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams
from Utils.segmentation import segment_lines_and_find_diagrams

UNIMER_MODEL_PATH = "/content/Uni-MuMER/models/Uni-MuMER-3B"

def clean_unimumer_output(text: str) -> str:
    if not text:
        return text
    # Merge spaced-out single chars: "G r a d i e n t" → "Gradient"
    text = re.sub(
        r'(?<!\S)((?:[A-Za-z] )+[A-Za-z])(?!\S)',
        lambda m: m.group(0).replace(' ', ''),
        text
    )
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)\.', r'\1. ', text)
    text = text.replace(" .", ".").replace(" ,", ",")
    return text.strip()

def main():
    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        args = json.load(f)

    page_paths = args["pages"]
    sheet_dir  = args["sheet_dir"]

    print("🔄 Loading Uni-MuMER...")
    llm = LLM(
        model=UNIMER_MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    sampling = SamplingParams(temperature=0, max_tokens=512)
    print("✅ Uni-MuMER loaded")

    page_line_counts = []

    for page_idx, page_path in enumerate(page_paths):
        img = Image.open(page_path).convert("RGB")
        arr = np.array(img)
        count, _ = segment_lines_and_find_diagrams(
            arr,
            output_folder=sheet_dir,
            llm=llm,
            sampling_params=sampling,
        )
        page_line_counts.append(count)
        print(f"📄 Page {page_idx+1}: Uni-MuMER lines={count}")

    raw_parts = []
    latex_file = os.path.join(sheet_dir, "formulas_latex.json")

    if os.path.exists(latex_file):
        with open(latex_file) as f:
            data = json.load(f)
        print("\n==============================")
        print("📝 Uni-MuMER RAW OUTPUT")
        print("==============================")
        for i, v in enumerate(data.values()):
            cleaned = clean_unimumer_output(v)
            if cleaned.strip():
                raw_parts.append(cleaned)
                print(f"Line {i+1}: {cleaned}")
        print("==============================\n")

    result = {
        "raw_text": " ".join(raw_parts),
        "page_line_counts": page_line_counts,
    }

    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"✅ Uni-MuMER output written to {output_path}")

if __name__ == "__main__":
    main()