import os
import sys
import cv2
import re
sys.path.insert(0, '/content/Uni-MuMER')

def resize_for_unimer(image_path, max_pixels=800):
    """Resize image to reduce token count."""
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    if max(h, w) > max_pixels:
        scale = max_pixels / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, img)
    return image_path

def print_unimer_lines(pred):
    """Print UniMuMER output line by line."""
    if r'\begin{array}' in pred or r'\begin {array}' in pred:
        inner = re.sub(r'\\begin\s*\{array\}.*?\}', '', pred)
        inner = re.sub(r'\\end\s*\{array\}', '', inner)
        lines = [l.strip() for l in inner.split('\\\\') if l.strip()]
        for j, line in enumerate(lines):
            print(f"   Line {j+1}: {line}")
    else:
        print(f"   Line 1: {pred}")

def recognize_formulas(line_paths, llm, sampling_params):
    """Run Uni-MuMER recognition using preloaded model."""
    if not llm or not line_paths:
        return {}
    
    try:
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(
            "/content/Uni-MuMER/models/Uni-MuMER-3B",
            trust_remote_code=True
        )

        prompts = []
        valid_paths = []
        for path in line_paths:
            resize_for_unimer(path, max_pixels=800)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(path)}"},
                    {"type": "text", "text": "I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format."}
                ]
            }]
            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
                prompts.append({
                    "prompt": text,
                    "multi_modal_data": {"image": image_inputs}
                })
                valid_paths.append(path)
            except Exception as e:
                print(f"   ⚠️ Skipping {path}: {e}")

        if not prompts:
            return {}

        outputs = llm.generate(prompts, sampling_params)

        results = {}
        print("\n   === UniMuMER Recognition Results ===")
        for i, (path, output) in enumerate(zip(valid_paths, outputs)):
            pred = output.outputs[0].text.strip()
            fname = os.path.basename(path)
            results[fname] = pred
            print_unimer_lines(pred)
        print("   =====================================\n")

        return results

    except Exception as e:
        print(f"[Error] Formula recognition failed: {e}")
        import traceback
        traceback.print_exc()
        return {}