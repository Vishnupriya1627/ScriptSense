import os
import sys
sys.path.insert(0, '/content/Uni-MuMER')

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
        for path in line_paths:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(path)}"},
                    {"type": "text", "text": "I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format."}
                ]
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
            prompts.append({
                "prompt": text,
                "multi_modal_data": {"image": image_inputs}
            })

        outputs = llm.generate(prompts, sampling_params)

        results = {}
        for i, (path, output) in enumerate(zip(line_paths, outputs)):
            pred = output.outputs[0].text.strip()
            fname = os.path.basename(path)
            results[fname] = pred
            print(f"   ✅ {fname} → {pred}")

        return results

    except Exception as e:
        print(f"[Error] Formula recognition failed: {e}")
        import traceback
        traceback.print_exc()
        return {}