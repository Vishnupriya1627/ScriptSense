# !pip install open_clip_torch
import open_clip
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

model = model.to(device)

def transform(x):
    return torch.where(
        x >= 0.9, x,  # Keep values ≥ 0.9 unchanged
        0.5 * (x / 0.85)  # Scale down everything below 0.9 significantly
    )



def image_similarity(truth_file, image_files):

    # preprocessing the image
    truth_tensor = preprocess(truth_file).unsqueeze(0).to(device)
    
    # preprocessing the images
    #test_images = [Image.open(image_file_path) for image_file_path in images_file_path_array]
    test_tensors = torch.stack([preprocess(img) for img in image_files]).to(device)
    
    #generating the image embedding
    with torch.no_grad():
        truth_embed = model.encode_image(truth_tensor)
        test_embeds = model.encode_image(test_tensors)
        
        
    similarities = torch.nn.functional.cosine_similarity(truth_embed, test_embeds)
    
    return transform(similarities).cpu().numpy()

if __name__ == '__main__':
    print(image_similarity(Image.open("backend/Utils/truth.png"), ["backend/Utils/image.png", "backend/Utils/image1.png"]))
