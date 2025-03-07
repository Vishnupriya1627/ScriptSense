# !pip install open_clip_torch
import open_clip
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

model = model.to(device)

def transform(x):
    return torch.where(
        x > 0.8, x,
        torch.where(
            x > 0.7, 0.4 * (x - 0.7) / 0.2,
            0.3 * x / 0.7  
        )
    )

def image_similarity(truth_file, images_file_path_array):

    # preprocessing the image
    truth_tensor = preprocess(truth_file).unsqueeze(0).to(device)
    
    # preprocessing the images
    test_images = [Image.open(image_file_path) for image_file_path in images_file_path_array]
    test_tensors = torch.stack([preprocess(img) for img in test_images]).to(device)
    

    #generating the image embedding
    with torch.no_grad():
        truth_embed = model.encode_image(truth_tensor)
        test_embeds = model.encode_image(test_tensors)
        
        
    similarities = torch.nn.functional.cosine_similarity(truth_embed, test_embeds)
    
    return transform(similarities).cpu().numpy()
