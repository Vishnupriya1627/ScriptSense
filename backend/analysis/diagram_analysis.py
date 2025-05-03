import open_clip
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import imagehash
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tabulate import tabulate

# def transform(x):
#     steepness = 15
#     center = 0.875  # midpoint of steep curve between 0.8 and 0.95

#     # Sigmoid function scaled to output range [0.5, 1]
#     y = 0.5 + 0.5 * torch.sigmoid(steepness * (x - center))
#     return y

def transform(x):
    steepness = 15
    center = 0.875  # midpoint of steep curve between 0.8 and 0.95

    # Sigmoid function scaled to output range [0.5, 1]
    y = 0.5 + 0.5 * torch.sigmoid(steepness * (x - center))
    return y

class ImageSimilarityComparator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_models()
        
    def _init_models(self):
        """Initialize all models and feature detectors"""
        # CLIP model
        self.clip_model, self.clip_preprocess, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        self.clip_model = self.clip_model.to(self.device).eval()
        
        # Traditional CV detectors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        
    def _clip_similarity(self, img1, img2):
        """CLIP-based semantic similarity"""
        img1_tensor = self.clip_preprocess(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.clip_preprocess(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            emb1 = self.clip_model.encode_image(img1_tensor)
            emb2 = self.clip_model.encode_image(img2_tensor)
            
        return transform(torch.nn.functional.cosine_similarity(emb1, emb2)).item()
    
    def _sift_similarity(self, img1, img2):
        """SIFT feature matching similarity"""
        img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(img1_gray, None)
        kp2, des2 = self.sift.detectAndCompute(img2_gray, None)
        
        if des1 is None or des2 is None:
            return 0.0
            
        # FLANN parameters and matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)
                
        return len(good_matches) / max(len(kp1), len(kp2))
    
    def _structural_similarity(self, img1, img2):
        """Structural Similarity Index (SSIM)"""
        img1_arr = np.array(img1.convert('L'))
        img2_arr = np.array(img2.convert('L'))
        
        # Ensure same shape
        min_shape = (min(img1_arr.shape[0], img2_arr.shape[0]),
                    min(img1_arr.shape[1], img2_arr.shape[1]))
        img1_arr = img1_arr[:min_shape[0], :min_shape[1]]
        img2_arr = img2_arr[:min_shape[0], :min_shape[1]]
        
        return ssim(img1_arr, img2_arr)
    
    def _phash_similarity(self, img1, img2):
        """Perceptual hash similarity"""
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        return 1 - (hash1 - hash2)/len(hash1.hash)**2
    
    def compare_images(self, ref_path, compare_paths):
        """Compare reference image with multiple images using all methods"""
        ref_img = Image.open(ref_path)
        results = []
        
        for path in compare_paths:
            cmp_img = Image.open(path)
            
            metrics = {
                'image': Path(path).name,
                'CLIP': self._clip_similarity(ref_img, cmp_img),
                'SIFT': self._sift_similarity(ref_img, cmp_img),
                'SSIM': self._structural_similarity(ref_img, cmp_img),
                'pHash': self._phash_similarity(ref_img, cmp_img)
            }
            
            # Normalize SIFT scores between 0-1
            metrics['SIFT'] = np.clip(metrics['SIFT'], 0, 1)
            
            results.append(metrics)
            
        return pd.DataFrame(results)
    
    def visualize_comparison(self, df):
        """Create visualization of comparison results"""
        plt.figure(figsize=(12, 6))
        
        # Melt dataframe for seaborn
        plot_df = df.melt(id_vars='image', var_name='Method', value_name='Score')
        
        # Create grouped bar plot
        sns.barplot(x='image', y='Score', hue='Method', data=plot_df)
        plt.ylim(0, 1)
        plt.ylabel('Similarity Score')
        plt.xlabel('Compared Images')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Create correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.drop(columns='image').corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Method Score Correlations')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == '__main__':
    comparator = ImageSimilarityComparator()
    
    # Example images
    ref_image = 'images/key.jpg'
    compare_images = ['images/example1.png', 'images/example2.png', 'images/example3.jpg']
    
    # Generate comparison results
    results_df = comparator.compare_images(ref_image, compare_images)
    
    # Print formatted table
    print("\nImage Similarity Comparison Table:")
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Generate visualizations
    comparator.visualize_comparison(results_df)
