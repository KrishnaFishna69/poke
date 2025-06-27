import pandas as pd
import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import logging
import time
from transformers import AutoProcessor, AutoModel
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('art_scoring.log'),
        logging.StreamHandler()
    ]
)

# Load dataset
logging.info("Loading art_scoring.csv...")
df = pd.read_csv("art_scoring.csv")
logging.info(f"Loaded {len(df)} cards from art_scoring.csv")

# Run on entire dataset
logging.info(f"PRODUCTION MODE: Processing all {len(df)} cards")

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load PickScore model & processor
logging.info("Loading PickScore model and processor...")
processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
logging.info("PickScore model loaded successfully")

# Prompt describing preferred card art style
prompt = ("A highly detailed full-art Pokémon card with vibrant colors, dramatic action, minimal text, creative background, and expressive character pose — looks premium and collectible.")

# Generate image URL helper
def get_image_url(row):
    try:
        set_code, number = row['id'].split('-')
        url = f"https://images.pokemontcg.io/{set_code}/{number}_hires.png"
        return url
    except Exception as e:
        logging.error(f"Failed to generate URL for {row['id']}: {str(e)}")
        return None

df['image_url'] = df.apply(get_image_url, axis=1)

# Download image helper
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        logging.error(f"Failed to download image {url}: {str(e)}")
        return None

# Score batch of PIL images with PickScore
def score_images_with_pickscore(images, prompt):
    inputs = processor(
        text=[prompt] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**{k: inputs[k] for k in inputs if k.startswith("pixel_values")})
        text_features = model.get_text_features(**{k: inputs[k] for k in inputs if k.startswith("input_ids") or k.startswith("attention_mask")})

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (text_features @ image_features.T)

        # ✅ FIX: Only return first row — one prompt, multiple images
        scores = logits[0].cpu().tolist()
        return scores

# Batch processing loop
batch_size = 50
scores = []
start_time = time.time()

logging.info(f"Starting scoring of {len(df)} cards in batches of {batch_size}...")

for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i:i+batch_size]
    images = []
    valid_indices = []

    for idx, row in batch_df.iterrows():
        img = download_image(row['image_url'])
        if img is not None:
            images.append(img)
            valid_indices.append(idx)
        else:
            scores.append((idx, None))

    if images:
        batch_scores = score_images_with_pickscore(images, prompt)
        for idx, score in zip(valid_indices, batch_scores):
            scores.append((idx, float(score)))
    else:
        logging.warning(f"No valid images in batch starting at index {i}")

# Fill score column
score_series = pd.Series(index=df.index, dtype=float)
for idx, score in scores:
    score_series.at[idx] = score if score is not None else None

df['pickscore_score'] = score_series

end_time = time.time()
logging.info(f"✅ Completed scoring {len(df)} cards in {end_time - start_time:.2f} seconds (PRODUCTION MODE)")

# Rescale scores
min_score = df['pickscore_score'].min()
max_score = df['pickscore_score'].max()
logging.info(f"Score range before rescale: min={min_score}, max={max_score}")

if pd.isna(min_score) or pd.isna(max_score) or min_score == max_score:
    logging.warning("Cannot rescale — min and max are equal or missing. Assigning neutral score.")
    df['art_score_0_10'] = 5
else:
    df['art_score_0_10'] = 10 * (df['pickscore_score'] - min_score) / (max_score - min_score)

# Save production results
df.to_csv("art_scoring_complete.csv", index=False)
logging.info("Production results saved to art_scoring_complete.csv")
logging.info(f"Art scoring completed for all {len(df)} cards with global rescaling.")
