from mindflix1 import ImageProcessingPipeline, ImageRetriever
import os
import logging
import mimetypes
import os.path
from typing import Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIR = "./chroma_db"
IMAGE_FOLDER = "BPLD Dataset"  # Your image folder path

def train_pipeline():
    logger.info("Starting training pipeline...")
    
    # Initialize pipeline with image folder
    pipeline = ImageProcessingPipeline(
        persist_directory=PERSIST_DIR,
        image_folder=IMAGE_FOLDER
    )
    
    # Rest of your training code remains the same
    logger.info("Generating captions...")
    captions = pipeline.generate_captions(IMAGE_FOLDER)
    logger.info(f"Generated captions for {len(captions)} images")
    
    logger.info("Fine-tuning BLIP model...")
    fine_tuned_model_path = pipeline.fine_tune_blip(captions)
    
    logger.info("Encoding data...")
    image_embeddings, text_embeddings = pipeline.encode_data(captions)
    
    logger.info("Storing embeddings...")
    pipeline.store_embeddings(image_embeddings)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train_pipeline()