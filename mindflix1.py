from __future__ import annotations

import os
import torch
import re
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import Dict, Tuple, Any
import numpy as np
import shutil
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Union, TYPE_CHECKING
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mimetypes
import os.path
from typing import Dict, Any
import time

if TYPE_CHECKING:
    from typing import Type

class ImageCaptionDataset(Dataset):
    def __init__(self, captions: Dict[str, str], processor):
        self.image_paths = list(captions.keys())
        self.captions = list(captions.values())
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        caption = self.captions[idx]

        # Process image and text
        encoding = self.processor(
            images=image,
            text=caption,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        return encoding

@dataclass
class SearchResult:
    """Data class for search results"""
    image_path: str
    similarity_score: float
    caption: str
    metadata: dict
    rank: int

class ImageProcessingPipeline:
    def __init__(self, persist_directory: str = "./chroma_db", image_folder: str = None):
        self.persist_directory = persist_directory
        self.image_folder = image_folder  # Store image_folder as instance variable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self._init_models()
        self.chroma_client = self._connect_to_chroma()

    def store_embeddings(self, image_embeddings: Dict[str, Any], collection_name: str = "image_embeddings"):
        """Store embeddings in ChromaDB with deduplication and relative paths"""
        try:
            if self.image_folder is None:
                raise ValueError("image_folder not provided to ImageProcessingPipeline")

            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            print(f"Created/accessed collection: {collection_name}")

            existing_hashes = set()
            existing_metadata = collection.get(include=["metadatas"])
            if existing_metadata and "metadatas" in existing_metadata:
                for metadata in existing_metadata["metadatas"]:
                    if "image_hash" in metadata:
                        existing_hashes.add(metadata["image_hash"])

            ids = []
            embeddings = []
            metadatas = []

            for image_path, embedding in image_embeddings.items():
                try:
                    # Generate image hash
                    image_hash = self._generate_image_hash(image_path)

                    if image_hash in existing_hashes:
                        print(f"Skipping duplicate image: {image_path}")
                        continue

                    # Store relative path instead of absolute
                    relative_path = os.path.relpath(image_path, self.image_folder)
                    mime_type = mimetypes.guess_type(image_path)[0]
                    
                    ids.append(str(uuid.uuid4()))
                    embeddings.append(embedding.tolist())
                    metadatas.append({
                        "image_path": relative_path,
                        "absolute_path": image_path,  # Store both paths for debugging
                        "image_hash": image_hash,
                        "mime_type": mime_type
                    })
                    
                    existing_hashes.add(image_hash)
                    print(f"Successfully processed {image_path}")
                    print(f"Relative path: {relative_path}")
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue

            # Batch processing
            if ids:  # Only process if we have valid entries
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    batch_end = min(i + batch_size, len(ids))
                    collection.add(
                        ids=ids[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        metadatas=metadatas[i:batch_end]
                    )
                print(f"Successfully stored {len(ids)} embeddings in ChromaDB")
            else:
                print("No valid embeddings to store")
            
        except Exception as e:
            print(f"Failed to store embeddings: {str(e)}")
            raise

    def _init_models(self):
        """Initialize all required models"""
        print("Loading pretrained models...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Use the same model for both text and image embeddings to ensure consistent dimensions
        self.text_embedder = SentenceTransformer('clip-ViT-B-32')  # Changed from 'all-MiniLM-L6-v2'
        self.image_embedder = SentenceTransformer('clip-ViT-B-32')

        self.caption_model = self.caption_model.to(self.device)
        self.text_embedder = self.text_embedder.to(self.device)
        self.image_embedder = self.image_embedder.to(self.device)
        print("Pretrained models loaded successfully!")

    def _connect_to_chroma(self) -> chromadb.Client:
        """Initialize ChromaDB client"""
        try:

            os.makedirs(self.persist_directory, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_directory)
            print("Successfully initialized ChromaDB")
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to initialize ChromaDB: {str(e)}")

    def _generate_image_hash(self, image_path: str) -> str:
        """Generate a hash for an image file"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            return hashlib.md5(image_bytes).hexdigest()

    def generate_captions(self, image_folder: str) -> Dict[str, str]:
        """Generate captions for all images in the folder"""
        captions = {}

        for root, _, files in os.walk(image_folder):
            folder_name = os.path.basename(root)
            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        image_path = os.path.join(root, file_name)
                        image = Image.open(image_path).convert('RGB')

                        hint = f"This is a {folder_name} plant."
                        print(f"Generating caption for {file_name} in {folder_name}")
                        
                        
                        inputs = self.processor(image, hint, return_tensors="pt").to(self.device)
                        out = self.caption_model.generate(**inputs)
                        caption = self.processor.decode(out[0], skip_special_tokens=True)

                        captions[image_path] = caption
                        print(f"Caption generated: {caption}")

                        # Delete unnecessary variables to free memory
                        del image, inputs, out
                    except Exception as e:
                        print(f"Error processing {file_name}: {str(e)}")
                        continue

        return captions

    def encode_data(self, captions: Dict[str, str], batch_size: int = 100) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image_embeddings = {}
        text_embeddings = {}

        image_paths = list(captions.keys())
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert('RGB')
                    image_embedding = self.image_embedder.encode(image)
                    image_embeddings[image_path] = image_embedding

                    caption = captions[image_path]
                    text_embedding = self.text_embedder.encode(caption)
                    text_embeddings[image_path] = text_embedding
                except Exception as e:
                    logger.error(f"Error encoding {image_path}: {str(e)}")
                    continue

        return image_embeddings, text_embeddings

    def fine_tune_blip(self, captions: Dict[str, str],
                    output_dir: str = "./fine_tuned_blip",
                    num_epochs: int = 10,
                    batch_size: int = 32,
                    learning_rate: float = 2e-5):
        """Fine-tune BLIP model on the generated captions"""
        print("Starting BLIP fine-tuning...")

        # Enhanced data augmentation with normalization
        from torchvision import transforms
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.Resize((224, 224)),  # Ensure consistent size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                                std=[0.229, 0.224, 0.225])
        ])

        # Modified dataset class with text normalization
        class AugmentedImageCaptionDataset(Dataset):
            def __init__(self, captions: Dict[str, str], processor, transform=None):
                self.image_paths = list(captions.keys())
                self.captions = list(captions.values())
                self.processor = processor
                self.transform = transform
                
                # Normalize text lengths
                self.max_length = 50  # Set maximum sequence length
                
                # Basic text normalization
                self.normalized_captions = []
                for caption in self.captions:
                    # Convert to lowercase
                    normalized = caption.lower()
                    # Remove extra whitespace
                    normalized = ' '.join(normalized.split())
                    # Remove special characters except basic punctuation
                    normalized = re.sub(r'[^a-z0-9\s.,!?-]', '', normalized)
                    self.normalized_captions.append(normalized)

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                # Load and normalize image
                image = Image.open(self.image_paths[idx]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                # Ensure the image tensor is in the range [0, 1]
                if isinstance(image, torch.Tensor):
                    image = torch.clamp(image, 0, 1)
                
                # Get normalized caption
                caption = self.normalized_captions[idx]

                # Process with BLIP processor
                encoding = self.processor(
                    images=image,
                    text=caption,
                    padding='max_length',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                )

                # Squeeze extra dimensions
                for k, v in encoding.items():
                    encoding[k] = v.squeeze()

                return encoding

        # Create dataset with augmentation
        dataset = AugmentedImageCaptionDataset(captions, self.processor, data_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Rest of your code remains the same
        from torch.optim import AdamW
        optimizer = AdamW(self.caption_model.parameters(), 
                        lr=learning_rate,
                        weight_decay=0.01,
                        betas=(0.9, 0.999))

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)

        self.caption_model.train()
        total_steps = len(dataloader) * num_epochs
        best_loss = float('inf')
        patience = 3
        patience_counter = 0

        # List to store loss values for plotting
        loss_values = []

        print(f"Training on {len(dataset)} image-caption pairs for {num_epochs} epochs")
        print(f"Total steps: {total_steps}")

        for epoch in range(num_epochs):
            total_loss = 0
            for step, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)

                outputs = self.caption_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=input_ids,
                )

                loss = outputs.loss
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.caption_model.parameters(), max_norm=1.0)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Append the loss value for plotting
                loss_values.append(loss.item())

                if step % 10 == 0:
                    print(f"Epoch: {epoch + 1}/{num_epochs} | Step: {step}/{len(dataloader)} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} completed | Average Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                os.makedirs(output_dir, exist_ok=True)
                self.caption_model.save_pretrained(output_dir)
                self.processor.save_pretrained(output_dir)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Plot the loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        loss_curve_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()

        print(f"Loss curve saved to {loss_curve_path}")

        print(f"Fine-tuned model saved to {output_dir}")

        print("\nTesting fine-tuned model on sample images:")
        test_images = list(captions.keys())[:5]
        for test_image_path in test_images:
            print(f"\nTesting image: {test_image_path}")
            test_image = Image.open(test_image_path).convert('RGB')
            inputs = self.processor(test_image, return_tensors="pt").to(self.device)

            generated_ids = self.caption_model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=5,
                temperature=0.7,
                repetition_penalty=1.2,
                length_penalty=1.0
            )
            generated_caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

            print(f"Original caption: {captions[test_image_path]}")
            print(f"Generated caption: {generated_caption}")

        return output_dir

    def search(self, query_image: str = None, query_text: str = None, top_k: int = 5) -> list:
        """
        Search for similar images using image and/or text query
        Returns top k matches with their generated captions
        """
        try:
            # Get query embeddings
            query_embedding = self._get_query_embedding(query_image, query_text)
            if query_embedding is None:
                raise ValueError("Please provide either an image or text query")

            # Search in ChromaDB
            collection = self.chroma_client.get_or_create_collection(name="image_embeddings")
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "distances"]
            )

            # Process results
            search_results = []
            for idx, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                image_path = metadata['image_path']
                similarity_score = 1 - (distance / 2)  # Convert distance to similarity score

                # Generate caption using fine-tuned BLIP
                generated_caption = self._generate_caption_for_image(image_path)

                search_results.append({
                    'rank': idx + 1,
                    'image_path': image_path,
                    'similarity_score': similarity_score,
                    'generated_caption': generated_caption
                })

            return search_results

        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

    def _get_query_embedding(self, query_image: str = None, query_text: str = None) -> np.ndarray:
        """
        Get embedding for the query (image or text)
        """
        if query_image:
            try:
                image = Image.open(query_image).convert('RGB')
                return self.image_embedder.encode(image)
            except Exception as e:
                raise Exception(f"Failed to process query image: {str(e)}")
        elif query_text:
            try:
                return self.text_embedder.encode(query_text)
            except Exception as e:
                raise Exception(f"Failed to process query text: {str(e)}")
        return None

    def _generate_caption_for_image(self, image_path: str) -> str:
        """
        Generate caption for an image using the fine-tuned BLIP model
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            generated_ids = self.caption_model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=5,
                temperature=0.7  # Add some randomness for more natural captions
            )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            raise Exception(f"Caption generation failed for {image_path}: {str(e)}")

    def get_retriever(self) -> 'ImageRetriever':
        """Get an instance of ImageRetriever"""
        return ImageRetriever(self)  # Only pass self  # Pass image_folder to ImageRetriever

class ImageRetriever:
    def __init__(self, pipeline: ImageProcessingPipeline):
        self.pipeline = pipeline
        self.collection = pipeline.chroma_client.get_collection(name="image_embeddings")
        print(f"Connected to existing collection with {self.collection.count()} entries")
        self.device = pipeline.device
        
    def get_visual_features(self, image_path: str) -> torch.Tensor:
        """Extract visual features from image"""
        try:
            image = Image.open(image_path).convert('RGB')
            with torch.no_grad():
                features = self.pipeline.image_embedder.encode(image)
            return torch.tensor(features).to(self.device)
        except Exception as e:
            raise Exception(f"Failed to extract visual features: {str(e)}")

    def get_text_features(self, text: str) -> torch.Tensor:
        """Extract text features from query"""
        try:
            with torch.no_grad():
                features = self.pipeline.text_embedder.encode(text)
            return torch.tensor(features).to(self.device)
        except Exception as e:
            raise Exception(f"Failed to extract text features: {str(e)}")

    def generate_enhanced_caption(self, image_path: str, query_text: Optional[str] = None) -> str:
        """Generate enhanced caption using fine-tuned BLIP and query context"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Base caption generation
            inputs = self.pipeline.processor(image, return_tensors="pt").to(self.device)
            base_caption_ids = self.pipeline.caption_model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=5,
                temperature=0.7
            )
            base_caption = self.pipeline.processor.decode(base_caption_ids[0], skip_special_tokens=True)
            
            # If there's a query text, enhance the caption with RAG
            if query_text:
                # Generate question-aware caption
                prompt = f"Question: {query_text}\nContext: {base_caption}\nDetailed description:"
                inputs = self.pipeline.processor(image, prompt, return_tensors="pt").to(self.device)
                enhanced_ids = self.pipeline.caption_model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=100,
                    num_beams=5,
                    temperature=0.7,
                    repetition_penalty=1.2
                )
                enhanced_caption = self.pipeline.processor.decode(enhanced_ids[0], skip_special_tokens=True)
                return enhanced_caption
            
            return base_caption
            
        except Exception as e:
            raise Exception(f"Caption generation failed: {str(e)}")

    def compute_similarity(self, query_features: torch.Tensor, target_features: torch.Tensor) -> float:
        """Compute similarity between query and target features"""
        query_features = F.normalize(query_features, p=2, dim=0)
        target_features = F.normalize(target_features, p=2, dim=0)
        return float(F.cosine_similarity(query_features, target_features, dim=0))

    def retrieve(self, query_image: Optional[str] = None, query_text: Optional[str] = None, top_k: int = 5) -> List[SearchResult]:
        try:
            # Get query features using existing methods
            if query_image and query_text:
                image_features = self.get_visual_features(query_image)
                text_features = self.get_text_features(query_text)
                query_features = (image_features * 0.7 + text_features * 0.3)  # Weighted combination
            elif query_image:
                query_features = self.get_visual_features(query_image)
            elif query_text:
                query_features = self.get_text_features(query_text)
            else:
                raise ValueError("Please provide either an image or text query")

            query_features = F.normalize(query_features, p=2, dim=0)  # Normalize for cosine similarity

            # Search in ChromaDB
            try:
                collection = self.pipeline.chroma_client.get_collection(name="image_embeddings")
            except Exception as e:
                print(f"Error accessing collection: {str(e)}")
                collection = self.pipeline.chroma_client.create_collection(name="image_embeddings")

            # Get collection count for debugging
            count = collection.count()
            print(f"Searching in collection with {count} entries")

            results = collection.query(
                query_embeddings=[query_features.cpu().numpy().tolist()],
                n_results=min(top_k, count),  # Ensure we don't request more than available
                include=["metadatas", "distances"]
            )

            # Process and enhance results
            search_results = []
            if results['metadatas'] and results['distances']:
                for idx, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                    try:
                        # Get relative path from metadata
                        relative_path = metadata.get('image_path')
                        if not relative_path:
                            print(f"Missing image_path in metadata for result {idx}")
                            continue

                        # Construct absolute path
                        absolute_path = os.path.join(self.pipeline.image_folder, relative_path)
                        
                        # Verify file exists
                        if not os.path.exists(absolute_path):
                            print(f"Image file not found: {absolute_path}")
                            continue

                        similarity_score = 1 - (distance / 2)

                        # Generate enhanced caption using existing method
                        enhanced_caption = self.generate_enhanced_caption(absolute_path, query_text)

                        # Prepare additional metadata
                        additional_metadata = {
                            'image_hash': metadata.get('image_hash', ''),
                            'query_type': 'image_text' if query_image and query_text else 'image' if query_image else 'text',
                            'absolute_path': absolute_path,
                            'relative_path': relative_path
                        }

                        # Create SearchResult object
                        result = SearchResult(
                            image_path=relative_path,  # Store relative path for URL generation
                            similarity_score=similarity_score,
                            caption=enhanced_caption,
                            metadata=additional_metadata,
                            rank=idx + 1
                        )

                        search_results.append(result)
                        print(f"Processed result {idx + 1}: {relative_path} (score: {similarity_score:.3f})")

                    except Exception as e:
                        print(f"Error processing result {idx}: {str(e)}")
                        continue

            if not search_results:
                print("No valid results found")

            return search_results

        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            raise Exception(f"Retrieval failed: {str(e)}")

def main():
    IMAGE_FOLDER = "BPLD Dataset" ## Image folder input here
    PERSIST_DIR = "./chroma_db"

    try:
        # Initialize pipeline
        pipeline = ImageProcessingPipeline(persist_directory=PERSIST_DIR)

        # Your existing pipeline steps
        captions = pipeline.generate_captions(IMAGE_FOLDER)
        fine_tuned_model_path = pipeline.fine_tune_blip(captions)
        image_embeddings, text_embeddings = pipeline.encode_data(captions)
        pipeline.store_embeddings(image_embeddings)

        # Initialize retriever
        retriever = pipeline.get_retriever()

        # Test case 1: Query with image only
        print("\nTest Case 1: Image Query")
        query_image = "BPLD Dataset/Healthy 220/4h.jpg" ##path of image u want to test 
        results = retriever.retrieve(query_image=query_image)
        for result in results:
            print(f"Rank: {result.rank}")
            print(f"Image: {result.image_path}")
            print(f"Similarity: {result.similarity_score:.4f}")
            print(f"Caption: {result.caption}\n")

        # Test case 2: Query with text only
        print("\nTest Case 2: Text Query")
        query_text = "Show me plants healthy"
        results = retriever.retrieve(query_text=query_text)
        for result in results:
            print(f"Rank: {result.rank}")
            print(f"Image: {result.image_path}")
            print(f"Similarity: {result.similarity_score:.4f}")
            print(f"Caption: {result.caption}\n")

        # Test case 3: Query with both image and text
        print("\nTest Case 3: Image + Text Query")
        query_text = "What disease symptoms are visible in this plant?"
        results = retriever.retrieve(query_image=query_image, query_text=query_text)
        for result in results:
            print(f"Rank: {result.rank}")
            print(f"Image: {result.image_path}")
            print(f"Similarity: {result.similarity_score:.4f}")
            print(f"Caption: {result.caption}\n")

        # Clean up
        del captions, image_embeddings, text_embeddings

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()