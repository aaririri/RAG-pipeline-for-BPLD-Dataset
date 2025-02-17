# RAG-pipeline-for-BPLD-Dataset

How to run the code - 

Install dependencies:
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.23.0 Pillow>=9.0.0 transformers>=4.30.0 sentence-transformers>=2.2.2 chromadb>=0.4.0 tqdm>=4.65.0 flask>=2.0.0

or install using: pip install -r requirements.txt

In mindflix1.py:
input downloaded image folder in {IMAGE_FOLDER}  (Sample input is already present for BPLD Dataset)
For a test image put the image file path in {query_image}

Run - 
1. train.py 
2. app.py

(Downloaded data to be input at IMAGE_FOLDER in train.py and app.py)
(for our particular dataset I have used BPLD dataset)

Example User Queries
"Find images of Blackgram leaves affected by Yellow Mosaic disease."
"Show similar images to this uploaded picture of a Blackgram leaf."
"Given this image, what disease does the leaf have?"
"Retrieve images of healthy Blackgram leaves for comparison."
