from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
from mindflix1 import ImageProcessingPipeline, ImageRetriever
import logging
import glob
import time
import traceback

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIR = "./chroma_db"
IMAGE_FOLDER = "BPLD Dataset"  ##### Input image folder here 

# Global variables
pipeline = None
retriever = None

def create_index_html():
    """Create the index.html template with improved styling and JavaScript"""
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Image Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .search-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        input[type="text"] {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            flex-grow: 1;
        }
        button {
            padding: 8px 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .results-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            padding: 20px 0;
        }
        .result-item {
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .result-item img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        .result-info {
            padding: 15px;
        }
        .result-info p {
            margin: 5px 0;
        }
        .error {
            color: red;
            padding: 10px;
            background-color: #ffebee;
            border-radius: 4px;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="search-container">
        <h1>Image Search</h1>
        <form id="searchForm" class="search-form">
            <input type="text" name="query_text" placeholder="Enter search text" required>
            <input type="file" name="query_image" accept="image/*">
            <button type="submit">Search</button>
        </form>
    </div>
    <div id="results" class="results-container"></div>

    <script>
        document.getElementById('searchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';
            
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Search request failed');
                }
                
                const results = await response.json();
                
                resultsDiv.innerHTML = '';
                
                if (results.error) {
                    resultsDiv.innerHTML = `<div class="error">${results.error}</div>`;
                    return;
                }
                
                if (!results.length) {
                    resultsDiv.innerHTML = '<div class="error">No results found</div>';
                    return;
                }
                
                results.forEach(result => {
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'result-item';
                    resultDiv.innerHTML = `
                        <img src="${result.image_url}" alt="${result.caption || 'Search result'}" 
                             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 1 1%22><rect width=%221%22 height=%221%22 fill=%22%23cccccc%22/></svg>'">
                        <div class="result-info">
                            <p><strong>Score:</strong> ${result.similarity_score.toFixed(2)}</p>
                            ${result.caption ? `<p><strong>Caption:</strong> ${result.caption}</p>` : ''}
                            <p><strong>Path:</strong> ${result.image_url}</p>
                        </div>
                    `;
                    resultsDiv.appendChild(resultDiv);
                });
            } catch (error) {
                console.error('Search error:', error);
                resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        });
    </script>
</body>
</html>
        """)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return f"Error: Make sure templates/index.html exists. Error: {str(e)}", 500

# Update these parts in your existing code

# Add this import at the top
import os.path

# Update the search route to handle paths correctly
@app.route('/search', methods=['POST'])
def search():
    if not retriever:
        logger.error("Retriever not initialized")
        return jsonify({'error': 'Search system not initialized'}), 500

    try:
        query_image = None
        query_text = request.form.get('query_text', '').strip()
        logger.info(f"Received search request with text: {query_text}")
        
        if 'query_image' in request.files:
            file = request.files['query_image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                query_image = filepath
                logger.info(f"Image uploaded: {query_image}")
        
        if not query_image and not query_text:
            return jsonify({'error': 'Please provide either an image or text query'}), 400

        top_k = min(max(int(request.form.get('top_k', 5)), 1), 20)
        logger.info(f"Processing search with params - Image: {query_image}, Text: {query_text}, Top K: {top_k}")
        
        results = retriever.retrieve(query_image=query_image, query_text=query_text, top_k=top_k)
        logger.info(f"Retrieved {len(results) if results else 0} results")
        
        if not results:
            return jsonify([]), 200

        results_json = []
        for idx, result in enumerate(results):
            try:
                # Get the image path from the result
                image_path = result.image_path if hasattr(result, 'image_path') else None
                logger.debug(f"Processing result {idx} with image path: {image_path}")
                
                if not image_path:
                    logger.warning(f"No image path for result {idx}")
                    continue
                
                # Construct the full path
                full_image_path = os.path.join(IMAGE_FOLDER, image_path)
                logger.debug(f"Full image path: {full_image_path}")
                
                if not os.path.exists(full_image_path):
                    logger.warning(f"Image file not found: {full_image_path}")
                    continue
                
                # Use the original path for the URL
                image_url = f'/images/{image_path}'
                
                result_dict = {
                    'image_url': image_url,
                    'similarity_score': float(result.similarity_score) if hasattr(result, 'similarity_score') else 0.0,
                    'caption': result.caption if hasattr(result, 'caption') else '',
                    'metadata': result.metadata if hasattr(result, 'metadata') else {},
                    'rank': idx + 1
                }
                results_json.append(result_dict)
                logger.debug(f"Added result: {result_dict}")
                
            except Exception as e:
                logger.error(f"Error processing result {idx}: {str(e)}")
                continue

        if query_image and os.path.exists(query_image):
            try:
                os.remove(query_image)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

        logger.info(f"Returning {len(results_json)} results")
        return jsonify(results_json), 200

    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Update the serve_image route to handle paths correctly
@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        # Construct the full path by joining IMAGE_FOLDER with the filename
        image_path = os.path.join(IMAGE_FOLDER, filename)
        logger.debug(f"Attempting to serve image: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return jsonify({'error': 'Image not found'}), 404
            
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

def initialize_serving():
    """Initialize the serving components without retraining"""
    global pipeline, retriever
    try:
        logger.info("Starting initialization of serving pipeline...")
        
        # Check directory existence
        logger.debug(f"Checking ChromaDB directory: {PERSIST_DIR}")
        if not os.path.exists(PERSIST_DIR):
            logger.error(f"ChromaDB directory not found at {PERSIST_DIR}")
            return False
            
        logger.debug(f"Checking Image folder: {IMAGE_FOLDER}")
        if not os.path.exists(IMAGE_FOLDER):
            logger.error(f"Image folder not found at {IMAGE_FOLDER}")
            return False
            
        # Count available images
        image_count = len(glob.glob(os.path.join(IMAGE_FOLDER, '**/*.jpg'), recursive=True))
        logger.info(f"Found {image_count} images in the dataset")
        
        # Initialize the pipeline
        try:
            logger.info("Initializing ImageProcessingPipeline...")
            pipeline = ImageProcessingPipeline(
                persist_directory=PERSIST_DIR,
                image_folder=IMAGE_FOLDER
            )
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        # Initialize the retriever
        try:
            logger.info("Initializing retriever...")
            retriever = pipeline.get_retriever()
            if retriever is None:
                logger.error("Retriever initialization returned None")
                return False
            logger.info("Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
        # Test the retriever
        try:
            logger.info("Testing retriever with dummy query...")
            test_results = retriever.retrieve(query_text="test", top_k=1)
            logger.debug(f"Test retrieval results: {test_results}")
            if not test_results:
                logger.warning("Test retrieval returned no results")
        except Exception as e:
            logger.error(f"Test retrieval failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("Serving pipeline initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize serving pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def init_app():
    """Initialize the Flask application and all required components"""
    try:
        logger.info("Starting Flask application initialization...")
        
        # Create required directories
        logger.debug("Creating required directories...")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        # Create index.html if it doesn't exist
        template_path = os.path.join('templates', 'index.html')
        if not os.path.exists(template_path):
            logger.info("Creating index.html template...")
            create_index_html()  # This is the function that creates the template
            
        # Initialize the serving pipeline
        logger.info("Initializing serving pipeline...")
        if not initialize_serving():
            logger.error("Failed to initialize serving pipeline")
            return False
            
        # Verify uploads directory permissions
        upload_dir = app.config['UPLOAD_FOLDER']
        try:
            test_file = os.path.join(upload_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info("Uploads directory permissions verified")
        except Exception as e:
            logger.error(f"Upload directory permission test failed: {str(e)}")
            return False
            
        # Test database connection
        try:
            logger.info("Testing ChromaDB connection...")
            if not os.path.exists(os.path.join(PERSIST_DIR, 'chroma.sqlite3')):
                logger.error("ChromaDB database file not found")
                return False
        except Exception as e:
            logger.error(f"ChromaDB connection test failed: {str(e)}")
            return False
            
        logger.info("Flask application initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Flask application: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    try:
        # Initialize everything
        if init_app():
            logger.info("Starting Flask server...")
            # Create required directories
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs('templates', exist_ok=True)
            
            # Start the Flask server
            app.run(host='127.0.0.1', port=5025, debug=False)
        else:
            logger.error("Application initialization failed. Please check the logs and try again.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)