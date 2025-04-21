from flask import *
from werkzeug.utils import secure_filename

import os
import cv2

import image_fuzzy_clustering as fem
import record_video
import label_image

from PIL import Image

import secrets
from flask import url_for, current_app
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np




def load_image(image):
    text = label_image.main(image)
    return text

def save_img(img, filename):
    picture_path = os.path.join('static/images', filename)
    i = Image.open(img)
    i.save(picture_path)
    return picture_path

def process(image_path):
    print("[INFO] Performing image clustering...")
    fem.plot_cluster_img(image_path, 3)
    print("[INFO] Clustering completed.")

    clustered_path = 'static/images/orig_image.jpg'  # Assuming this is where clustering saves
    result = load_image(clustered_path)

    return result if result else "Prediction failed"






def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

app = Flask(__name__)
model = None

app.secret_key = 'Surya778@'

UPLOAD_FOLDER = os.path.join(app.root_path ,'static','uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')
    
@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/output')
def output():
    vitamin = session.get('vitamin')
    return render_template('output.html', vitamin=vitamin)

@app.route('/upload')
def upload():
    return render_template('index1.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        original_filename = secure_filename(f.filename)
        original_path = save_img(f, original_filename)

        result = process(original_path)
        print("Raw result:", result)

        if not result or result.strip().lower() == "prediction failed":
            return jsonify({'error': 'Prediction failed'}), 500

        result = result.strip().title()

        # Normalize prediction to match output.html conditions
        vitamin_names = ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin B12', 'Vitamin E']
        matched_vitamin = next((v for v in vitamin_names if v.lower() in result.lower()), None)
        print("Matched Vitamin to display:", matched_vitamin)

        # Cleanup
        if os.path.exists(original_path):
            os.remove(original_path)
        else:
            print(f"Warning: {original_path} not found, skipping deletion.")

        # Store matched vitamin in session
        session['vitamin'] = matched_vitamin

        # Render template with correct variable
        return render_template('output.html', vitamin=matched_vitamin)

    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        video = request.files['file']
        filename = secure_filename(video.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(video_path)
        print(f"[INFO] Video saved at {video_path}")

        # Extract best face
        from video_detect import detect_best_face
        print("[INFO] Starting face detection...")
        best_face_path = detect_best_face(video_path)

        if not best_face_path or not os.path.exists(best_face_path):
            return jsonify({'error': 'No face detected'}), 400

        print(f"[INFO] Face detected and saved at {best_face_path}")
        detected_path ="static/Detected/detected.jpg"
        if os.path.exists(detected_path):
            result = process(detected_path)
            print("Raw result:", result)

            if not result or result == "Prediction failed":
                return jsonify({'error': 'Prediction failed'}), 500
        
            result = result.strip().title()

            vitamin_names = ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin B12', 'Vitamin E']
            matched_vitamin = next((v for v in vitamin_names if v.lower() in result.lower()), None)
            print("Matched Vitamin to display:", matched_vitamin)

            # Cleanup
            if os.path.exists(detected_path): 
                os.remove(detected_path)
            else:
                print(f"Warning: {detected_path} not found, skipping deletion.")
            session['vitamin'] = matched_vitamin

            # Render template with correct variable
            return render_template('output.html', vitamin=matched_vitamin)
        else:
            return "Error: Detected image not found."
    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500
        


@app.route('/record_video', methods=['GET','POST'])
def record_video_route():
    try:
        from record_video import record
        from video_detect import detect_best_face

        # Step 1: Record video from webcam
        print("[INFO] Starting video recording...")
        video_path = record()

        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video recording failed'}), 500

        print(f"[INFO] Video recorded and saved at {video_path}")

        # Step 2: Extract best face from video
        print("[INFO] Starting face detection...")
        best_face_path = detect_best_face(video_path)

        if not best_face_path or not os.path.exists(best_face_path):
            return jsonify({'error': 'No face detected from video'}), 400

        print(f"[INFO] Face detected and saved at {best_face_path}")
        
        detected_path ="static/Detected/detected.jpg"
        if os.path.exists(detected_path):
            result = process(detected_path)
            print("Raw result:", result)

            if not result or result == "Prediction failed":
                return jsonify({'error': 'Prediction failed'}), 500
        
            result = result.strip().title()

            vitamin_names = ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin B12', 'Vitamin E']
            matched_vitamin = next((v for v in vitamin_names if v.lower() in result.lower()), None)
            print("Matched Vitamin to display:", matched_vitamin)

            # Cleanup
            if os.path.exists(detected_path): 
                os.remove(detected_path)
            else:
                print(f"Warning: {detected_path} not found, skipping deletion.")

            # Store matched vitamin in session
            session['vitamin'] = matched_vitamin

            # Render template with correct variable
            return render_template('output.html', vitamin=matched_vitamin)
        else:
            return "Error: Detected image not found."
            
    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/record_live_image', methods=['GET', 'POST'])
def record_live_image_route():
    try:
        from record_video import record_live_image  # Function to capture live image using webcam

        # Step 1: Capture image from the webcam
        print("[INFO] Starting live image capture...")
        image_path = record_live_image()

        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Live image capture failed'}), 500

        print(f"[INFO] Live image captured and saved at {image_path}")

        # Step 2: Process the captured image for clustering and prediction
        result = process(image_path)
        print("Raw result:", result)

        if not result or result == "Prediction failed":
            return jsonify({'error': 'Prediction failed'}), 500
        
        result = result.strip().title()

        vitamin_names = ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin B12', 'Vitamin E']
        matched_vitamin = next((v for v in vitamin_names if v.lower() in result.lower()), None)
        print("Matched Vitamin to display:", matched_vitamin)

        if os.path.exists(image_path): 
            os.remove(image_path)
        else:
            print(f"Warning: {image_path} not found, skipping deletion.")
        # return result
        #return f"Prediction: {result}"
        # Store matched vitamin in session
        session['vitamin'] = matched_vitamin

        # Render template with correct variable
        return render_template('output.html', vitamin=matched_vitamin)
    
    except Exception as e:
        print(f"[ERROR] /record_live_image failed: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/capture_image', methods=['GET', 'POST'])
def capture_image():
    try:
        from record_video import capture_image  # Function to capture live image using webcam

        # Step 1: Capture image from the webcam
        print("[INFO] Starting live image capture...")
        image_path = capture_image()

        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Live image capture failed'}), 500

        print(f"[INFO] Live image captured and saved at {image_path}")

        # Step 2: Process the captured image for clustering and prediction
        result = process(image_path)
        print("Raw result:", result)

        if not result or result == "Prediction failed":
            return jsonify({'error': 'Prediction failed'}), 500
        
        result = result.strip().title()

        # Normalize prediction to match output.html conditions
        vitamin_names = ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin B12', 'Vitamin E']
        matched_vitamin = next((v for v in vitamin_names if v.lower() in result.lower()), None)
        print("Matched Vitamin to display:", matched_vitamin)

        # Cleanup
        if os.path.exists(image_path): 
            os.remove(image_path)
        else:
            print(f"Warning: {image_path} not found, skipping deletion.")

        # Store matched vitamin in session
        session['vitamin'] = matched_vitamin

        # Render template with correct variable
        return render_template('output.html', vitamin=matched_vitamin)

    except Exception as e:
        print(f"[ERROR] /record_live_image failed: {e}")
        return jsonify({'error': str(e)}), 500
    


    
    
if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run()