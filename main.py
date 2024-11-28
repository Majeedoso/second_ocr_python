import os
import cv2
import pytesseract
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Allowed file extensions and default upload folder
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# A dictionary to store OCR results by task ID
ocr_results = {}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Format dates (e.g., YYYYMMDD to YYYY/MM/DD)
def format_date(date_str):
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
    return date_str

# Improved OCR processing function in the background using ThreadPoolExecutor
def run_ocr_in_background(file_path, task_id):
    # Read the image
    img = cv2.imread(file_path)
    if img is None:
        ocr_results[task_id] = {'error': 'Failed to read image. Please upload a valid image.'}
        return

    # Convert image to grayscale for better OCR performance
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image: Reduce size by resizing to a smaller width
    img_resized = cv2.resize(img_gray, (100, int(100 * img.shape[0] / img.shape[1])))  # Smaller width (600)

    # Optionally: Crop to a region of interest (ROI) if you know where the text is located
    roi = img_resized[100:500, 100:500]  # Crop the region of interest (adjust accordingly)

    # Perform OCR using Tesseract
    try:
        custom_config = r'--oem 3 --psm 6'  # OEM and PSM configuration for Tesseract
        text_results = pytesseract.image_to_string(roi, config=custom_config, lang='ara')  # Arabic language
    except Exception as e:
        ocr_results[task_id] = {'error': f'OCR processing failed: {str(e)}'}
        return

    # Process OCR results
    lines_with_numbers = []
    lines_with_strings = []

    filter_phrases = [
        "Rh:", "بطاقة", "الديمقراطية", "الجمهورية", "سلطة", "تاررخ", "التعريف",
        "اللقب", "بلدية", "تاريخ", ":", "الجنس", "ائرية", "الإسم", "مكان"
    ]

    # Split text by lines and filter based on the criteria
    for line in text_results.split('\n'):
        line = line.strip()  # Remove any extra spaces or newlines
        # Detect numeric lines
        if any(char.isdigit() for char in line):
            numbers_in_line = ''.join(char for char in line if char.isdigit())
            if len(numbers_in_line) == 18:  # Example: ID number
                lines_with_numbers.append(numbers_in_line)
            elif len(numbers_in_line) >= 8:  # Example: Date
                lines_with_numbers.append(format_date(numbers_in_line))
        # Detect meaningful text lines
        elif len(line) >= 3 and not any(phrase in line for phrase in filter_phrases):
            if len(line) > 3 or line == "ذكر":  # Arabic-specific rule
                lines_with_strings.append(line)

    # Save the OCR results
    ocr_results[task_id] = {
        'lines_with_numbers': lines_with_numbers,
        'lines_with_strings': lines_with_strings
    }

@app.route('/ocr', methods=['POST'])
def ocr():
    # Check if file is present in the request
    if 'file' not in request.files or not request.files['file']:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only jpg, jpeg, and png allowed.'}), 400

    # Securely save the file
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Generate a unique task ID
    task_id = str(len(ocr_results) + 1)

    # Use ThreadPoolExecutor for background processing (better management of multiple threads)
    executor = ThreadPoolExecutor(max_workers=1)  # Only 1 worker for OCR to avoid overloading CPU
    executor.submit(run_ocr_in_background, file_path, task_id)

    # Respond with the task ID immediately
    return jsonify({'message': 'OCR is processing in the background', 'task_id': task_id}), 202

@app.route('/ocr/status/<task_id>', methods=['GET'])
def get_status(task_id):
    # Check if the OCR task is completed
    result = ocr_results.get(task_id)
    
    if result is None:
        return jsonify({'error': 'Task not found'}), 404
    
    if 'error' in result:
        return jsonify({'error': result['error']}), 400

    # Return the OCR results
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=port, debug=True)
