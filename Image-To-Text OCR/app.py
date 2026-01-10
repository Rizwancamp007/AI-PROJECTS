from flask import Flask, render_template, request
import os
from model import extract_text_from_image, character_error_rate, word_error_rate

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No file selected")

    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Upload JPG or PNG only.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        extracted_text = extract_text_from_image(filepath)

        if not extracted_text:
            extracted_text = "No readable text detected."

        reference_text = "This is sample reference text used for OCR evaluation."

        cer = character_error_rate(reference_text, extracted_text)
        wer = word_error_rate(reference_text, extracted_text)

        return render_template(
            'index.html',
            extracted_text=extracted_text,
            filename=file.filename,
            cer=round(cer, 2),
            wer=round(wer, 2)
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
