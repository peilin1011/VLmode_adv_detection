import os
import json
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from pipline.pipline_pure import main
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_method = request.form.get('image_method')
    text_method = request.form.get('text_method')
    joint_method = request.form.get('joint_method')
    model_selected = request.form.get('model')
    user_text = request.form.get('text_input')
    mode = request.form.get('mode')

    folder_path = app.config['UPLOAD_FOLDER']
    os.makedirs(folder_path, exist_ok=True)

    results = []

    image_files = request.files.getlist('images')
    folder_files = request.files.getlist('folder[]')

    uploaded_json_path = None

    # Handle uploaded folder (which includes images and a JSON file)
    if folder_files and any(f.filename.endswith('.json') for f in folder_files):
        for file in folder_files:
            filename = secure_filename(os.path.basename(file.filename))
            #filename = f"upload_{filename}"
            save_path = os.path.join(folder_path, filename)
            file.save(save_path)
            if file.filename.endswith('.json'):
                uploaded_json_path = save_path

    # Handle uploaded images + user text input
    elif image_files:
        for file in image_files:
            filename = secure_filename(os.path.basename(file.filename))
            filename = f"upload_{filename}"
            save_path = os.path.join(folder_path, filename)
            file.save(save_path)

        # Construct a JSON object to describe the input
        uploaded_json_path = os.path.join(folder_path, "upload_info.json")
        json_obj = {"question": user_text, "image_path": filename}
        with open(uploaded_json_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False)
    else:
        return "No files uploaded.", 400

    
    # Call the main processing function
    main(data_file=uploaded_json_path, 
         VLmodel_name=model_selected, 
         image_detector=image_method, 
         text_detector=text_method, 
         image_text_detector=joint_method,
         mode=mode
         )

    result_json_path = os.path.join(folder_path, 'result.json')
    if not os.path.exists(result_json_path):
        return "⚠️ Detection did not generate a result file", 500
    if mode == 'inference':
        return redirect(url_for('show_result_inference'))
    elif mode == 'test':
        return redirect(url_for('show_result_test'))


@app.route('/result_inference')
def show_result_inference():
    folder_path = app.config['UPLOAD_FOLDER']
    json_path = os.path.join(folder_path, 'result.json')

    if not os.path.exists(json_path):
        return "⚠️ Result not found", 404

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return render_template(
        'result_inference.html',
        results=data['results'],
        folder='',
        image_method=data['image_method'],
        text_method=data['text_method'],
        joint_method=data['joint_method'],
        model_selected=data['model_selected']
    )

@app.route('/result_test')
def show_result_test():
    folder_path = app.config['UPLOAD_FOLDER']
    json_path = os.path.join(folder_path, 'result.json')

    if not os.path.exists(json_path):
        return "⚠️ Result not found", 404

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return render_template(
        'result_test.html',
        results=data['results'],
        folder='',
        image_method=data['image_method'],
        text_method=data['text_method'],
        joint_method=data['joint_method'],
        model_selected=data['model_selected'],
        precision=data['precision'],
        recall=data['recall']

    )
from io import BytesIO
import csv

@app.route('/download_csv')
def download_csv():
    folder_path = app.config['UPLOAD_FOLDER']
    json_path = os.path.join(folder_path, 'result.json')

    if not os.path.exists(json_path):
        return "⚠️ Result not found", 404

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

 
    output = BytesIO()

    csv_str = "Image Path,Question,Prediction,Processing Time\n"
    for item in data['results']:
        csv_str += f"{item.get('image_path','')},{item.get('question','')},{item.get('prediction','')},{item.get('processing_time','')}\n"
    

    output.write(csv_str.encode('utf-8'))
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='results.csv'
    )


if __name__ == '__main__':
    print(f'configing...')
    url = "https://huggingface.co/peilin1011/vqa/resolve/main/model_vqa.pth"

    save_dir = "pipline/model"
    os.makedirs(save_dir, exist_ok=True)  
    save_path = os.path.join(save_dir, "model_vqa.pth")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"model is downloaded: {save_path}")
    else:
        print(f"download model fails, error: {response.status_code}")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
