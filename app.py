'''
import os
import json
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from model import predict

app = Flask(__name__)
UPLOAD_FOLDER = 'pipline/uploads'
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

    # 使用固定目录：static/uploads
    folder_path = app.config['UPLOAD_FOLDER']
    os.makedirs(folder_path, exist_ok=True)

    results = []

    image_files = request.files.getlist('images')
    folder_files = request.files.getlist('folder[]')

    json_data = {}

    # 处理上传的文件夹（包含图片和 JSON）
    if folder_files and any(f.filename.endswith('.json') for f in folder_files):
        for file in folder_files:
            filename = secure_filename(file.filename)
            save_path = os.path.join(folder_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)

            if filename.endswith('.json'):
                with open(save_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

        for item in json_data.get('images', []):
            rel_path = item['path']
            filename = os.path.basename(rel_path)                   # 获取文件名，如 image1.jpg
            new_filename = f"upload_{filename}"
            image_path = os.path.join(folder_path, new_filename)
            if not os.path.exists(image_path):
                continue
            
            description = item.get('description', 'N/A')
            prediction = predict(image_path, description)
            results.append({
                'image': new_filename.replace("\\", "/"),
                'description': description,
                'prediction': prediction
            })

    # 处理上传的图片 + 文本
    elif image_files:
        for file in image_files:
            filename = secure_filename(file.filename)
            save_path = os.path.join(folder_path, filename)
            file.save(save_path)
            prediction = predict(save_path, user_text or "N/A")
            results.append({
                'image': filename,
                'description': user_text or "N/A",
                'prediction': prediction
            })

    else:
        return "No files uploaded.", 400

    # 保存结果
    result_data = {
        'results': results,
        'image_method': image_method,
        'text_method': text_method,
        'joint_method': joint_method,
        'model_selected': model_selected
    }

    with open(os.path.join(folder_path, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 重定向到结果页（不带 ID）
    return redirect(url_for('show_result'))


@app.route('/result')
def show_result():
    folder_path = app.config['UPLOAD_FOLDER']
    json_path = os.path.join(folder_path, 'result.json')

    if not os.path.exists(json_path):
        return "⚠️ 结果未找到", 404

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return render_template(
        'result.html',
        results=data['results'],
        folder='',  # 图片路径使用 static/uploads/
        image_method=data['image_method'],
        text_method=data['text_method'],
        joint_method=data['joint_method'],
        model_selected=data['model_selected']
    )


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
'''
import os
import json
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pipline.pipline_pure import main

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

    folder_path = app.config['UPLOAD_FOLDER']
    os.makedirs(folder_path, exist_ok=True)

    results = []

    image_files = request.files.getlist('images')
    folder_files = request.files.getlist('folder[]')

    uploaded_json_path = None

    # 处理上传的文件夹（包含图片和 JSON）
    if folder_files and any(f.filename.endswith('.json') for f in folder_files):
        for file in folder_files:
            filename = secure_filename(os.path.basename(file.filename))
            filename = f"upload_{filename}"
            save_path = os.path.join(folder_path, filename)
            file.save(save_path)
            if file.filename.endswith('.json'):
                uploaded_json_path = save_path

    # 处理上传的图片 + 文本
    elif image_files:
        for file in image_files:
            filename = secure_filename(os.path.basename(file.filename))
            save_path = os.path.join(folder_path, filename)
            file.save(save_path)

        # 构造 JSON 数据
        
        uploaded_json_path = os.path.join(folder_path, "upload_info.json")
        '''
        with open(uploaded_json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"question": user_text,
                "image_path": save_path
                } for file in image_files
            ], f, ensure_ascii=False, indent=2)
        '''

            # 构造单个 JSON 对象
        json_obj = {"question": user_text,"image_path": filename}

            # 写入 JSON 文件（单对象，不是数组）
        
        with open(uploaded_json_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False)
    else:
        return "No files uploaded.", 400

    # 调用主检测脚本（main 函数）
    main(data_file=uploaded_json_path, 
         VLmodel_name=model_selected, 
         image_detector= image_method, 
         text_detector=text_method, 
         image_text_detector=joint_method
         )

    result_json_path = os.path.join(folder_path, 'result.json')
    if not os.path.exists(result_json_path):
        return "⚠️ 检测未生成结果文件", 500

    return redirect(url_for('show_result'))

@app.route('/result')
def show_result():
    folder_path = app.config['UPLOAD_FOLDER']
    json_path = os.path.join(folder_path, 'result.json')

    if not os.path.exists(json_path):
        return "⚠️ 结果未找到", 404

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return render_template(
        'result.html',
        results=data['results'],
        folder='',
        image_method=data['image_method'],
        text_method=data['text_method'],
        joint_method=data['joint_method'],
        model_selected=data['model_selected']
    )

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
