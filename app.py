from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# 載入 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 圖片預處理函式
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ResNet50 的預處理
    return img_array.astype(np.float32)  # 必須轉成 float32

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img_file = request.files["file"]
    path = os.path.join("temp", img_file.filename)
    os.makedirs("temp", exist_ok=True)
    img_file.save(path)

    try:
        img_array = preprocess(path)

        # 設定輸入與執行模型
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        print("Model prediction (raw):", pred)

        label = int(np.argmax(pred, axis=1)[0])
        prob = float(pred[0][label])

        result = "Dog" if label == 1 else "Cat"
        print(f"預測類別為: {result}，機率: {prob:.4f}")
        return jsonify({"label": result, "probability": prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 預設為 5000，但 Render 會給 PORT
    app.run(host="0.0.0.0", port=port)