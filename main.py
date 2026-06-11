import os
import io
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

import tensorflow as tf

from xray_validation import decode_imagefile_to_bgr



# =======================
# CONFIG
# =======================
APP_TITLE = "X-Ray Classification API (TFLite)"
MODEL_PATH = "model/model.tflite"
IMG_SIZE = (224, 224)

CLASS_NAMES = ['Normal', 'pneumonia', 'pneumothorax', 'tb']

# ✅ Correct File ID from your Google Drive link (Layer 3 / main pathology model)
FILE_ID = "1Cwc_bXCbLJpaKcFHbIzPuww27rxUXI1B"

# ✅ Correct File ID from your Google Drive link (Layer 2 / xray gate model)
# URL: https://drive.google.com/file/d/1D4w4UwEISKyp_r3Tn3siBP5f2L20_e-J/view?usp=drive_link
GATE_FILE_ID = "1D4w4UwEISKyp_r3Tn3siBP5f2L20_e-J"

GATE_MODEL_PATH = "model/model1.tflite"
GATE_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "0.5"))

app = FastAPI(title=APP_TITLE)


interpreter = None
input_details = None
output_details = None

gate_interpreter = None
gate_input_details = None
gate_output_details = None



# =======================
# GOOGLE DRIVE DOWNLOADER
# =======================
def download_model():
    os.makedirs("model", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("✅ TFLite Model already exists")
        return

    print("📥 Downloading TFLite model from Google Drive...")

    try:
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        print("✅ TFLite Model downloaded successfully")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise


# =======================
# APP STARTUP
# =======================
@app.on_event("startup")
def load_model():
    global interpreter, input_details, output_details
    global gate_interpreter, gate_input_details, gate_output_details

    download_model()

    # Main 4-class model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Layer-2 gate model (X-ray vs non-X-ray)
    if not os.path.exists(GATE_MODEL_PATH):
        # Download gate model lazily if missing
        os.makedirs(os.path.dirname(GATE_MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?export=download&id={GATE_FILE_ID}"
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(GATE_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    gate_interpreter = tf.lite.Interpreter(model_path=GATE_MODEL_PATH)
    gate_interpreter.allocate_tensors()
    gate_input_details = gate_interpreter.get_input_details()
    gate_output_details = gate_interpreter.get_output_details()

    print("🚀 TFLite interpreters loaded and ready")



# =======================
# ROUTES
# =======================
@app.get("/health")
def health():
    return {"status": "ok", "engine": "tflite_runtime"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image")

    # Read image bytes
    image_bytes = await file.read()


    # decode step kept minimal; layer 1 (grayscale/color validation) skipped
    try:
        _ = decode_imagefile_to_bgr(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")






    # Preprocess for TFLite model (PIL path)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)


    # Layer-2 gate inference (X-ray vs non-X-ray)
    if gate_interpreter is None:
        raise HTTPException(status_code=503, detail="Gate model not loaded")

    gate_interpreter.set_tensor(gate_input_details[0]["index"], x)
    gate_interpreter.invoke()

    gate_preds = gate_interpreter.get_tensor(gate_output_details[0]["index"])[0]
    # Supports either single sigmoid/logit (shape [...]) or 2-class softmax ([0]=non_xray, [1]=xray)
    if isinstance(gate_preds, np.ndarray) and gate_preds.ndim >= 1 and gate_preds.shape[0] == 2:
        # Assume [non_xray, xray]
        gate_conf = float(gate_preds[1])
        gate_label = "xray" if gate_conf >= GATE_THRESHOLD else "non_xray"
    else:
        gate_conf = float(np.ravel(gate_preds)[0])
        gate_label = "xray" if gate_conf >= GATE_THRESHOLD else "non_xray"

    if gate_label != "xray":
        raise HTTPException(
            status_code=422,
            detail={"reason": "Rejected by xray gate model", "gate_confidence": gate_conf},
        )

    # Layer-3 main pathology inference
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    best_idx = int(np.argmax(preds))


    return {
        "scores": preds.tolist(),
        "predicted_index": best_idx,
        "predicted_label": CLASS_NAMES[best_idx],
        "confidence": float(preds[best_idx]),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)