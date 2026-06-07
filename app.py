from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import requests
import base64
import re
import json as _json

app = Flask(__name__)
CORS(app)

model  = pickle.load(open('stroke_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl',       'rb'))

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.3-70b-versatile"

EXTRACT_PROMPT = """You are a medical data extraction assistant.
Extract the following fields from this medical lab report:
- age (integer)
- gender (exactly "Male" or "Female")
- bmi (float, body mass index)
- avg_glucose_level (float, blood glucose mg/dL — fasting or average)
- hypertension (1 if diagnosed/mentioned, else 0)
- heart_disease (1 if diagnosed/mentioned, else 0)
- smoking_status (one of exactly: "never smoked" / "formerly smoked" / "smokes" / "Unknown")

Rules:
- Return ONLY a valid JSON object, no extra text, no markdown, no explanation.
- If a value is not found, use null.
- Example: {"age": 55, "gender": "Male", "bmi": 27.5, "avg_glucose_level": 105.0, "hypertension": 1, "heart_disease": 0, "smoking_status": "never smoked"}
"""

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data    = request.json
        gender  = 1 if data['gender'] == 'Male' else 0
        age     = float(data['age'])
        hyper   = int(data['hypertension'])
        heart   = int(data['heart_disease'])
        glucose = float(data['avg_glucose_level'])
        bmi     = float(data['bmi'])
        sm      = data['smoking_status']
        s_formerly = 1 if sm == 'formerly smoked' else 0
        s_never    = 1 if sm == 'never smoked'    else 0
        s_smokes   = 1 if sm == 'smokes'          else 0

        features        = np.array([[gender, age, hyper, heart, glucose, bmi, s_formerly, s_never, s_smokes]])
        features_scaled = scaler.transform(features)
        probability     = model.predict_proba(features_scaled)[0][1]

        if probability < 0.3:   risk_level = 'Low'
        elif probability < 0.6: risk_level = 'Medium'
        else:                   risk_level = 'High'

        return jsonify({'prediction': int(probability >= 0.5),
                        'probability': round(float(probability) * 100, 1),
                        'risk_level': risk_level})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        lang         = request.json.get('lang', 'en')
        system_prompt = (
            "You are a medical assistant specialized in stroke and cardiovascular diseases. "
            "Answer in Arabic. Remind that answers are general info, not a substitute for a doctor."
            if lang == 'ar' else
            "You are a medical assistant specialized in stroke and cardiovascular diseases. "
            "Answer clearly in English. Remind that answers are general info, not a substitute for a doctor."
        )
        headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'}
        payload = {"model": TEXT_MODEL,
                   "messages": [{"role": "system", "content": system_prompt},
                                 {"role": "user",   "content": user_message}],
                   "max_tokens": 600}
        response = requests.post('https://api.groq.com/openai/v1/chat/completions',
                                 headers=headers, json=payload, timeout=30)
        result = response.json()
        print("GROQ CHAT:", result)
        if 'choices' not in result:
            return jsonify({'reply': 'Groq error: ' + result.get('error', {}).get('message', str(result))})
        return jsonify({'reply': result['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def call_groq_vision(b64_img, mime):
    """Send an image to Groq vision and return extracted JSON dict or None."""
    headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
            {"type": "text", "text": EXTRACT_PROMPT}
        ]}],
        "max_tokens": 500
    }
    r = requests.post('https://api.groq.com/openai/v1/chat/completions',
                      headers=headers, json=payload, timeout=40)
    result = r.json()
    print("GROQ VISION:", result)
    if 'choices' not in result:
        return None, result.get('error', {}).get('message', 'Vision error')
    content = result['choices'][0]['message']['content']
    # Try to extract JSON
    match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            return _json.loads(match.group()), None
        except Exception:
            pass
    return None, content


def call_groq_text(text_content):
    """Send text to Groq and return extracted JSON dict or None."""
    headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        "model": TEXT_MODEL,
        "messages": [{"role": "user", "content": EXTRACT_PROMPT + "\n\nReport text:\n" + text_content[:6000]}],
        "max_tokens": 500
    }
    r = requests.post('https://api.groq.com/openai/v1/chat/completions',
                      headers=headers, json=payload, timeout=40)
    result = r.json()
    print("GROQ TEXT:", result)
    if 'choices' not in result:
        return None, result.get('error', {}).get('message', 'Text error')
    content = result['choices'][0]['message']['content']
    match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            return _json.loads(match.group()), None
        except Exception:
            pass
    return None, content


@app.route('/extract', methods=['POST'])
def extract():
    try:
        file     = request.files.get('file')
        filetype = request.form.get('type', 'image')

        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        file_bytes = file.read()

        # ── IMAGE ──────────────────────────────────────────
        if filetype == 'image':
            mime    = file.content_type or 'image/jpeg'
            b64_img = base64.b64encode(file_bytes).decode('utf-8')
            data, err = call_groq_vision(b64_img, mime)
            if data:
                return jsonify({'success': True, 'data': data})
            return jsonify({'success': False, 'raw': str(err)})

        # ── PDF ────────────────────────────────────────────
        else:
            # Strategy 1: extract text with pdfplumber
            pdf_text = ""
            try:
                import pdfplumber, io
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        pdf_text += (page.extract_text() or "") + "\n"
            except Exception as e:
                print("pdfplumber error:", e)

            # If we got real text, use text model
            if pdf_text.strip() and len(pdf_text.strip()) > 30:
                data, err = call_groq_text(pdf_text)
                if data:
                    return jsonify({'success': True, 'data': data, 'method': 'text'})

            # Strategy 2: convert PDF pages to images → vision model
            # (works for scanned PDFs)
            try:
                import fitz  # PyMuPDF
                pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                combined_data = {}

                for page_num in range(min(len(pdf_doc), 3)):  # max 3 pages
                    page = pdf_doc[page_num]
                    mat  = fitz.Matrix(2, 2)  # 2x zoom for clarity
                    pix  = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("jpeg")
                    b64_img   = base64.b64encode(img_bytes).decode('utf-8')

                    page_data, err = call_groq_vision(b64_img, 'image/jpeg')
                    if page_data:
                        # Merge: fill nulls with values from later pages
                        for k, v in page_data.items():
                            if v is not None and combined_data.get(k) is None:
                                combined_data[k] = v

                pdf_doc.close()
                if combined_data:
                    return jsonify({'success': True, 'data': combined_data, 'method': 'vision'})

            except ImportError:
                print("PyMuPDF not installed — skipping vision fallback for PDF")
            except Exception as e:
                print("PDF vision error:", e)

            # Strategy 3: last resort — send raw base64 text description
            data, err = call_groq_text(f"[PDF file, base64 excerpt]: {base64.b64encode(file_bytes).decode()[:3000]}")
            if data:
                return jsonify({'success': True, 'data': data, 'method': 'b64fallback'})

            return jsonify({'success': False, 'raw': 'Could not extract data from this PDF'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
