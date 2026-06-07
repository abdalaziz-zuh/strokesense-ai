# StrokeSense AI — دليل التشغيل الكامل

## هيكل المشروع
```
stroke-app/
├── app.py                 ← السيرفر الرئيسي
├── stroke_model.pkl       ← المودل (من Colab)
├── scaler.pkl             ← الـ Scaler (من Colab)
├── requirements.txt
└── templates/
    └── index.html         ← الموقع
```

---

## الخطوة 1: نزّل ملفات المودل من Colab

أضف هذا الكود في آخر الـ Notebook:

```python
import pickle
from google.colab import files

pickle.dump(lr_model, open('stroke_model.pkl', 'wb'))
pickle.dump(scaler,   open('scaler.pkl',       'wb'))

files.download('stroke_model.pkl')
files.download('scaler.pkl')
```

ثم ضعهما في نفس فولدر stroke-app/

---

## الخطوة 2: مفتاح DeepSeek المجاني

1. روح على: https://platform.deepseek.com
2. سجّل حساب مجاني
3. من API Keys → Create new key
4. انسخ المفتاح وضعه في app.py:
   DEEPSEEK_API_KEY = "sk-xxxxxx"

---

## الخطوة 3: تشغيل الموقع

```bash
# في Terminal داخل فولدر stroke-app
pip install -r requirements.txt
python app.py
```

ثم افتح المتصفح على: http://localhost:5000

---

## نشر الموقع مجاناً (للعرض)

### Frontend + Backend معاً على Render:
1. ارفع المشروع على GitHub
2. روح على render.com
3. New → Web Service → اختر الـ repo
4. Start Command: `python app.py`
5. أضف Environment Variable:
   - Key: DEEPSEEK_API_KEY
   - Value: مفتاحك

---

## ملاحظة مهمة
تأكد أن ترتيب الـ features في app.py مطابق لترتيب التدريب:
[gender, age, hypertension, heart_disease, avg_glucose_level, bmi,
 smoking_formerly, smoking_never, smoking_smokes]
