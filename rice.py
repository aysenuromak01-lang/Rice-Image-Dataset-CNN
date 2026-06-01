import streamlit as st  
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# 1. Kaydedilen pirinç modelinizi yükleyin (Lütfen modelinizin gerçek tam yolunu yazın)
model = load_model('rice_model.h5')

def process_image(img):
    # Modelinizin eğitim aşamasındaki hedef boyutuna göre yeniden boyutlandırıyoruz (Örn: 170x170)
    # Eğer pirinç modelini eğitirken farklı bir boyut kullandıysanız burayı güncelleyin
    img = img.resize((64, 64))
    img = np.array(img, dtype=np.float32)
    
    # RGB'den BGR'ye dönüşüm (Görüntü okuma kütüphanenizin uyumu için)
    img = img[:, :, ::-1] 
    
    # Normalize etme ve boyut genişletme
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    return img

st.title('Pirinç Türü Teşhisi ve Sınıflandırma 🌾')
st.write('Bir pirinç tanesi resmi yükleyin, model türünü tahmin etsin!')

file = st.file_uploader('Bir pirinç resmi yükle', type=['jpg', 'jpeg', 'png'])

if file is not None: 
    img = Image.open(file)
    st.image(img, caption='Yüklenen Pirinç Resmi')
    
    # Resmi işle ve tahmin al
    image = process_image(img)
    prediction = model.predict(image)
    
    # En yüksek olasılıklı sınıfı seç (5 sınıf içinden)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    
    # Modelin tahmin güven oranı (Olasılık yüzdesi)
    confidence = np.max(prediction) * 100
    
    # Pirinç sınıf isimleri listesi (Klasör sıralamanızla birebir aynı olmalıdır)
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    
    # Sonucu ekrana basma
    st.success(f"Tahmin Sonucu: **{class_names[predicted_class_idx]}** (%{confidence:.2f} Güven Oranı)")
    
    # Tüm pirinç türlerinin olasılık dağılımını göster
    with st.expander("Tüm Olasılık Dağılımlarını Gör"):
        for name, prob in zip(class_names, prediction[0]):
            st.write(f"{name}: %{prob*100:.2f}")
