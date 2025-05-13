import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import random
from sklearn.metrics import cohen_kappa_score
import pydicom
import io
import base64

# Uygulama başlığı ve açıklaması
st.set_page_config(page_title="Görsel Turing Testi", layout="wide")
st.title("Görsel Turing Testi - Kardiyak Görüntüler")
st.markdown("Bu uygulama, gerçek ve sentetik kardiyak görüntüleri ayırt etme yeteneğinizi değerlendirir.")

# Oturum durumlarını kontrol et ve başlat
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_idx = 0
    st.session_state.results = []
    st.session_state.all_images = []
    st.session_state.completed = False
    st.session_state.radiologist_id = ""

# Ana fonksiyonlar
def load_images(directory, img_type):
    """Dizinden görüntüleri gerçek türleriyle yükle"""
    images = []
    image_files = []
    
    # Dizindeki tüm görüntü dosyalarını al
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if filepath.lower().endswith(('.dcm', '.png', '.jpg', '.jpeg')):
                image_files.append(filepath)
    except FileNotFoundError:
        st.error(f"Dizin bulunamadı: {directory}")
        return []
    
    # 50'den fazla görüntü varsa, rastgele 50 tane seç
    if len(image_files) > 50:
        # Zamana dayalı rastgele seçimi sağla
        import time
        random.seed(time.time())
        image_files = random.sample(image_files, 50)
    else:
        # 50'den az görüntü varsa, tümünü kullan
        image_files = image_files[:50]
        
    # Seçilen görüntüleri işle
    for filepath in image_files:
        if filepath.lower().endswith(('.dcm')):
            # DICOM dosyaları için
            try:
                ds = pydicom.dcmread(filepath)
                pixel_array = ds.pixel_array
                images.append({
                    'path': filepath,
                    'true_type': img_type,
                    'pixel_data': pixel_array
                })
            except Exception as e:
                st.warning(f"DICOM dosyası yüklenirken hata oluştu {filepath}: {e}")
        elif filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Standart görüntü formatları için
            try:
                # Sadece görüntünün açılabildiğini doğrulamak için
                img = Image.open(filepath)
                img.verify()
                images.append({
                    'path': filepath,
                    'true_type': img_type
                })
            except Exception as e:
                st.warning(f"Görüntü dosyası yüklenirken hata oluştu {filepath}: {e}")
    
    st.write(f"{len(images)} {img_type} görüntü yüklendi")
    return images

def initialize_app():
    """Uygulamayı başlat ve görüntüleri yükle"""
    # Radyolog kimliğini al
    st.session_state.radiologist_id = st.text_input("Radyolog Kimliğinizi Girin:", value="", key="rad_id_input")
    
    if st.session_state.radiologist_id:
        # Görüntüleri yükle
        with st.spinner("Görüntüler yükleniyor..."):
            real_img_dir = st.session_state.real_dir
            synth_img_dir = st.session_state.synth_dir
            
            real_images = load_images(real_img_dir, 'gerçek')
            synth_images = load_images(synth_img_dir, 'sentetik')
            
            # Görüntüleri birleştir ve karıştır
            st.session_state.all_images = real_images + synth_images
            
            # Sistem zamanına dayalı gerçek rastgele tohum oluştur
            import time
            random.seed(time.time())
            random.shuffle(st.session_state.all_images)
            
            st.session_state.initialized = True
        
        st.success(f"Toplamda {len(st.session_state.all_images)} görüntü yüklendi! Değerlendirmeye başlayabilirsiniz.")
        st.rerun()


def display_current_image():
    """Mevcut görüntüyü göster"""
    if st.session_state.current_idx < len(st.session_state.all_images):
        img_data = st.session_state.all_images[st.session_state.current_idx]
        
        try:
            if 'pixel_data' in img_data:
                # DICOM görüntüsünü standart pencere ayarlarıyla göster
                pixel_array = img_data['pixel_data']
                
                # Standart pencere ayarlarını uygula
                window_center = 500
                window_width = 1500
                
                # Pencerelemeyi uygula
                lower_bound = window_center - window_width // 2
                upper_bound = window_center + window_width // 2
                windowed_image = np.clip(pixel_array, lower_bound, upper_bound)
                
                # Görüntülemek için 0-255 aralığına normalize et
                windowed_image = ((windowed_image - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
                
                # PIL Image'a dönüştür
                img = Image.fromarray(windowed_image)
            else:
                # Standart görüntü dosyasını yükle
                img = Image.open(img_data['path'])
            
            # Yeniden boyutlandır
            img = img.resize((256, 256))
            
            # Görüntüyü göster
            st.image(img, caption=f"Görüntü {st.session_state.current_idx + 1} / {len(st.session_state.all_images)}", width=256)
            
            # Sınıflandırma butonları
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Gerçek", key=f"real_{st.session_state.current_idx}"):
                    record_classification("gerçek")
            with col2:
                if st.button("Sentetik", key=f"synth_{st.session_state.current_idx}"):
                    record_classification("sentetik")
            
        except Exception as e:
            st.error(f"Görüntü gösterilemiyor: {e}")
            record_classification("hata")
    else:
        finish_evaluation()

def record_classification(classification):
    """Radyoloğun sınıflandırmasını kaydet ve sonraki görüntüye geç"""
    if st.session_state.current_idx < len(st.session_state.all_images):
        # Sonucu kaydet
        img_data = st.session_state.all_images[st.session_state.current_idx]
        result = {
            'radiologist_id': st.session_state.radiologist_id,
            'image_path': img_data['path'],
            'true_type': img_data['true_type'],
            'classified_as': classification,
            'correct': img_data['true_type'] == classification if classification != "hata" else False
        }
        st.session_state.results.append(result)
        
        # Her değerlendirmeden sonra mevcut sonuçları dosyaya kaydet
        df = pd.DataFrame(st.session_state.results)
        output_file = os.path.join(st.session_state.output_dir, f"vtt_sonuclari_{st.session_state.radiologist_id}.csv")
        df.to_csv(output_file, index=False)
        
        # Sonraki görüntüye geç
        st.session_state.current_idx += 1
        
        # Sayfayı yeniden yükle
        st.rerun()


def finish_evaluation():
    """Değerlendirmeyi bitir ve sonuçları göster"""
    if not st.session_state.completed:
        # Özet istatistikleri göster
        df = pd.DataFrame(st.session_state.results)
        accuracy = np.mean(df['correct']) * 100
        
        # Ek metrikleri hesapla ve göster
        true_positive = np.sum((df['true_type'] == 'gerçek') & (df['classified_as'] == 'gerçek'))
        false_positive = np.sum((df['true_type'] == 'sentetik') & (df['classified_as'] == 'gerçek'))
        true_negative = np.sum((df['true_type'] == 'sentetik') & (df['classified_as'] == 'sentetik'))
        false_negative = np.sum((df['true_type'] == 'gerçek') & (df['classified_as'] == 'sentetik'))
        
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        
        st.success("Değerlendirme tamamlandı!")
        
        st.subheader("Sonuçlar")
        st.write(f"Genel doğruluk: %{accuracy:.2f}")
        st.write(f"Duyarlılık (gerçek görüntüleri doğru tanımlama): {sensitivity:.2f}")
        st.write(f"Özgüllük (sentetik görüntüleri doğru tanımlama): {specificity:.2f}")
        
        # Sonuç grafiği
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(['Gerçek', 'Sentetik'], [sensitivity, specificity])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Doğruluk Oranı')
        ax.set_title('Görüntü Türüne Göre Doğruluk')
        
        # Grafiği göster
        st.pyplot(fig)
        
        # Sonuç verilerini CSV olarak indirmek için
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="vtt_sonuclari_{st.session_state.radiologist_id}.csv">Sonuçları CSV olarak indir</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.session_state.completed = True

# Yan panel ayarları
with st.sidebar:
    st.header("Ayarlar")
    
    # Dizin ayarları
    st.subheader("Veri Dizinleri")
    
    if not st.session_state.initialized:
        st.session_state.real_dir = st.text_input("Gerçek Görüntüler Dizini:", value="C:/Users/Mertcan/Desktop/gata-yazilim/images/real")
        st.session_state.synth_dir = st.text_input("Sentetik Görüntüler Dizini:", value="C:/Users/Mertcan/Desktop/gata-yazilim/images/synthetic")
        st.session_state.output_dir = st.text_input("Sonuç Dizini:", value="C:/Veriler/Sonuclar")
        
        # Çıktı dizinini oluştur (yoksa)
        os.makedirs(st.session_state.output_dir, exist_ok=True)
    else:
        st.write(f"**Gerçek Görüntüler:** {st.session_state.real_dir}")
        st.write(f"**Sentetik Görüntüler:** {st.session_state.synth_dir}")
        st.write(f"**Sonuç Dizini:** {st.session_state.output_dir}")
    
    # Değerlendirmeyi sıfırla
    if st.session_state.initialized:
        if st.button("Değerlendirmeyi Sıfırla"):
            st.session_state.initialized = False
            st.session_state.current_idx = 0
            st.session_state.results = []
            st.session_state.all_images = []
            st.session_state.completed = False
            st.session_state.radiologist_id = ""
            st.rerun()


# Ana uygulama mantığı
if not st.session_state.initialized:
    # Uygulama henüz başlatılmadıysa, başlatma formunu göster
    initialize_app()
else:
    # Uygulama başlatıldıysa, değerlendirme arayüzünü göster
    if not st.session_state.completed:
        display_current_image()
    else:
        finish_evaluation()