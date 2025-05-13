import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import random
from sklearn.metrics import cohen_kappa_score
import io
import base64
import tempfile

# Google Drive imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import tempfile

# Uygulama başlığı ve açıklaması
st.set_page_config(page_title="Görsel Turing Testi", layout="wide")
st.title("Görsel Turing Testi - Kardiyak Görüntüler")
st.markdown("Bu uygulama, gerçek ve sentetik kardiyak görüntüleri ayırt etme yeteneğinizi değerlendirir.")

# Google Drive kimlik doğrulama fonksiyonu
def authenticate_gdrive():
    """Google Drive API'sine bağlan ve servis objesi döndür"""
    # İki yöntem var: 
    # 1. Streamlit secrets ile
    if 'gcp_service_account' in st.secrets:
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive']
        )
        drive_service = build('drive', 'v3', credentials=credentials)
        return drive_service
    
    # 2. Yüklenen kimlik dosyası ile
    else:
        st.warning("Google Drive kimlik bilgileri bulunamadı!")
        uploaded_file = st.file_uploader("Service Account JSON dosyasını yükleyin:", type=['json'])
        if uploaded_file is not None:
            credentials = Credentials.from_service_account_info(
                eval(uploaded_file.getvalue().decode('utf-8')),
                scopes=['https://www.googleapis.com/auth/drive']
            )
            drive_service = build('drive', 'v3', credentials=credentials)
            return drive_service
    return None

# Oturum durumlarını kontrol et ve başlat
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_idx = 0
    st.session_state.results = []
    st.session_state.all_images = []
    st.session_state.completed = False
    st.session_state.radiologist_id = ""
    st.session_state.drive_service = None

# Google Drive'dan dosya/klasör listesi alma fonksiyonu
def list_files_in_folder(drive_service, folder_id=None):
    """Google Drive'daki klasörün içindeki dosyaları listeler"""
    query = f"'{folder_id}' in parents" if folder_id else "'root' in parents"
    query += " and trashed = false"
    
    results = drive_service.files().list(
        q=query,
        pageSize=1000,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    
    return results.get('files', [])

# Görüntü yükleme fonksiyonu (Google Drive'dan)
def download_file(drive_service, file_id, file_name):
    """Google Drive'dan dosyayı indir ve temp dosya olarak kaydet"""
    request = drive_service.files().get_media(fileId=file_id)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
        downloader = MediaIoBaseDownload(temp_file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    
    return temp_file.name

# Ana fonksiyonlar
def load_images_from_drive(drive_service, folder_id, img_type):
    """Google Drive klasöründen görüntüleri gerçek türleriyle yükle"""
    images = []
    image_files = []
    
    # Klasördeki tüm dosyaları listele
    try:
        files = list_files_in_folder(drive_service, folder_id)
        for file in files:
            file_name = file.get('name', '')
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append({
                    'id': file.get('id'),
                    'name': file_name
                })
    except Exception as e:
        st.error(f"Google Drive klasörü listelenirken hata oluştu: {e}")
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
    
    with st.progress(0.0) as progress_bar:
        # Seçilen görüntüleri işle
        for i, file_info in enumerate(image_files):
            file_name = file_info['name']
            file_id = file_info['id']
            
            progress_bar.progress((i + 1) / len(image_files))
            
            try:
                # Dosyayı geçici olarak indir
                temp_file_path = download_file(drive_service, file_id, file_name)
                
                # Standart görüntü formatları için
                try:
                    # Sadece görüntünün açılabildiğini doğrulamak için
                    img = Image.open(temp_file_path)
                    img.verify()
                    images.append({
                        'path': temp_file_path,
                        'drive_id': file_id,
                        'name': file_name,
                        'true_type': img_type
                    })
                except Exception as e:
                    st.warning(f"Görüntü dosyası yüklenirken hata oluştu {file_name}: {e}")
            except Exception as e:
                st.warning(f"Dosya indirirken hata oluştu {file_name}: {e}")
    
    st.write(f"{len(images)} {img_type} görüntü yüklendi")
    return images

def upload_to_drive(drive_service, file_path, filename, folder_id=None):
    """Dosyayı Google Drive'a yükle"""
    file_metadata = {
        'name': filename
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(file_path, resumable=True)
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    return file.get('id')

def initialize_app():
    """Uygulamayı başlat ve görüntüleri yükle"""
    # Google Drive servisini başlat
    if not st.session_state.drive_service:
        st.session_state.drive_service = authenticate_gdrive()
        if not st.session_state.drive_service:
            st.error("Google Drive bağlantısı kurulamadı!")
            return
    
    # Radyolog kimliğini al
    st.session_state.radiologist_id = st.text_input("Radyolog Kimliğinizi Girin:", value="", key="rad_id_input")
    
    if st.session_state.radiologist_id:
        # Görüntüleri yükle
        with st.spinner("Görüntüler yükleniyor..."):
            real_folder_id = st.session_state.real_folder_id
            synth_folder_id = st.session_state.synth_folder_id
            
            real_images = load_images_from_drive(st.session_state.drive_service, real_folder_id, 'gerçek')
            synth_images = load_images_from_drive(st.session_state.drive_service, synth_folder_id, 'sentetik')
            
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
            # Standart görüntü dosyasını yükle
            img = Image.open(img_data['path'])
            
            # Yeniden boyutlandır
            img = img.resize((256, 256))
            
            # Görüntüyü göster
            st.image(img, caption=f"Görüntü {st.session_state.current_idx + 1} / {len(st.session_state.all_images)}", width=256)
            
            # Görüntü adını göster
            st.write(f"Dosya adı: {img_data['name']}")
            
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
            'image_name': img_data['name'],
            'image_drive_id': img_data['drive_id'],
            'true_type': img_data['true_type'],
            'classified_as': classification,
            'correct': img_data['true_type'] == classification if classification != "hata" else False
        }
        st.session_state.results.append(result)
        
        # Her değerlendirmeden sonra mevcut sonuçları geçici dosyaya kaydet
        df = pd.DataFrame(st.session_state.results)
        
        # Geçici CSV dosyası oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        # Google Drive'a yükle (her adımda güncelle)
        filename = f"vtt_sonuclari_{st.session_state.radiologist_id}.csv"
        
        # Mevcut sonuç dosyası varsa sil
        if 'result_file_id' in st.session_state:
            try:
                st.session_state.drive_service.files().delete(fileId=st.session_state.result_file_id).execute()
            except:
                pass
        
        # Yeni dosyayı yükle
        file_id = upload_to_drive(
            st.session_state.drive_service, 
            temp_file_path, 
            filename, 
            st.session_state.results_folder_id
        )
        
        # Dosya ID'sini kaydet
        st.session_state.result_file_id = file_id
        
        # Geçici dosyayı sil
        os.unlink(temp_file_path)
        
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
        
        # Son sonuç dosyasına bağlantı
        if 'result_file_id' in st.session_state:
            file_id = st.session_state.result_file_id
            st.write(f"Sonuç dosyası Google Drive'a kaydedildi. ID: {file_id}")
            file_link = f"https://drive.google.com/file/d/{file_id}/view"
            st.markdown(f"[Sonuç dosyasını Google Drive'da görüntüle]({file_link})")
        
        st.session_state.completed = True

# Yan panel ayarları
with st.sidebar:
    st.header("Ayarlar")
    
    # Google Drive bağlantısı
    if not st.session_state.initialized:
        st.subheader("Google Drive Bağlantısı")
        if 'drive_service' not in st.session_state or st.session_state.drive_service is None:
            st.session_state.drive_service = authenticate_gdrive()
            if st.session_state.drive_service:
                st.success("Google Drive bağlantısı kuruldu!")
            else:
                st.error("Google Drive bağlantısı kurulamadı!")
    
    # Dizin ayarları
    st.subheader("Google Drive Klasörleri")
    
    if not st.session_state.initialized:
        # Folder ID'ler için girdi alanları
        st.session_state.real_folder_id = st.text_input(
            "Gerçek Görüntüler Klasörü ID:", 
            value="",
            help="Google Drive'daki gerçek görüntülerin bulunduğu klasörün ID'si"
        )
        
        st.session_state.synth_folder_id = st.text_input(
            "Sentetik Görüntüler Klasörü ID:", 
            value="",
            help="Google Drive'daki sentetik görüntülerin bulunduğu klasörün ID'si"
        )
        
        st.session_state.results_folder_id = st.text_input(
            "Sonuçlar Klasörü ID:", 
            value="",
            help="Google Drive'da sonuçların kaydedileceği klasörün ID'si"
        )
        
        # Drive'daki klasörleri göster
        if st.session_state.drive_service:
            if st.button("Google Drive Klasörlerini Listele"):
                files = list_files_in_folder(st.session_state.drive_service)
                
                # Sadece klasörleri filtrele
                folders = [f for f in files if f.get('mimeType') == 'application/vnd.google-apps.folder']
                
                st.write("Erişilebilir klasörler:")
                for folder in folders:
                    st.write(f"{folder.get('name')}: `{folder.get('id')}`")
    else:
        st.write(f"**Gerçek Görüntüler Klasörü ID:** {st.session_state.real_folder_id}")
        st.write(f"**Sentetik Görüntüler Klasörü ID:** {st.session_state.synth_folder_id}")
        st.write(f"**Sonuçlar Klasörü ID:** {st.session_state.results_folder_id}")
    
    # Değerlendirmeyi sıfırla
    if st.session_state.initialized:
        if st.button("Değerlendirmeyi Sıfırla"):
            st.session_state.initialized = False
            st.session_state.current_idx = 0
            st.session_state.results = []
            st.session_state.all_images = []
            st.session_state.completed = False
            st.session_state.radiologist_id = ""
            # Sonuç dosyası ID'sini temizle ama drive_service'i koru
            if 'result_file_id' in st.session_state:
                del st.session_state.result_file_id
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
