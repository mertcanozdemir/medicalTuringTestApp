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
from datetime import datetime
import tempfile
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import json
import googleapiclient

# Uygulama başlığı ve açıklaması
st.set_page_config(page_title="Görsel Turing Testi", layout="wide")
st.title("Görsel Turing Testi - Kardiyak Görüntüler")
st.markdown("Bu uygulama, gerçek ve sentetik kardiyak görüntüleri ayırt etme yeteneğinizi değerlendirir.")

# Varsayılan dizin yolu (sadece sonuçlar için)
DEFAULT_OUTPUT_DIR = r".\results"  # Yerel dizin yolu

# Google Drive entegrasyonu için değişkenler
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/drive.file']

# Google Drive klasör ID'leri
DEFAULT_REAL_FOLDER_ID = "1XJgpXqdVSfOIriECXwXuwccs3N0KiqQ_"  # Gerçek klasör ID'si ile değiştirin
DEFAULT_SYNTHETIC_FOLDER_ID = "1iGykeA2-cG68wj-4xZDXLp6CH4DcisLo"  # Sentetik klasör ID'si ile değiştirin
DEFAULT_RESULTS_FOLDER_ID = "1Zjh8EDGnUAJGor4sVxIyMllw1zswlWQA"  # Sonuçlar klasör ID'si ile değiştirin

# Oturum durumlarını kontrol et ve başlat
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_idx = 0
    st.session_state.results = []
    st.session_state.all_images = []
    st.session_state.completed = False
    st.session_state.radiologist_id = ""
    # Sadece sonuç dizinini başlat
    st.session_state.output_dir = DEFAULT_OUTPUT_DIR
    # Çıktı dizinini oluştur (yoksa)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    # Google Drive ile ilgili durumlar
    st.session_state.drive_service = None
    st.session_state.real_folder_id = DEFAULT_REAL_FOLDER_ID
    st.session_state.synth_folder_id = DEFAULT_SYNTHETIC_FOLDER_ID
    st.session_state.results_folder_id = DEFAULT_RESULTS_FOLDER_ID
    # Geçici klasör
    st.session_state.temp_dir = tempfile.mkdtemp()
    # Kimlik bilgileri dosyası yüklenmiş mi?
    st.session_state.credentials_uploaded = False
    # Drive'a sonuçlar kaydedilsin mi?
    st.session_state.save_to_drive = True
    # Drive'daki sonuç dosyasının ID'si
    st.session_state.drive_result_file_id = None

# Ana fonksiyonlar
def upload_file_to_drive(drive_service, file_path, folder_id, file_name=None):
    """Google Drive'a dosya yükle"""
    try:
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        
        media = googleapiclient.http.MediaFileUpload(
            file_path, 
            resumable=True
        )
        
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    except Exception as e:
        st.error(f"Drive'a dosya yükleme hatası: {e}")
        return None

def update_file_in_drive(drive_service, file_path, file_id, file_name=None):
    """Google Drive'daki dosyayı güncelle"""
    try:
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        file_metadata = {
            'name': file_name
        }
        
        media = googleapiclient.http.MediaFileUpload(
            file_path, 
            resumable=True
        )
        
        file = drive_service.files().update(
            fileId=file_id,
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    except Exception as e:
        st.error(f"Drive'daki dosyayı güncelleme hatası: {e}")
        return None

def authenticate_google_drive(credentials_json):
    """Google Drive API ile kimlik doğrulama yap"""
    try:
        credentials_dict = json.loads(credentials_json)
        credentials = Credentials.from_service_account_info(
            credentials_dict, scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=credentials)
        return drive_service
    except Exception as e:
        st.error(f"Google Drive kimlik doğrulama hatası: {e}")
        return None

def authenticate_google_drive(credentials_json):
    """Google Drive API ile kimlik doğrulama yap"""
    try:
        credentials_dict = json.loads(credentials_json)
        credentials = Credentials.from_service_account_info(
            credentials_dict, scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=credentials)
        return drive_service
    except Exception as e:
        st.error(f"Google Drive kimlik doğrulama hatası: {e}")
        return None

def list_files_in_folder(drive_service, folder_id):
    """Google Drive klasöründeki dosyaları listele"""
    try:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=1000,
            fields="files(id, name, mimeType)").execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"Klasör içeriği listelenirken hata oluştu: {e}")
        return []

def download_file_from_drive(drive_service, file_id, file_name, destination_folder):
    """Google Drive'dan dosyayı indir"""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        
        file_path = os.path.join(destination_folder, file_name)
        
        with open(file_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        
        return file_path
    except Exception as e:
        st.error(f"Dosya indirme hatası (ID: {file_id}): {e}")
        return None

def load_images_from_drive(drive_service, folder_id, img_type, temp_dir):
    """Google Drive klasöründen görüntüleri yükle"""
    images = []
    
    # Klasördeki dosyaları listele
    files = list_files_in_folder(drive_service, folder_id)
    
    if not files:
        st.warning(f"Google Drive klasöründe ({folder_id}) görüntü bulunamadı!")
        return []
    
    # Sadece resim dosyalarını filtrele (DICOM desteği çıkartıldı)
    image_files = [f for f in files if f['mimeType'].startswith('image/') or
                   f['name'].lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning(f"Google Drive klasöründe desteklenen görüntü formatı bulunamadı!")
        return []
    
    # 50'den fazla görüntü varsa, rastgele 50 tane seç
    if len(image_files) > 50:
        import time
        random.seed(time.time())
        image_files = random.sample(image_files, 50)
    
    # İndirilecek görüntü sayısı
    total_images = len(image_files)
    progress_bar = st.progress(0)
    
    # Görüntüleri indir ve işle
    for i, file in enumerate(image_files):
        try:
            # İlerleme durumunu güncelle
            progress_text = st.empty()
            progress_text.text(f"İndiriliyor: {file['name']} ({i+1}/{total_images})")
            progress_bar.progress((i+1)/total_images)
            
            # Dosyayı indir
            file_path = download_file_from_drive(drive_service, file['id'], file['name'], temp_dir)
            
            if not file_path:
                continue
                
            # Standart görüntü formatları için
            img = Image.open(file_path)
            images.append({
                'path': file_path,
                'drive_id': file['id'],
                'true_type': img_type
            })
        except Exception as e:
            st.warning(f"Dosya işlenirken hata oluştu {file['name']}: {e}")
    
    # İlerleme çubuğunu ve metni temizle
    progress_bar.empty()
    
    st.success(f"{len(images)} {img_type} görüntü Google Drive'dan yüklendi")
    return images

def display_current_image():
    """Mevcut görüntüyü göster"""
    if st.session_state.current_idx < len(st.session_state.all_images):
        # İlerleme bilgisi
        progress = int((st.session_state.current_idx / len(st.session_state.all_images)) * 100)
        st.progress(progress)
        st.subheader(f"Görüntü {st.session_state.current_idx + 1} / {len(st.session_state.all_images)}")
        
        img_data = st.session_state.all_images[st.session_state.current_idx]
        
        try:
            # Standart görüntü dosyasını yükle
            img = Image.open(img_data['path'])
            
            # Yeniden boyutlandır
            img = img.resize((256, 256))
            
            # Görüntüyü merkeze yerleştir
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img, width=256)
            
            # Kullanıcı talimatları
            st.info("Lütfen yukarıdaki görüntünün gerçek mi yoksa yapay zeka tarafından üretilmiş (sentetik) mi olduğunu değerlendirin.")
            
            # Sınıflandırma butonları
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Gerçek", key=f"real_{st.session_state.current_idx}", use_container_width=True):
                    record_classification("gerçek")
            with col2:
                if st.button("Sentetik", key=f"synth_{st.session_state.current_idx}", use_container_width=True):
                    record_classification("sentetik")
            
        except Exception as e:
            st.error(f"Görüntü gösterilemiyor: {e}")
            st.session_state.current_idx += 1
            st.experimental_rerun()
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
            'image_id': img_data.get('drive_id', ''),
            'true_type': img_data['true_type'],
            'classified_as': classification,
            'correct': img_data['true_type'] == classification,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.results.append(result)
        
        # Her değerlendirmeden sonra mevcut sonuçları yerel dosyaya kaydet
        try:
            df = pd.DataFrame(st.session_state.results)
            df.to_csv(st.session_state.output_file, index=False)
            
            # Eğer Drive'a kaydetme seçiliyse ve klasör ID'si varsa
            if st.session_state.save_to_drive and st.session_state.results_folder_id:
                if st.session_state.drive_result_file_id:
                    # Drive'daki dosyayı güncelle
                    update_file_in_drive(
                        st.session_state.drive_service, 
                        st.session_state.output_file, 
                        st.session_state.drive_result_file_id,
                        st.session_state.result_file_name
                    )
                else:
                    # İlk kez Drive'a yükle
                    file_id = upload_file_to_drive(
                        st.session_state.drive_service, 
                        st.session_state.output_file, 
                        st.session_state.results_folder_id,
                        st.session_state.result_file_name
                    )
                    if file_id:
                        st.session_state.drive_result_file_id = file_id
        except Exception as e:
            st.warning(f"Sonuçlar kaydedilirken hata oluştu: {e}")
        
        # Sonraki görüntüye geç
        st.session_state.current_idx += 1
        
        # Sayfayı yeniden yükle
        st.experimental_rerun()

def initialize_app():
    """Uygulamayı başlat ve görüntüleri yükle"""
    st.header("Değerlendirmeyi Başlat")
    
    # Radyolog bilgileri
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.radiologist_id = st.text_input("Radyolog Kimliği:", value="", key="rad_id_input")
    with col2:
        tarih = datetime.now().strftime("%Y-%m-%d")
        st.text_input("Tarih:", value=tarih, disabled=True)
    
    # Google Drive ayarları
    st.subheader("Google Drive Ayarları")
    
    # Eğer Streamlit Cloud üzerinde çalışıyorsa ve secrets yüklenmişse
    if hasattr(st, 'secrets') and 'google_service_account' in st.secrets:
        st.success("☁️ Streamlit Cloud'da çalışıyor. Google Drive kimlik bilgileri secrets'dan yüklendi.")
        credentials_json = st.secrets["google_service_account"]
        st.session_state.credentials_uploaded = True
    else:
        # Servis hesabı kimlik bilgileri
        uploaded_file = st.file_uploader(
            "Servis Hesabı Kimlik Bilgileri (JSON dosyası):",
            type=["json"],
            help="Google Cloud Console'dan indirdiğiniz servis hesabı anahtarı JSON dosyasını yükleyin."
        )
        
        if uploaded_file is not None:
            try:
                # JSON dosyasını oku
                credentials_json = uploaded_file.getvalue().decode('utf-8')
                st.session_state.credentials_uploaded = True
            except Exception as e:
                st.error(f"Dosya okuma hatası: {e}")
                st.session_state.credentials_uploaded = False
    
    # Klasör ID'leri
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.real_folder_id = st.text_input(
            "Gerçek Görüntüler Klasör ID:", 
            value=st.session_state.real_folder_id,
            help="Google Drive'daki gerçek görüntüleri içeren klasörün ID'si"
        )
    with col2:
        st.session_state.synth_folder_id = st.text_input(
            "Sentetik Görüntüler Klasör ID:",
            value=st.session_state.synth_folder_id,
            help="Google Drive'daki sentetik görüntüleri içeren klasörün ID'si"
        )
    
    # Sonuç ayarları
    st.subheader("Sonuç Ayarları")
    
    # Drive Sonuç klasörü
    st.session_state.results_folder_id = st.text_input(
        "Sonuçlar Klasör ID:", 
        value=st.session_state.results_folder_id,
        help="Google Drive'da sonuçların kaydedileceği klasörün ID'si"
    )
    
    st.session_state.save_to_drive = st.checkbox(
        "Sonuçları Google Drive'a da kaydet", 
        value=True,
        help="İşaretliyse, sonuçlar hem yerel olarak hem de Google Drive'da belirtilen klasöre kaydedilir."
    )
    
    # Yerel Sonuç dizini
    with st.expander("Yerel Sonuç Dizini"):
        st.session_state.output_dir = st.text_input("Sonuç Dizini:", 
                                                  value=DEFAULT_OUTPUT_DIR, key="output_dir_input")
        os.makedirs(st.session_state.output_dir, exist_ok=True)
    
    # Yardım metni
    st.info("""
    **Nasıl Kullanılır?**
    1. Radyolog kimliğinizi girin
    2. Google Cloud'dan indirdiğiniz servis hesabı JSON dosyasını yükleyin
    3. Google Drive'daki görüntü klasörlerinin ID'lerini girin
    4. Sonuçların kaydedileceği Google Drive klasör ID'sini girin
    5. "Değerlendirmeyi Başlat" butonuna tıklayın
    """)
    
    # Başlatma butonu
    if st.button("Değerlendirmeyi Başlat", key="start_button", use_container_width=True):
        if not st.session_state.radiologist_id:
            st.error("Lütfen Radyolog Kimliğinizi girin!")
            return

        if not st.session_state.credentials_uploaded:
            st.error("Lütfen servis hesabı kimlik bilgilerini (JSON) yükleyin!")
            return
        
        if not st.session_state.real_folder_id or not st.session_state.synth_folder_id:
            st.error("Lütfen her iki görüntü klasör ID'sini de girin!")
            return
        
        if st.session_state.save_to_drive and not st.session_state.results_folder_id:
            st.error("Lütfen sonuçlar klasör ID'sini girin veya Drive'a kaydetme seçeneğini kapatın!")
            return
            
        with st.spinner("Google Drive bağlantısı kuruluyor..."):
            drive_service = authenticate_google_drive(credentials_json)
            
            if not drive_service:
                st.error("Google Drive kimlik doğrulaması başarısız!")
                return
            
            # Klasörlerin varlığını kontrol et
            real_files = list_files_in_folder(drive_service, st.session_state.real_folder_id)
            synth_files = list_files_in_folder(drive_service, st.session_state.synth_folder_id)
            
            if not real_files:
                st.error(f"Gerçek görüntüler klasörüne erişilemiyor veya klasör boş! (ID: {st.session_state.real_folder_id})")
                return
            
            if not synth_files:
                st.error(f"Sentetik görüntüler klasörüne erişilemiyor veya klasör boş! (ID: {st.session_state.synth_folder_id})")
                return
            
            # Sonuçlar klasörünü kontrol et (eğer Drive'a kaydetme seçiliyse)
            if st.session_state.save_to_drive:
                results_files = list_files_in_folder(drive_service, st.session_state.results_folder_id)
                if results_files is None:
                    st.error(f"Sonuçlar klasörüne erişilemiyor! (ID: {st.session_state.results_folder_id})")
                    return
            
            # Başarılı ise drive_service'i kaydet
            st.session_state.drive_service = drive_service
        
        # Google Drive'dan görüntüleri yükle
        with st.spinner("Görüntüler Google Drive'dan yükleniyor..."):
            real_images = load_images_from_drive(
                st.session_state.drive_service, 
                st.session_state.real_folder_id, 
                'gerçek', 
                st.session_state.temp_dir
            )
            
            synth_images = load_images_from_drive(
                st.session_state.drive_service, 
                st.session_state.synth_folder_id, 
                'sentetik', 
                st.session_state.temp_dir
            )
        
        # Görüntü yükleme başarılı mı kontrol et
        if not real_images or not synth_images:
            st.error("Görüntüler yüklenemedi! Lütfen klasör ID'lerini kontrol edin.")
            return
        
        # Görüntüleri birleştir ve karıştır
        st.session_state.all_images = real_images + synth_images
        
        # Sistem zamanına dayalı gerçek rastgele tohum oluştur
        import time
        random.seed(time.time())
        random.shuffle(st.session_state.all_images)
        
        st.session_state.initialized = True
        
        # Sonuç dosyasının adını oluştur
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file_name = f"vtt_sonuclari_{st.session_state.radiologist_id}_{timestamp}.csv"
        output_file = os.path.join(st.session_state.output_dir, result_file_name)
        st.session_state.output_file = output_file
        st.session_state.result_file_name = result_file_name
        
        st.success(f"Toplamda {len(st.session_state.all_images)} görüntü yüklendi! Değerlendirmeye başlayabilirsiniz.")
        st.experimental_rerun()

def finish_evaluation():
    """Değerlendirmeyi bitir ve sonuçları göster"""
    if not st.session_state.completed:
        # Özet istatistikleri göster
        df = pd.DataFrame(st.session_state.results)
        accuracy = np.mean(df['correct']) * 100
        
        # Ek metrikleri hesapla
        true_positive = np.sum((df['true_type'] == 'gerçek') & (df['classified_as'] == 'gerçek'))
        false_positive = np.sum((df['true_type'] == 'sentetik') & (df['classified_as'] == 'gerçek'))
        true_negative = np.sum((df['true_type'] == 'sentetik') & (df['classified_as'] == 'sentetik'))
        false_negative = np.sum((df['true_type'] == 'gerçek') & (df['classified_as'] == 'sentetik'))
        
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        
        # Sonuç ve grafikler dosyasını oluştur
        try:
            # Grafikler için bir figür oluştur
            plt.figure(figsize=(12, 10))
            
            # Üst grafik: Görüntü türüne göre doğruluk
            plt.subplot(2, 1, 1)
            types = ['Gerçek Görüntüler', 'Sentetik Görüntüler']
            values = [sensitivity*100, specificity*100]
            colors = ['#2986cc', '#e06666']
            plt.bar(types, values, color=colors)
            plt.ylim([0, 100])
            plt.ylabel('Doğruluk Oranı (%)')
            plt.title('Görüntü Türüne Göre Doğruluk')
            
            # Alt grafik: Doğru/Yanlış oranı pasta grafiği
            plt.subplot(2, 1, 2)
            labels = ['Doğru', 'Yanlış']
            sizes = [accuracy, 100-accuracy]
            explode = (0.1, 0)  # Doğru dilimi vurgula
            plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                   shadow=True, startangle=90, colors=['#60bd68', '#f15854'])
            plt.axis('equal')  # Daire şeklinde olmasını sağla
            plt.title('Genel Doğruluk Oranı')
            
            # Grafiği kaydet
            graph_file_name = f"vtt_grafikler_{st.session_state.radiologist_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            graph_file_path = os.path.join(st.session_state.output_dir, graph_file_name)
            plt.tight_layout()
            plt.savefig(graph_file_path)
            
            # Grafiği Drive'a yükle (eğer Drive'a kaydetme seçiliyse)
            if st.session_state.save_to_drive and st.session_state.results_folder_id:
                graph_id = upload_file_to_drive(
                    st.session_state.drive_service,
                    graph_file_path,
                    st.session_state.results_folder_id,
                    graph_file_name
                )
                if graph_id:
                    st.session_state.drive_graph_file_id = graph_id
        except Exception as e:
            st.warning(f"Grafik dosyası oluşturulurken hata oluştu: {e}")
        
        st.balloons()  # Kutlama animasyonu
        st.success("🎉 Değerlendirme tamamlandı! Teşekkür ederiz.")
        
        # Sonuçları yeni bir sekmeli arayüze yerleştir
        tab1, tab2, tab3 = st.tabs(["Özet", "Grafikler", "Detaylı Veriler"])
        
        with tab1:
            st.subheader("Performans Özeti")
            
            # Metrikler için üç sütunlu düzen
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Genel Doğruluk", value=f"%{accuracy:.2f}")
            
            with col2:
                st.metric(label="Duyarlılık (Gerçek Görüntüler)", value=f"%{sensitivity*100:.2f}")
            
            with col3:
                st.metric(label="Özgüllük (Sentetik Görüntüler)", value=f"%{specificity*100:.2f}")
            
            st.markdown("""
            **Tanımlar:**
            - **Duyarlılık**: Gerçek görüntüleri doğru tanımlama yeteneği
            - **Özgüllük**: Sentetik görüntüleri doğru tanımlama yeteneği
            """)
            
            # Sonuçların nereye kaydedildiği bilgisi
            st.subheader("Sonuç Dosyaları")
            st.write(f"**Yerel sonuç dosyası**: {st.session_state.output_file}")
            
            if st.session_state.save_to_drive and st.session_state.drive_result_file_id:
                st.write(f"**Google Drive sonuç dosyası ID**: {st.session_state.drive_result_file_id}")
                drive_file_link = f"https://drive.google.com/file/d/{st.session_state.drive_result_file_id}/view"
                st.markdown(f"[Google Drive'da Sonuç Dosyasını Aç]({drive_file_link})")
            
            if hasattr(st.session_state, 'drive_graph_file_id') and st.session_state.drive_graph_file_id:
                st.write(f"**Google Drive grafik dosyası ID**: {st.session_state.drive_graph_file_id}")
                graph_file_link = f"https://drive.google.com/file/d/{st.session_state.drive_graph_file_id}/view"
                st.markdown(f"[Google Drive'da Grafik Dosyasını Aç]({graph_file_link})")

        with tab2:
            st.subheader("Performans Grafikleri")
            
            # Görüntü türüne göre doğruluk grafiği
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            types = ['Gerçek Görüntüler', 'Sentetik Görüntüler']
            values = [sensitivity*100, specificity*100]
            colors = ['#2986cc', '#e06666']
            ax1.bar(types, values, color=colors)
            ax1.set_ylim([0, 100])
            ax1.set_ylabel('Doğruluk Oranı (%)')
            ax1.set_title('Görüntü Türüne Göre Doğruluk')
            
            # Grafiği göster
            st.pyplot(fig1)
            
            # Pasta grafiği - Doğru/Yanlış oranı
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            labels = ['Doğru', 'Yanlış']
            sizes = [accuracy, 100-accuracy]
            explode = (0.1, 0)  # Doğru dilimi vurgula
            ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                   shadow=True, startangle=90, colors=['#60bd68', '#f15854'])
            ax2.axis('equal')  # Daire şeklinde olmasını sağla
            
            st.pyplot(fig2)
        
        with tab3:
            st.subheader("Görüntü Değerlendirme Detayları")
            
            # Veri çerçevesini göster
            show_df = df.copy()
            show_df['image_path'] = show_df['image_path'].apply(lambda x: os.path.basename(x))  # sadece dosya adını göster
            show_df = show_df.rename(columns={
                'radiologist_id': 'Radyolog',
                'image_path': 'Görüntü',
                'image_id': 'Görüntü ID',
                'true_type': 'Gerçek Tür',
                'classified_as': 'Değerlendirme',
                'correct': 'Doğruluk',
                'timestamp': 'Zaman'
            })
            
            st.dataframe(show_df, use_container_width=True)
        
        # Sonuç verilerini CSV olarak indirmek için
        st.download_button(
            label="Sonuçları CSV Olarak İndir",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=st.session_state.result_file_name,
            mime="text/csv",
        )
        
        # Yeni değerlendirme başlatma butonu
        if st.button("Yeni Değerlendirme Başlat", key="new_eval"):
            st.session_state.initialized = False
            st.session_state.current_idx = 0
            st.session_state.results = []
            st.session_state.all_images = []
            st.session_state.completed = False
            st.session_state.radiologist_id = ""
            st.session_state.drive_result_file_id = None
            if hasattr(st.session_state, 'drive_graph_file_id'):
                delattr(st.session_state, 'drive_graph_file_id')
            st.experimental_rerun()
        
        st.session_state.completed = True

# Yan panel ayarları
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/cardiology-concept-illustration_114360-6921.jpg", width=100)
    st.header("Görsel Turing Testi")
    st.markdown("---")
    
    if not st.session_state.initialized:
        st.info("Değerlendirmeye başlamak için formu doldurun ve 'Değerlendirmeyi Başlat' butonuna tıklayın.")
        
        # Google Drive Bağlantı Durumu
        st.subheader("Google Drive Durumu")
        if st.session_state.credentials_uploaded:
            st.success("✅ Kimlik bilgileri yüklendi")
        else:
            st.warning("❌ Kimlik bilgileri yüklenmedi")
            
        if st.session_state.real_folder_id != DEFAULT_REAL_FOLDER_ID:
            st.success(f"✅ Gerçek görüntü klasörü: {st.session_state.real_folder_id[:5]}...")
        else:
            st.warning("❌ Gerçek görüntü klasörü: Ayarlanmadı")
            
        if st.session_state.synth_folder_id != DEFAULT_SYNTHETIC_FOLDER_ID:
            st.success(f"✅ Sentetik görüntü klasörü: {st.session_state.synth_folder_id[:5]}...")
        else:
            st.warning("❌ Sentetik görüntü klasörü: Ayarlanmadı")
            
        if st.session_state.results_folder_id != DEFAULT_RESULTS_FOLDER_ID:
            st.success(f"✅ Sonuçlar klasörü: {st.session_state.results_folder_id[:5]}...")
        else:
            st.warning("❌ Sonuçlar klasörü: Ayarlanmadı")
    else:
        # Değerlendirme durumu
        st.subheader("Değerlendirme Durumu")
        st.write(f"**Radyolog:** {st.session_state.radiologist_id}")
        st.write(f"**İlerleme:** {st.session_state.current_idx}/{len(st.session_state.all_images)} görüntü")
        
        # İşlemleri göster
        completed_real = sum(1 for r in st.session_state.results if r['classified_as'] == 'gerçek')
        completed_synth = sum(1 for r in st.session_state.results if r['classified_as'] == 'sentetik')
        
        st.write(f"**Gerçek olarak değerlendirilen:** {completed_real}")
        st.write(f"**Sentetik olarak değerlendirilen:** {completed_synth}")
        
        # Drive'a kayıt durumu
        if st.session_state.save_to_drive:
            if st.session_state.drive_result_file_id:
                st.success("✅ Sonuçlar Google Drive'a kaydediliyor")
            else:
                st.info("⏳ Sonuçlar henüz Drive'a kaydedilmedi")
        
        # Değerlendirmeyi sıfırla
        st.markdown("---")
        if st.button("Değerlendirmeyi Sıfırla", key="reset_button"):
            if st.session_state.current_idx > 0:
                reset_confirm = st.checkbox("Eminim, değerlendirmeyi sıfırla")
                if reset_confirm:
                    st.session_state.initialized = False
                    st.session_state.current_idx = 0
                    st.session_state.results = []
                    st.session_state.all_images = []
                    st.session_state.completed = False
                    st.session_state.radiologist_id = ""
                    st.session_state.drive_result_file_id = None
                    st.experimental_rerun()
            else:
                st.session_state.initialized = False
                st.session_state.current_idx = 0
                st.session_state.results = []
                st.session_state.all_images = []
                st.session_state.completed = False
                st.session_state.radiologist_id = ""
                st.session_state.drive_result_file_id = None
                st.experimental_rerun()
    
    # Uygulama bilgileri
    st.markdown("---")
    st.caption("Görsel Turing Testi v1.0")
    st.caption("© 2025 Streamlit ile geliştirilmiştir")

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
