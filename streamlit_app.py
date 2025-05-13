import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import random
from sklearn.metrics import cohen_kappa_score
import seaborn as sns
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
st.set_page_config(page_title="Kardiyak Görüntü Değerlendirme Platformu", layout="wide")
st.title("Kardiyak Görüntü Değerlendirme Platformu")
st.markdown("Bu platform, kardiyak görüntülerin değerlendirilmesi için iki farklı test sunar.")

# Varsayılan dizin yolu (sadece sonuçlar için)
DEFAULT_OUTPUT_DIR = r".\results"  # Yerel dizin yolu
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Google Drive entegrasyonu için değişkenler
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/drive.file']

# Google Drive klasör ID'leri
DEFAULT_REAL_FOLDER_ID = "1XJgpXqdVSfOIriECXwXuwccs3N0KiqQ_"  # Gerçek klasör ID'si
DEFAULT_SYNTHETIC_FOLDER_ID = "1iGykeA2-cG68wj-4xZDXLp6CH4DcisLo"  # Sentetik klasör ID'si
DEFAULT_RESULTS_FOLDER_ID = "1Zjh8EDGnUAJGor4sVxIyMllw1zswlWQA"  # Sonuçlar klasör ID'si

# Anatomik Olabilirlik Değerlendirmesi özellikleri
APA_FEATURES = [
    "Genel Anatomik Olabilirlik",
    "Ventrikül Morfolojisi",
    "Miyokard Kalınlığı",
    "Papiller Kas Tanımı",
    "Kan Havuzu Kontrastı"
]

# Oturum durumlarını kontrol et ve başlat
if 'test_type' not in st.session_state:
    st.session_state.test_type = "vtt"  # Varsayılan olarak Görsel Turing Testi seçili
    st.session_state.initialized = False
    st.session_state.current_idx = 0
    st.session_state.results = []
    st.session_state.all_images = []
    st.session_state.completed = False
    st.session_state.radiologist_id = ""
    st.session_state.output_dir = DEFAULT_OUTPUT_DIR
    st.session_state.drive_service = None
    st.session_state.real_folder_id = DEFAULT_REAL_FOLDER_ID
    st.session_state.synth_folder_id = DEFAULT_SYNTHETIC_FOLDER_ID
    st.session_state.results_folder_id = DEFAULT_RESULTS_FOLDER_ID
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.credentials_uploaded = False
    st.session_state.save_to_drive = True
    st.session_state.drive_result_file_id = None
    # APA özellikleri için varsayılan puanlar
    st.session_state.ratings = {feature: 3 for feature in APA_FEATURES}
    # Görüntü önbelleği
    st.session_state.cached_images = {
        'real': None,  # Gerçek görüntülerin önbelleği
        'synth': None  # Sentetik görüntülerin önbelleği
    }
    # Önbellek durumu
    st.session_state.cache_loaded = False

## ORTAK FONKSİYONLAR ##

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
    """Google Drive kimlik doğrulama"""
    try:
        # Eğer zaten bir dictionary ise
        if isinstance(credentials_json, dict):
            credentials_dict = credentials_json
        else:
            # String ise JSON olarak parse et
            credentials_dict = json.loads(credentials_json)
        
        # Özel anahtardaki kaçış karakterlerini düzelt
        if 'private_key' in credentials_dict:
            credentials_dict['private_key'] = credentials_dict['private_key'].replace('\\n', '\n')
        
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

def load_images_from_drive(drive_service, folder_id, img_type, temp_dir, max_images=50):
    """Google Drive klasöründen görüntüleri yükle"""
    images = []
    
    # Klasördeki dosyaları listele
    files = list_files_in_folder(drive_service, folder_id)
    
    if not files:
        st.warning(f"Google Drive klasöründe ({folder_id}) görüntü bulunamadı!")
        return []
    
    # Sadece desteklenen görüntü formatlarını filtrele
    image_files = [f for f in files if f['mimeType'].startswith('image/') or
                  f['name'].lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning(f"Google Drive klasöründe desteklenen görüntü formatı bulunamadı!")
        return []
    
    # Görüntü sayısını sınırla
    if len(image_files) > max_images:
        random.seed(datetime.now().timestamp())
        image_files = random.sample(image_files, max_images)
    
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

def initialize_app():
    """Uygulamayı başlat - ortak giriş formu"""
    st.header("Değerlendirmeyi Başlat")
    
    # Radyolog bilgileri
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.radiologist_id = st.text_input("Radyolog Kimliği:", value="", key="rad_id_input")
    with col2:
        tarih = datetime.now().strftime("%Y-%m-%d")
        st.text_input("Tarih:", value=tarih, disabled=True)
    
    # Kimlik bilgilerini otomatik yükle
    if hasattr(st, 'secrets') and 'google_service_account' in st.secrets:
        st.success("☁️ Streamlit Cloud'da çalışıyor. Google Drive kimlik bilgileri secrets'dan yüklendi.")
        credentials_json = dict(st.secrets["google_service_account"])
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
    
    # Yardım metni
    if st.session_state.test_type == "apa":
        st.info("""
        **Anatomik Olabilirlik Değerlendirmesi - Nasıl Kullanılır?**
        1. Radyolog kimliğinizi girin
        2. Google Cloud'dan indirdiğiniz servis hesabı JSON dosyasını yükleyin
        3. "Değerlendirmeyi Başlat" butonuna tıklayın
        4. Her görüntüyü dikkatle inceleyin ve istenen anatomik özellikleri 1-5 ölçeğinde değerlendirin
        5. Değerlendirme sonuçlarınız otomatik olarak kaydedilecektir
        """)
    elif st.session_state.test_type == "vtt":
        st.info("""
        **Görsel Turing Testi - Nasıl Kullanılır?**
        1. Radyolog kimliğinizi girin
        2. Google Cloud'dan indirdiğiniz servis hesabı JSON dosyasını yükleyin
        3. "Değerlendirmeyi Başlat" butonuna tıklayın
        4. Her görüntüyü dikkatle inceleyin ve gerçek mi yoksa sentetik mi olduğunu belirtin
        5. Değerlendirme sonuçlarınız otomatik olarak kaydedilecektir
        """)
    else:
        st.info("""
        **Nasıl Kullanılır?**
        1. Yan menüden test türünü seçin (Anatomik Olabilirlik Değerlendirmesi veya Görsel Turing Testi)
        2. Radyolog kimliğinizi girin
        3. Google Cloud'dan indirdiğiniz servis hesabı JSON dosyasını yükleyin
        4. "Değerlendirmeyi Başlat" butonuna tıklayın
        """)
    
    # Başlatma butonu - Test türü seçilmişse aktifleştir
    if st.session_state.test_type:
        if st.button("Değerlendirmeyi Başlat", key="start_button", use_container_width=True):
            if not st.session_state.radiologist_id:
                st.error("Lütfen Radyolog Kimliğinizi girin!")
                return

            if not st.session_state.credentials_uploaded:
                st.error("Lütfen servis hesabı kimlik bilgilerini (JSON) yükleyin!")
                return
            
            with st.spinner("Google Drive bağlantısı kuruluyor..."):
                drive_service = authenticate_google_drive(credentials_json)
                
                if not drive_service:
                    st.error("Google Drive kimlik doğrulaması başarısız!")
                    return
                
                # Klasörlerin varlığını kontrol et
                if st.session_state.test_type == "vtt":
                    # VTT için gerçek ve sentetik görüntüler gerekli
                    real_files = list_files_in_folder(drive_service, st.session_state.real_folder_id)
                    if not real_files:
                        st.error(f"Gerçek görüntüler klasörüne erişilemiyor veya klasör boş! (ID: {st.session_state.real_folder_id})")
                        return
                
                # Her iki test için de sentetik görüntüler gerekli
                synth_files = list_files_in_folder(drive_service, st.session_state.synth_folder_id)
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
                # Test türüne göre görüntüleri yükle
                if st.session_state.test_type == "apa":
                    # Anatomik Olabilirlik Değerlendirmesi için sadece sentetik görüntüler
                    max_images = 100  # APA için daha fazla görüntü
                    
                    # Önbellekte sentetik görüntü var mı kontrol et
                    if st.session_state.cached_images['synth'] is None:
                        synth_images = load_images_from_drive(
                            st.session_state.drive_service, 
                            st.session_state.synth_folder_id, 
                            'sentetik', 
                            st.session_state.temp_dir,
                            max_images
                        )
                        # Önbelleğe kaydet
                        st.session_state.cached_images['synth'] = synth_images
                    else:
                        synth_images = st.session_state.cached_images['synth']
                        st.success(f"{len(synth_images)} sentetik görüntü önbellekten yüklendi")
                    
                    # Görüntü yükleme başarılı mı kontrol et
                    if not synth_images:
                        st.error("Görüntüler yüklenemedi! Lütfen klasör ID'lerini kontrol edin.")
                        return
                    
                    # Tüm görüntüleri ayarla
                    st.session_state.all_images = synth_images
                
                elif st.session_state.test_type == "vtt":
                    # Görsel Turing Testi için gerçek ve sentetik görüntüler
                    max_images = 50  # VTT için daha az görüntü
                    
                    # Önbellekte gerçek görüntü var mı kontrol et
                    if st.session_state.cached_images['real'] is None:
                        real_images = load_images_from_drive(
                            st.session_state.drive_service, 
                            st.session_state.real_folder_id, 
                            'gerçek', 
                            st.session_state.temp_dir,
                            max_images
                        )
                        # Önbelleğe kaydet
                        st.session_state.cached_images['real'] = real_images
                    else:
                        real_images = st.session_state.cached_images['real']
                        st.success(f"{len(real_images)} gerçek görüntü önbellekten yüklendi")
                    
                    # Önbellekte sentetik görüntü var mı kontrol et
                    if st.session_state.cached_images['synth'] is None:
                        synth_images = load_images_from_drive(
                            st.session_state.drive_service, 
                            st.session_state.synth_folder_id, 
                            'sentetik', 
                            st.session_state.temp_dir,
                            max_images
                        )
                        # Önbelleğe kaydet
                        st.session_state.cached_images['synth'] = synth_images
                    else:
                        synth_images = st.session_state.cached_images['synth']
                        st.success(f"{len(synth_images)} sentetik görüntü önbellekten yüklendi")
                    
                    # Görüntü yükleme başarılı mı kontrol et
                    if not real_images or not synth_images:
                        st.error("Görüntüler yüklenemedi! Lütfen klasör ID'lerini kontrol edin.")
                        return
                    
                    # Görüntüleri birleştir ve karıştır
                    st.session_state.all_images = real_images + synth_images
            
            # Görüntüleri karıştır
            random.seed(datetime.now().timestamp())
            random.shuffle(st.session_state.all_images)
            
            st.session_state.initialized = True
            
            # Sonuç dosyasının adını oluştur
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            test_prefix = "apa" if st.session_state.test_type == "apa" else "vtt"
            result_file_name = f"{test_prefix}_sonuclari_{st.session_state.radiologist_id}_{timestamp}.csv"
            output_file = os.path.join(st.session_state.output_dir, result_file_name)
            st.session_state.output_file = output_file
            st.session_state.result_file_name = result_file_name
            
            st.success(f"Toplamda {len(st.session_state.all_images)} görüntü yüklendi! Değerlendirmeye başlayabilirsiniz.")
            st.rerun()

## ANATOMİK OLABİLİRLİK DEĞERLENDİRMESİ (APA) FONKSİYONLARI ##

def display_apa_image():
    """Anatomik Olabilirlik Değerlendirmesi için görüntü göster"""
    if st.session_state.current_idx < len(st.session_state.all_images):
        # İlerleme bilgisi
        progress = int((st.session_state.current_idx / len(st.session_state.all_images)) * 100)
        st.progress(progress)
        st.subheader(f"Görüntü {st.session_state.current_idx + 1} / {len(st.session_state.all_images)}")
        
        img_data = st.session_state.all_images[st.session_state.current_idx]
        
        try:
            # Görüntü dosyasını yükle
            img = Image.open(img_data['path'])
            
            # Görüntüyü yeniden boyutlandır (256x256)
            img = img.resize((256, 256), Image.LANCZOS)
            
            # Görüntüyü merkeze yerleştir
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img, width=256)
            
            # Değerlendirme talimatı
            st.info("Lütfen aşağıdaki özellikleri 1-5 ölçeğinde değerlendirin (1: Çok Kötü, 5: Mükemmel)")
            
            # Değerlendirme kaydırıcıları
            with st.container():
                # Her özellik için kaydırıcı
                for feature in APA_FEATURES:
                    st.session_state.ratings[feature] = st.slider(
                        f"{feature}", 
                        min_value=1, 
                        max_value=5, 
                        value=st.session_state.ratings.get(feature, 3),
                        key=f"slider_{feature}_{st.session_state.current_idx}"
                    )
            
            # Gönder butonu
            if st.button("Değerlendirmeyi Gönder ve İlerle", use_container_width=True):
                record_apa_assessment()
            
        except Exception as e:
            st.error(f"Görüntü gösterilemiyor: {e}")
            st.session_state.current_idx += 1
            st.rerun()
    else:
        finish_apa_evaluation()

def record_apa_assessment():
    """Anatomik Olabilirlik Değerlendirmesini kaydet"""
    if st.session_state.current_idx < len(st.session_state.all_images):
        # Sonucu kaydet
        img_data = st.session_state.all_images[st.session_state.current_idx]
        
        result = {
            'radiologist_id': st.session_state.radiologist_id,
            'image_path': img_data['path'],
            'image_id': img_data.get('drive_id', ''),
            'image_number': st.session_state.current_idx + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Her özellik için puanları kaydet
        for feature in APA_FEATURES:
            result[feature.replace(" ", "_").lower()] = st.session_state.ratings[feature]
        
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
        
        # Sonraki görüntü için kaydırıcıları sıfırla
        for feature in APA_FEATURES:
            st.session_state.ratings[feature] = 3
        
        # Sonraki görüntüye geç
        st.session_state.current_idx += 1
        
        # Sayfayı yeniden yükle
        st.rerun()

def finish_apa_evaluation():
    """Anatomik Olabilirlik Değerlendirmesini bitir ve sonuçları göster"""
    if not st.session_state.completed:
        # Özet istatistikleri göster
        df = pd.DataFrame(st.session_state.results)
        
        # Her özellik için ortalama puanları hesapla
        mean_scores = {feature: np.mean(df[feature.replace(" ", "_").lower()]) 
                      for feature in APA_FEATURES}
        
        # Görselleştirme oluştur
        try:
            # Grafikler için bir figür oluştur
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Çubuk grafik için verileri hazırla
            feature_names = [f for f in APA_FEATURES]
            values = [mean_scores[f] for f in APA_FEATURES]
            
            # Ortalama puanları çubuk grafik olarak göster
            bars = ax.bar(feature_names, values, color='#2986cc')
            
            # Çubukların üzerine değerleri ekle
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, 
                        val + 0.1, 
                        f'{val:.2f}', 
                        ha='center', 
                        va='bottom',
                        fontweight='bold')
            
            ax.set_ylim([0, 5.5])
            ax.set_ylabel('Ortalama Puan', fontsize=12)
            ax.set_title('Anatomik Olabilirlik Puanları', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            
            # Grafiği kaydet
            graph_file_name = f"apa_grafikler_{st.session_state.radiologist_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            graph_file_path = os.path.join(st.session_state.output_dir, graph_file_name)
            plt.tight_layout()
            plt.savefig(graph_file_path)
            
            # Grafiği Drive'a yükle
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
        
        # Sonuçları sekmeli arayüzde göster
        tab1, tab2, tab3 = st.tabs(["Özet", "Grafikler", "Detaylı Veriler"])
        
        with tab1:
            st.subheader("Değerlendirme Özeti")
            
            # Metrikler için sütunlar
            cols = st.columns(len(APA_FEATURES))
            for i, feature in enumerate(APA_FEATURES):
                with cols[i]:
                    st.metric(
                        label=feature, 
                        value=f"{mean_scores[feature]:.2f}"
                    )
            
            # Sonuçların kaydedildiği yerler
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
            st.subheader("Puanlama Grafikleri")
            
            # Ortalama puanlar grafiğini göster
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(feature_names, values, color='#2986cc')
            ax.set_ylim([0, 5])
            ax.set_ylabel('Ortalama Puan')
            ax.set_title('Anatomik Olabilirlik Ortalama Puanları')
            plt.xticks(rotation=45, ha='right')
            
            st.pyplot(fig)
            
            # Puan dağılımı ısı haritası
            st.subheader("Puan Dağılımı")
            
            # Isı haritası için verileri hazırla
            heatmap_data = []
            for feature in APA_FEATURES:
                feature_key = feature.replace(" ", "_").lower()
                if feature_key in df.columns:
                    scores = df[feature_key].value_counts().reindex(range(1, 6), fill_value=0)
                    heatmap_data.append(scores.values)
            
            if heatmap_data:
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                
                # Yüzdelere dönüştür
                heatmap_array = np.array(heatmap_data)
                data_percent = (heatmap_array / heatmap_array.sum(axis=1)[:, np.newaxis]) * 100
                
                sns.heatmap(data_percent, annot=True, fmt='.1f', cmap='YlGnBu', 
                          xticklabels=['1', '2', '3', '4', '5'],
                          yticklabels=feature_names, ax=ax2)
                
                ax2.set_title('Puan Dağılımı (% olarak)')
                ax2.set_xlabel('5 Basamaklı Likert Ölçeğinde Puan')
                
                st.pyplot(fig2)
        
        with tab3:
            st.subheader("Değerlendirme Detayları")
            
            # Veri çerçevesini göster
            show_df = df.copy()
            show_df['image_path'] = show_df['image_path'].apply(lambda x: os.path.basename(x))  # Sadece dosya adını göster
            
            # Sütun isimlerini daha anlaşılır hale getir
            column_mapping = {
                'radiologist_id': 'Radyolog',
                'image_path': 'Görüntü',
                'image_id': 'Görüntü ID',
                'image_number': 'Görüntü No',
                'timestamp': 'Zaman'
            }
            
            # Özellik sütunlarını eşleştir
            for feature in APA_FEATURES:
                feature_key = feature.replace(" ", "_").lower()
                column_mapping[feature_key] = feature
            
            # Sütun isimlerini değiştir
            show_df = show_df.rename(columns=column_mapping)
            
            st.dataframe(show_df, use_container_width=True)
        
        # Sonuçları CSV olarak indir
        st.download_button(
            label="Sonuçları CSV Olarak İndir",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=st.session_state.result_file_name,
            mime="text/csv",
        )
        
        # Yeni değerlendirme başlat butonu
        if st.button("Yeni Değerlendirme Başlat", key="new_eval"):
            st.session_state.initialized = False
            st.session_state.current_idx = 0
            st.session_state.results = []
            st.session_state.all_images = []
            st.session_state.completed = False
            st.session_state.radiologist_id = ""
            st.session_state.drive_result_file_id = None
            for feature in APA_FEATURES:
                st.session_state.ratings[feature] = 3
            # Önbellek verilerini korumak için cache_loaded'ı false yap
            st.session_state.cache_loaded = False
            if hasattr(st.session_state, 'drive_graph_file_id'):
                delattr(st.session_state, 'drive_graph_file_id')
            st.rerun()
        
        st.session_state.completed = True

def analyze_apa_results(radiologist1_file, radiologist2_file):
    """İki radyolog arasındaki Anatomik Olabilirlik Değerlendirmelerini analiz et"""
    st.header("İki Radyolog Arasındaki Değerlendirme Analizi")
    
    try:
        # Sonuçları yükle
        df1 = pd.read_csv(radiologist1_file)
        df2 = pd.read_csv(radiologist2_file)
        
        # Görüntü yoluna göre sonuçları birleştir
        merged = pd.merge(df1, df2, on='image_path', suffixes=('_rad1', '_rad2'))
        
        # Analiz için özellik sütunları
        feature_cols = [feature.replace(" ", "_").lower() for feature in APA_FEATURES]
        
        # Her özellik için Cohen's kappa hesapla
        kappa_scores = {}
        for feature in feature_cols:
            # Puanları tamsayıya dönüştür
            rad1_scores = merged[f"{feature}_rad1"].astype(int)
            rad2_scores = merged[f"{feature}_rad2"].astype(int)
            
            # Ağırlıklı kappa hesapla (Likert ölçekleri için daha uygun)
            kappa = cohen_kappa_score(rad1_scores, rad2_scores, weights='linear')
            kappa_scores[feature] = kappa
        
        # Her özellik ve radyolog için ortalama puanları hesapla
        mean_scores_rad1 = {feature: np.mean(merged[f"{feature}_rad1"]) for feature in feature_cols}
        mean_scores_rad2 = {feature: np.mean(merged[f"{feature}_rad2"]) for feature in feature_cols}
        
        # Görselleştirme oluştur
        tab1, tab2, tab3 = st.tabs(["Cohen's Kappa", "Ortalama Puanlar", "Detaylı Veriler"])
        
        with tab1:
            st.subheader("Değerlendiriciler Arası Uyum (Cohen's Kappa)")
            
            # Kappa puanları için çubuk grafik
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_names = [f.replace("_", " ").title() for f in feature_cols]
            kappa_values = [kappa_scores[f] for f in feature_cols]
            
            # Kappa değerine göre renklendirme
            colors = ['#ff9999' if k < 0.4 else '#ffcc99' if k < 0.6 else '#99cc99' if k < 0.8 else '#99ccff' for k in kappa_values]
            
            bars = ax.bar(feature_names, kappa_values, color=colors)
            
            # Değerleri ekle
            for bar, val in zip(bars, kappa_values):
                ax.text(bar.get_x() + bar.get_width()/2, 
                        val + 0.02, 
                        f'{val:.2f}', 
                        ha='center', 
                        va='bottom',
                        fontweight='bold')
            
            ax.set_ylim([0, 1])
            ax.set_ylabel('Cohen\'s Kappa')
            ax.set_title('Değerlendiriciler Arası Uyum')
            plt.xticks(rotation=45, ha='right')
            
            # Kappa yorumlama çizgileri
            ax.axhline(y=0.4, linestyle='--', color='r', alpha=0.3)
            ax.axhline(y=0.6, linestyle='--', color='y', alpha=0.3)
            ax.axhline(y=0.8, linestyle='--', color='g', alpha=0.3)
            
            st.pyplot(fig)
            
            # Kappa yorumlama rehberi
            st.info("""
            **Cohen's Kappa Yorumlama Rehberi:**
            - < 0.4: Zayıf uyum (kırmızı)
            - 0.4 - 0.6: Orta düzeyde uyum (turuncu)
            - 0.6 - 0.8: İyi uyum (yeşil)
            - > 0.8: Çok iyi uyum (mavi)
            """)
        
        with tab2:
            st.subheader("Ortalama Puanlar Karşılaştırması")
            
            # Ortalama puanlar için çubuk grafik
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(feature_names))
            width = 0.35
            
            # Ortalama puanları göster
            ax.bar(x - width/2, [mean_scores_rad1[f] for f in feature_cols], width, label='Radyolog 1')
            ax.bar(x + width/2, [mean_scores_rad2[f] for f in feature_cols], width, label='Radyolog 2')
            
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_ylim([0, 5])
            ax.set_ylabel('Ortalama Puan')
            ax.set_title('Özelliğe Göre Ortalama Anatomik Olabilirlik Puanları')
            ax.legend()
            
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Detaylı Veri Analizi")
            
            # Puan dağılımı ısı haritası
            st.subheader("Puan Dağılımı (%)")
            
            # Her özellik için puan dağılımını hesapla
            score_distributions = {}
            for feature in feature_cols:
                # Her iki radyologdan puanları birleştir
                all_scores = list(merged[f"{feature}_rad1"]) + list(merged[f"{feature}_rad2"])
                score_distributions[feature] = np.bincount(all_scores, minlength=6)[1:]  # 1-5 puanlar
            
            # Isı haritası oluştur
            data = np.array([score_distributions[f] for f in feature_cols])
            # Yüzdelere dönüştür
            data_percent = (data / data.sum(axis=1)[:, np.newaxis]) * 100
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(data_percent, annot=True, fmt='.1f', cmap='YlGnBu', 
                       xticklabels=['1', '2', '3', '4', '5'],
                       yticklabels=feature_names, ax=ax)
            
            ax.set_title('Olabilirlik Puanları Dağılımı (Toplam %)')
            ax.set_xlabel('5 Basamaklı Likert Ölçeğinde Puan')
            
            st.pyplot(fig)
            
            # Birleştirilmiş veri tablosunu göster
            st.subheader("Birleştirilmiş Veri")
            st.dataframe(merged)
            
            # Özet rapor oluştur
            st.subheader("Özet Rapor")
            
            summary_text = """
            # Anatomical Plausibility Assessment - Summary Report
            ================================================
            
            ## Inter-rater agreement (Cohen's kappa) by feature:
            """
            
            for feature, kappa in kappa_scores.items():
                feature_name = feature.replace("_", " ").title()
                summary_text += f"- {feature_name}: {kappa:.2f}\n"
            
            summary_text += "\n## Mean scores by radiologist:\n\n### Radiologist 1:\n"
            for feature, score in mean_scores_rad1.items():
                feature_name = feature.replace("_", " ").title()
                summary_text += f"- {feature_name}: {score:.2f}\n"
            
            summary_text += "\n### Radiologist 2:\n"
            for feature, score in mean_scores_rad2.items():
                feature_name = feature.replace("_", " ").title()
                summary_text += f"- {feature_name}: {score:.2f}\n"
            
            summary_text += "\n## Score distribution (count):\n"
            for feature in feature_cols:
                feature_name = feature.replace("_", " ").title()
                summary_text += f"\n### {feature_name}:\n"
                for score, count in enumerate(score_distributions[feature], start=1):
                    summary_text += f"- Score {score}: {count}\n"
            
            st.markdown(summary_text)
            
            # Özet raporu indir
            st.download_button(
                label="Özet Raporu İndir",
                data=summary_text,
                file_name=f"apa_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
    
    except Exception as e:
        st.error(f"Sonuçlar analiz edilirken hata oluştu: {e}")

## GÖRSEL TURING TESTİ (VTT) FONKSİYONLARI ##

def display_vtt_image():
    """Görsel Turing Testi için görüntü göster"""
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
                    record_vtt_classification("gerçek")
            with col2:
                if st.button("Sentetik", key=f"synth_{st.session_state.current_idx}", use_container_width=True):
                    record_vtt_classification("sentetik")
            
        except Exception as e:
            st.error(f"Görüntü gösterilemiyor: {e}")
            st.session_state.current_idx += 1
            st.rerun()
    else:
        finish_vtt_evaluation()

def record_vtt_classification(classification):
    """Görsel Turing Testi sınıflandırmasını kaydet"""
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
        st.rerun()

def finish_vtt_evaluation():
    """Görsel Turing Testini bitir ve sonuçları göster"""
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
            
            # Grafiği Drive'a yükle
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
        
        # Sonuçları sekmeli arayüzde göster
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
            show_df['image_path'] = show_df['image_path'].apply(lambda x: os.path.basename(x))  # Sadece dosya adını göster
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
            st.rerun()
        
        st.session_state.completed = True

# Yan panel ayarları
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/cardiology-concept-illustration_114360-6921.jpg", width=100)
    st.header("Kardiyak Görüntü Değerlendirme")
    st.markdown("---")
    
    # Test türü seçimi (eğer henüz başlatılmadıysa)
    if not st.session_state.initialized:
        st.subheader("Test Seçimi")
        
        test_selection = st.radio(
            "Hangi testi yapmak istiyorsunuz?",
            ["Görsel Turing Testi", "Anatomik Olabilirlik Değerlendirmesi"],
            index=0,  # Varsayılan olarak Görsel Turing Testi seçili
            key="test_selection"
        )
        
        # Test seçimine göre durumu ayarla
        if test_selection == "Anatomik Olabilirlik Değerlendirmesi":
            st.session_state.test_type = "apa"
            st.info("""
            **Anatomik Olabilirlik Değerlendirmesi**
            
            Bu test, sentetik kardiyak görüntülerin anatomik özelliklerini 1-5 ölçeğinde değerlendirmenizi sağlar.
            """)
        elif test_selection == "Görsel Turing Testi":
            st.session_state.test_type = "vtt"
            st.info("""
            **Görsel Turing Testi**
            
            Bu test, kardiyak görüntülerin gerçek mi yoksa yapay zeka tarafından üretilmiş mi olduğunu ayırt etme yeteneğinizi değerlendirir.
            """)
        
        # Google Drive Bağlantı Durumu
        st.subheader("Google Drive Durumu")
        if st.session_state.credentials_uploaded:
            st.success("✅ Kimlik bilgileri yüklendi")
        else:
            st.warning("❌ Kimlik bilgileri yüklenmedi")
        
        # Sonuç analizi (APA için)
        if st.session_state.test_type == "apa":
            st.subheader("Sonuç Analizi")
            if st.checkbox("İki radyolog sonucunu analiz et"):
                rad1_file = st.file_uploader("Radyolog 1 CSV Dosyası:", type=["csv"])
                rad2_file = st.file_uploader("Radyolog 2 CSV Dosyası:", type=["csv"])
                
                if rad1_file is not None and rad2_file is not None:
                    # Yüklenen dosyaları geçici dizine kaydet
                    rad1_path = os.path.join(st.session_state.temp_dir, "rad1_results.csv")
                    rad2_path = os.path.join(st.session_state.temp_dir, "rad2_results.csv")
                    
                    with open(rad1_path, "wb") as f:
                        f.write(rad1_file.getbuffer())
                    
                    with open(rad2_path, "wb") as f:
                        f.write(rad2_file.getbuffer())
                    
                    if st.button("Sonuçları Analiz Et"):
                        analyze_apa_results(rad1_path, rad2_path)
    else:
        # Test süreci başlatıldıysa değerlendirme durumunu göster
        st.subheader("Değerlendirme Durumu")
        st.write(f"**Radyolog:** {st.session_state.radiologist_id}")
        st.write(f"**İlerleme:** {st.session_state.current_idx}/{len(st.session_state.all_images)} görüntü")
        
        # Test türüne özgü bilgiler
        if st.session_state.test_type == "vtt":
            # VTT için sınıflandırma istatistikleri
            completed_real = sum(1 for r in st.session_state.results if r['classified_as'] == 'gerçek')
            completed_synth = sum(1 for r in st.session_state.results if r['classified_as'] == 'sentetik')
            
            st.write(f"**Gerçek olarak değerlendirilen:** {completed_real}")
            st.write(f"**Sentetik olarak değerlendirilen:** {completed_synth}")
        elif st.session_state.test_type == "apa":
            # APA için ortalama puanlar (eğer varsa sonuç)
            if st.session_state.results:
                st.subheader("Mevcut Ortalama Puanlar")
                df = pd.DataFrame(st.session_state.results)
                for feature in APA_FEATURES:
                    feature_key = feature.replace(" ", "_").lower()
                    if feature_key in df.columns:
                        avg_score = np.mean(df[feature_key])
                        st.write(f"**{feature}:** {avg_score:.2f}")
        
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
                    st.session_state.test_type = None
                    st.session_state.drive_result_file_id = None
                    # APA puanlarını sıfırla
                    for feature in APA_FEATURES:
                        st.session_state.ratings[feature] = 3
                    # Önbellek verilerini korumak için cache_loaded'ı false yap
                    st.session_state.cache_loaded = False
                    st.rerun()
            else:
                st.session_state.initialized = False
                st.session_state.current_idx = 0
                st.session_state.results = []
                st.session_state.all_images = []
                st.session_state.completed = False
                st.session_state.radiologist_id = ""
                st.session_state.test_type = None
                st.session_state.drive_result_file_id = None
                # APA puanlarını sıfırla
                for feature in APA_FEATURES:
                    st.session_state.ratings[feature] = 3
                # Önbellek verilerini korumak için cache_loaded'ı false yap
                st.session_state.cache_loaded = False
                st.rerun()
    
    # Uygulama bilgileri
    st.markdown("---")
    st.caption("Kardiyak Görüntü Değerlendirme Platformu v1.0")
    st.caption("© 2025 Streamlit ile geliştirilmiştir")

# Ana uygulama mantığı
if not st.session_state.initialized:
    # Uygulama henüz başlatılmadıysa, başlatma formunu göster
    initialize_app()
else:
    # Uygulama başlatıldıysa, test türüne göre değerlendirme arayüzünü göster
    if not st.session_state.completed:
        if st.session_state.test_type == "apa":
            display_apa_image()
        elif st.session_state.test_type == "vtt":
            display_vtt_image()
    else:
        # Tamamlanmış değerlendirme için sonuçları göster
        if st.session_state.test_type == "apa":
            finish_apa_evaluation()
        elif st.session_state.test_type == "vtt":
            finish_vtt_evaluation()
