import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO

# --- 設定頁面標題 ---
st.set_page_config(page_title="皮膚偵測 AI 系統", layout="wide")
st.title("🔍 皮膚偵測與分析系統")
st.write("上傳圖片並調整亮度，即可進行即時 AI 偵測")

# --- 載入模型 (快取處理) ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 側邊欄設定 ---
st.sidebar.header("參數設定")
brightness = st.sidebar.slider("圖片亮度調整", 0.5, 2.0, 1.0, 0.1)
conf_threshold = st.sidebar.slider("AI 信心度門檻", 0.1, 1.0, 0.25, 0.05)

# --- 圖片上傳區域 ---
uploaded_file = st.file_uploader("請選擇一張皮膚照片 (jpg, png, jpeg)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # 1. 調整亮度
    enhancer = ImageEnhance.Brightness(image)
    processed_image = enhancer.enhance(brightness)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("待測圖片 (已調亮度)")
        st.image(processed_image, use_container_width=True)
    
    # 2. 進行 YOLOv8 偵測
    img_array = np.array(processed_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if st.button("開始 AI 偵測"):
        with st.spinner('AI 正在分析中...'):
            results = model.predict(source=img_bgr, conf=conf_threshold)
            
            # 取得畫好框的圖片
            annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("偵測結果圖")
                st.image(annotated_img, use_container_width=True)
            
            # --- 新增：顯示偵測詳細數據 ---
            st.divider() # 加入分隔線
            st.subheader("📊 偵測數據明細")
            
            boxes = results[0].boxes
            num_detections = len(boxes)
            
            if num_detections > 0:
                # 建立資料清單
                detection_data = []
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]  # 取得類別名稱
                    conf = float(box.conf[0])      # 取得信心數值
                    
                    detection_data.append({
                        "偵測目標": label,
                        "信心指數": f"{conf:.2%}" # 轉為百分比格式
                    })
                
                # 使用 Streamlit 表格呈現
                st.success(f"偵測完成！共發現 {num_detections} 處目標。")
                st.table(detection_data)
                
            else:
                st.warning("未偵測到任何目標，建議調整亮度或降低信心度門檻。")