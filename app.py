import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
import gdown
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# =================== CONFIG ===================
DRIVE_FILE_ID = "1xu2z6l7Ey_93ou4CzACfJkMvCMeKGHNg"  # <-- PASTE YOUR REAL GOOGLE DRIVE FILE ID HERE
MODEL_PATH = "maskrcnn_egg_model.pth"
# ==============================================

st.set_page_config(page_title="AI Egg Counting", page_icon="🥚", layout="wide")

# =================== CSS ===================
st.markdown("""
<style>
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(74, 144, 226, 0.4); }
    70% { box-shadow: 0 0 0 20px rgba(74, 144, 226, 0); }
    100% { box-shadow: 0 0 0 0 rgba(74, 144, 226, 0); }
}

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

.main .block-container {
    background: rgba(232, 240, 248, 0.98);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    backdrop-filter: blur(12px);
    margin-top: 1.5rem;
    border: 1px solid rgba(255,255,255,0.2);
    color: #1a1a2e;
}

.title-text {
    font-size: 3.2rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3, #ff6b6b);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}

.subtitle-text {
    text-align: center;
    color: #4a5568;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 2.5rem;
    letter-spacing: 4px;
    text-transform: uppercase;
}

.stButton>button {
    background: linear-gradient(90deg, #4a90e2, #63b3ed);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 0.8rem 2.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
    animation: pulse 2s infinite;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(74, 144, 226, 0.6);
    background: linear-gradient(90deg, #357abd, #4a90e2);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #4a90e2, #63b3ed, #48dbfb) !important;
    border-radius: 10px;
    height: 12px !important;
}

div[data-testid="stSpinner"] > div {
    justify-content: center;
    color: #4a90e2;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# =================== TITLE ===================
st.markdown('<div class="title-text">AI-powered Egg Counting & Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Mask R-CNN Instance Segmentation</div>', unsafe_allow_html=True)

# =================== MODEL LOADING ===================
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2

    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    if os.path.exists(MODEL_PATH):
        with st.spinner("📂 Loading local model..."):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        os.makedirs("models", exist_ok=True)
        save_path = "models/maskrcnn_egg_model.pth"
        if not os.path.exists(save_path):
            with st.spinner("📥 Downloading model from Google Drive..."):
                gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", save_path, quiet=False)
        model.load_state_dict(torch.load(save_path, map_location=device))

    model.to(device)
    model.eval()
    return model, device

with st.spinner("🧠 Loading AI Model..."):
    model, device = load_model()

st.success("✅ Model loaded successfully!")

# =================== HELPERS ===================
def preprocess(img_rgb):
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_norm = img_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor

def get_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    return xs.mean(), ys.mean()

def draw_on_image(img_rgb, pred, orig_h, orig_w):
    scores = pred[0]['scores'].cpu().numpy()
    masks = pred[0]['masks'].cpu().numpy()
    boxes = pred[0]['boxes'].cpu().numpy()
    labels = pred[0]['labels'].cpu().numpy()

    result = img_rgb.copy()
    count = 0

    for i in range(len(scores)):
        if scores[i] > 0.5 and labels[i] == 1:
            count += 1
            mask = (masks[i, 0] > 0.5).astype(np.uint8)
            mask_big = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            overlay = result.copy()
            overlay[mask_big > 0] = (255, 0, 0)
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

            x1, y1, x2, y2 = boxes[i]
            x1 = int(x1 * orig_w / 512); y1 = int(y1 * orig_h / 512)
            x2 = int(x2 * orig_w / 512); y2 = int(y2 * orig_h / 512)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(result, f"Total Eggs: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return result, count

# =================== UI ===================
option = st.selectbox("📂 Select Input Type", ["🖼️ Image", "🎬 Video"], index=0)

if option == "🖼️ Image":
    uploaded = st.file_uploader("Upload egg image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Original", use_container_width=True)

        if st.button("🔍 PROCESS IMAGE"):
            with st.spinner("🥚 AI is detecting eggs... Please wait"):
                img_tensor = preprocess(img_rgb)
                with torch.no_grad():
                    pred = model(img_tensor)
                result_img, count = draw_on_image(img_rgb, pred, img_rgb.shape[0], img_rgb.shape[1])

            st.balloons()
            st.success(f"✅ Done! Detected **{count}** eggs.")
            st.image(result_img, caption=f"Result: {count} Eggs", use_container_width=True)

            result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', result_bgr)
            st.download_button("⬇️ DOWNLOAD RESULT", buf.tobytes(), "egg_result.jpg", "image/jpeg")

else:  # VIDEO
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        st.video(uploaded)

        with st.expander("⚙️ ROI Box Settings (Auto-calculated %)"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: rx1 = st.slider("Left %", 0.0, 1.0, 0.05)
            with c2: ry1 = st.slider("Top %", 0.0, 1.0, 0.25)
            with c3: rx2 = st.slider("Right %", 0.0, 1.0, 0.95)
            with c4: ry2 = st.slider("Bottom %", 0.0, 1.0, 0.75)

        if st.button("🎬 PROCESS VIDEO"):
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, "in.mp4")
                raw_path = os.path.join(tmpdir, "raw.mp4")
                final_path = os.path.join(tmpdir, "final.mp4")

                with open(in_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                cap = cv2.VideoCapture(in_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                roi_x1, roi_y1 = int(w * rx1), int(h * ry1)
                roi_x2, roi_y2 = int(w * rx2), int(h * ry2)

                out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                active_eggs, next_id, total_count = {}, 0, 0
                match_dist = 60

                prog = st.progress(0)
                status = st.empty()

                idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame is None or frame.size == 0:
                        continue

                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    oh, ow = img_rgb.shape[:2]

                    with torch.no_grad():
                        pred = model(preprocess(img_rgb))

                    scores = pred[0]['scores'].cpu().numpy()
                    masks = pred[0]['masks'].cpu().numpy()
                    boxes = pred[0]['boxes'].cpu().numpy()
                    labels = pred[0]['labels'].cpu().numpy()

                    dets = []
                    for i in range(len(scores)):
                        if scores[i] > 0.5 and labels[i] == 1:
                            m = (masks[i, 0] > 0.5).astype(np.uint8)
                            cx512, cy512 = get_centroid(m)
                            if cx512 is None:
                                continue
                            cx = int(cx512 * ow / 512)
                            cy = int(cy512 * oh / 512)
                            x1, y1, x2, y2 = boxes[i]
                            x1 = int(x1 * ow / 512); y1 = int(y1 * oh / 512)
                            x2 = int(x2 * ow / 512); y2 = int(y2 * oh / 512)
                            dets.append({'cx': cx, 'cy': cy, 'box': [x1, y1, x2, y2], 'mask': m})

                    # ROI filter
                    roi_dets = [d for d in dets if roi_x1 <= d['cx'] <= roi_x2 and roi_y1 <= d['cy'] <= roi_y2]

                    # Track
                    matched = set()
                    new_dets = []
                    for det in roi_dets:
                        cx, cy = det['cx'], det['cy']
                        best_id, best_dist = None, 999999
                        for eid, egg in active_eggs.items():
                            if eid in matched:
                                continue
                            dist = ((egg['cx']-cx)**2 + (egg['cy']-cy)**2)**0.5
                            if dist < best_dist and dist < match_dist:
                                best_dist = dist
                                best_id = eid

                        if best_id is not None:
                            active_eggs[best_id].update({'cx': cx, 'cy': cy, 'missed': 0})
                            matched.add(best_id)
                        else:
                            new_dets.append(det)

                    for det in new_dets:
                        active_eggs[next_id] = {'cx': det['cx'], 'cy': det['cy'], 'missed': 0}
                        total_count += 1
                        next_id += 1

                    to_del = [eid for eid in active_eggs if eid not in matched and active_eggs[eid]['missed'] > 5]
                    for eid in to_del:
                        del active_eggs[eid]
                    for eid in active_eggs:
                        if eid not in matched:
                            active_eggs[eid]['missed'] += 1

                    # Draw
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)
                    for det in roi_dets:
                        mb = cv2.resize(det['mask'], (ow, oh), interpolation=cv2.INTER_NEAREST)
                        overlay = frame.copy()
                        overlay[mb > 0] = (0, 0, 255)
                        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                        x1, y1, x2, y2 = det['box']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(frame, f"EGG COUNT: {total_count}", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                    out.write(frame)

                    # SAFE: Only update progress bar and text. NO st.image() in loop.
                    if idx % 5 == 0:
                        prog.progress(min((idx + 1) / total, 1.0))
                        status.markdown(f"**Frame:** `{idx}/{total}` &nbsp;&nbsp;|&nbsp;&nbsp; 🥚 **Eggs Counted:** `{total_count}`")
                    idx += 1

                cap.release()
                out.release()
                prog.empty()
                status.empty()

                # Fix codec for browser
                try:
                    import subprocess
                    subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-vcodec', 'libx264',
                                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', final_path],
                                   check=True, capture_output=True)
                    out_file = final_path
                except Exception:
                    out_file = raw_path

                st.balloons()
                st.success(f"✅ Done! Total eggs counted: **{total_count}**")
                st.video(out_file)

                with open(out_file, "rb") as f:
                    st.download_button("⬇️ DOWNLOAD PROCESSED VIDEO", f,
                                       "egg_counting_output.mp4", "video/mp4")
