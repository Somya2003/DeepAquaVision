import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import io

from models import Generator

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌊 Underwater Image Enhancer",
    page_icon="🌊",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        color: #0077b6;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #48cae4;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #0077b6, #00b4d8);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached so it loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(weight_path="checkpoints/generator.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, device


def calculate_psnr(target, prediction):
    """Calculate PSNR between two tensors [0,1]"""
    target = torch.clamp(target, 0.0, 1.0)
    prediction = torch.clamp(prediction, 0.0, 1.0)
    mse = F.mse_loss(target, prediction)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def tensor_to_image(tensor):
    """Convert tensor [1, 3, H, W] → PIL Image"""
    img = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(img)
    return img


def resize_to_model_input(image, target_size=256):
    """Resize image to dimensions divisible by 64 (for encoder/decoder)"""
    w, h = image.size
    new_w = (w // 64) * 64
    new_h = (h // 64) * 64
    if new_w < 64:
        new_w = 64
    if new_h < 64:
        new_h = 64
    return image.resize((new_w, new_h), Image.BILINEAR)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🌊 Underwater Image Enhancer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">GAN-based deep learning model to restore underwater images</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    weight_path = st.text_input(
        "Model Weight Path",
        value="checkpoints/generator.pth"
    )

    resize_option = st.selectbox(
        "Resize for Processing",
        options=["Original Size", "256×256", "512×512"],
        index=0
    )

    show_psnr = st.checkbox("Calculate PSNR (upload ground truth)", value=False)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses a **U-Net Generator** trained with a 
    **GAN framework** to enhance underwater images 
    by removing haze, color casts, and improving visibility.
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using PyTorch & Streamlit")

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

# Load model
try:
    model, device = load_model(weight_path)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model weights not found at `{weight_path}`. "
             f"Please train the model first using `python train.py`")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ─────────────────────────────────────────────
# IMAGE UPLOAD
# ─────────────────────────────────────────────
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.subheader("📤 Upload Underwater Image")
    uploaded_file = st.file_uploader(
        "Choose an underwater image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="input"
    )

with col_upload2:
    ground_truth_file = None
    if show_psnr:
        st.subheader("📤 Upload Ground Truth (Optional)")
        ground_truth_file = st.file_uploader(
            "Choose the ground truth image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="ground_truth"
        )

# ─────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────
if uploaded_file is not None:

    # Read input image
    input_image = Image.open(uploaded_file).convert("RGB")
    original_size = input_image.size

    # Store original for display
    display_input = input_image.copy()

    # Resize based on user selection
    if resize_option == "256×256":
        input_image = input_image.resize((256, 256), Image.BILINEAR)
    elif resize_option == "512×512":
        input_image = input_image.resize((512, 512), Image.BILINEAR)
    else:
        # Ensure dimensions are divisible by 64
        input_image = resize_to_model_input(input_image)

    # Transform to tensor
    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    st.markdown("---")

    # ─────────────────────────────────────────
    # ENHANCE BUTTON
    # ─────────────────────────────────────────
    if st.button("🚀 Enhance Image", type="primary", use_container_width=True):

        with st.spinner("🔄 Enhancing image... Please wait"):
            with torch.no_grad():
                output_tensor = model(input_tensor)

        # Convert to display images
        enhanced_image = tensor_to_image(output_tensor)

        # ─────────────────────────────────────
        # DISPLAY RESULTS
        # ─────────────────────────────────────
        st.markdown("## 📊 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📷 Original (Underwater)")
            st.image(
                display_input,
                caption=f"Input ({original_size[0]}×{original_size[1]})",
                use_column_width=True
            )

        with col2:
            st.markdown("### ✨ Enhanced (Output)")
            st.image(
                enhanced_image,
                caption=f"Enhanced ({input_image.size[0]}×{input_image.size[1]})",
                use_column_width=True
            )

        # ─────────────────────────────────────
        # PSNR METRICS
        # ─────────────────────────────────────
        if show_psnr and ground_truth_file is not None:
            gt_image = Image.open(ground_truth_file).convert("RGB")

            # Resize ground truth to match output
            gt_image = gt_image.resize(
                (input_image.size[0], input_image.size[1]),
                Image.BILINEAR
            )
            gt_tensor = transform(gt_image).unsqueeze(0).to(device)

            psnr_value = calculate_psnr(gt_tensor, output_tensor)

            st.markdown("## 📈 Quality Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.metric("📏 PSNR (dB)", f"{psnr_value:.2f}")

            with col_m2:
                st.metric("📐 Original Size", f"{original_size[0]}×{original_size[1]}")

            with col_m3:
                st.metric("📐 Processed Size", f"{input_image.size[0]}×{input_image.size[1]}")

        # ─────────────────────────────────────
        # DOWNLOAD BUTTON
        # ─────────────────────────────────────
        st.markdown("---")
        st.markdown("## 💾 Download")

        # Convert enhanced image to bytes for download
        buf = io.BytesIO()
        enhanced_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(
                label="⬇️ Download Enhanced Image",
                data=byte_im,
                file_name="enhanced_underwater.png",
                mime="image/png",
                use_container_width=True
            )

    # ─────────────────────────────────────────
    # SIDE-BY-SIDE SLIDER VIEW (bonus)
    # ─────────────────────────────────────────
    if st.checkbox("🔍 Show comparison slider"):
        with torch.no_grad():
            output_tensor = model(input_tensor)
        enhanced_image = tensor_to_image(output_tensor)

        st.markdown("### Before / After Comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(display_input, caption="Before", use_column_width=True)
        with col_b:
            st.image(enhanced_image, caption="After", use_column_width=True)

else:
    st.info("👆 Upload an underwater image to get started!")
