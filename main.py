# Legal Document Extractor - Advanced OCR + LLM Processing
# Modified from grievance extractor for legal document analysis

import streamlit as st
from pdf2image import convert_from_path
import pytesseract
import easyocr
import tempfile
import os
import json
import requests
import time
from PIL import Image, ImageEnhance, ImageStat
import re
import numpy as np
import concurrent.futures
import traceback
import io

# --- OCR-Preprocessing ---
def needs_preprocessing(image, contrast_threshold=30):
    gray = image.convert('L')
    stat = ImageStat.Stat(gray)
    contrast = stat.stddev[0] if isinstance(stat.stddev, (list, tuple)) else stat.stddev
    return contrast < contrast_threshold

def preprocess_image(image):
    if needs_preprocessing(image):
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
    return image

# --- EasyOCR Reader (cache to avoid reloading) ---
@st.cache_resource
def get_easyocr_reader():
    try:
        reader = easyocr.Reader(['en', 'hi'], gpu=True)
    except:
        reader = easyocr.Reader(['en', 'hi'], gpu=False)
    return reader

# --- Hybrid OCR Function ---
def ocr_page_hybrid_fast(image, min_words=10):
    prepped = preprocess_image(image)
    tess_text = pytesseract.image_to_string(prepped, lang='hin+eng', config='--oem 1 --psm 6 -c preserve_interword_spaces=1')
    
    if len(tess_text.split()) >= min_words:
        return tess_text
    
    reader = get_easyocr_reader()
    easy_text = "\n".join(reader.readtext(np.array(image), detail=0, paragraph=True))
    return easy_text if len(easy_text.split()) > len(tess_text.split()) else tess_text

def quick_text_fix(text):
    fixes = [
        (r'([a-z])([A-Z])', r'\1 \2'),
        (r'([a-zA-Z])(\d)', r'\1 \2'),
        (r'(\d)([a-zA-Z])', r'\1 \2'),
        (r'(\.)([A-Z])', r'\1 \2'),
        (r'\s+', ' ')
    ]
    
    for pattern, repl in fixes:
        text = re.sub(pattern, repl, text)
    return text.strip()

# --- New function to handle single images ---
def process_single_image(image):
    """Process a single uploaded image"""
    st.session_state.progress_bar.progress(20)
    st.session_state.status_text.text("🖼️ Processing image...")
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Perform OCR
    extracted_text = ocr_page_hybrid_fast(image)
    fixed_text = quick_text_fix(extracted_text)
    
    st.session_state.progress_bar.progress(60)
    return fixed_text

# --- Enhanced function to handle multiple images ---
def process_multiple_images(images, max_workers=3):
    """Process multiple uploaded images"""
    total_images = len(images)
    st.session_state.progress_bar.progress(10)
    st.session_state.status_text.text(f"🖼️ Processing {total_images} images in parallel...")
    
    # Convert uploaded files to PIL Images
    pil_images = []
    for img_file in images:
        pil_image = Image.open(img_file)
        pil_images.append(pil_image)
    
    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        ocr_texts = list(executor.map(ocr_page_hybrid_fast, pil_images))
    
    # Combine results
    full_text = ""
    for i, page_text in enumerate(ocr_texts):
        fixed_text = quick_text_fix(page_text)
        full_text += f"\n--- Image {i+1} ---\n{fixed_text}"
        
        progress = int(((i + 1) / total_images) * 60) + 10
        st.session_state.progress_bar.progress(progress)
        st.session_state.status_text.text(f"🖼️ Processed image {i+1}/{total_images}")
    
    return full_text

def fast_pdf_to_text(pdf_path, dpi=200, max_workers=5):
    images = convert_from_path(pdf_path, dpi=dpi)
    total_pages = len(images)
    
    if total_pages > 50:
        st.warning("⚠️ This PDF has more than 50 pages and is marked for review. Processing is skipped to avoid excessive resource usage.")
        st.session_state.progress_bar.empty()
        st.session_state.status_text.empty()
        return None
    
    st.session_state.progress_bar.progress(10)
    st.session_state.status_text.text(f"📄 Processing {total_pages} pages in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        ocr_texts = list(executor.map(ocr_page_hybrid_fast, images))
    
    full_text = ""
    for i, page_text in enumerate(ocr_texts):
        fixed_text = quick_text_fix(page_text)
        full_text += f"\n--- Page {i+1} ---\n{fixed_text}"
        
        progress = int(((i + 1) / total_pages) * 60) + 10
        st.session_state.progress_bar.progress(progress)
        st.session_state.status_text.text(f"📄 Processed page {i+1}/{total_pages}")
    
    return full_text

# [Keep all your existing LLM API functions unchanged]
DEFAULT_LLM_API_URL = "https://cdis.iitk.ac.in/llama_api/llama_api/invoke"

def call_llm_api_fast(prompt, max_tokens=512, llm_api_url=DEFAULT_LLM_API_URL):
    request_body = {"input": {"prompt": prompt}}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    try:
        response = requests.post(llm_api_url, headers=headers, json=request_body, timeout=40)
        response.raise_for_status()
        result = response.json()
        
        for key in ['output', 'response', 'text', 'content', 'result', 'completion']:
            if isinstance(result, dict) and key in result:
                return result[key] if not isinstance(result[key], dict) else str(result[key])
        
        return str(result)
    except Exception as e:
        return f"API Error: {str(e)}"

def test_api_connection(llm_api_url):
    test_prompt = "Hello, respond with 'API working'"
    st.write("🔍 **Testing API Connection**")
    st.write(f"**URL:** {llm_api_url}")
    
    try:
        response = requests.post(llm_api_url, headers={"Content-Type": "application/json"},
                               json={"input": {"prompt": test_prompt}}, timeout=10)
        
        st.write(f"**Status Code:** {response.status_code}")
        
        if response.status_code == 200:
            try:
                st.success("✅ **Connection Successful!**")
                st.code(json.dumps(response.json(), indent=2))
            except:
                st.warning("⚠️ **Response not JSON**")
                st.text(response.text)
        else:
            st.error(f"❌ HTTP Error {response.status_code}")
            st.code(response.text)
    except Exception as e:
        st.error(f"❌ **Connection Failed:** {e}")

def extract_legal_document_data(text, llm_api_url):
    prompt = f"""
Extract and summarize legal document details as JSON:

{{
"document_type": "",
"parties": "",
"effective_dates": "",
"payment_terms": "",
"termination_or_expiry": "",
"key_obligations": "",
"important_clauses": "",
"simplified_summary": ""
}}

Instructions:
Carefully extract the document type (contract, lease, agreement, etc.), all parties involved (individuals, organizations, roles), all important dates (effective dates, expiry dates, deadlines), payment/compensation terms, termination conditions, key obligations for each party, and any important clauses or restrictions.

Write a concise, plain-language summary (2-3 sentences) explaining what the agreement is about, who is involved, and the main obligations or rights.

Focus on extracting information that would be important for someone to understand their rights, obligations, and key terms.

If any detail is missing or unclear, write "MISSING" for that field.

Output only a valid JSON object containing the required fields.

Text: {text[:3000]}
"""
    
    raw = call_llm_api_fast(prompt, max_tokens=500, llm_api_url=llm_api_url)
    
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group()), raw
        else:
            return None, raw
    except Exception:
        return None, raw

def generate_legal_summary_response(data, llm_api_url):
    prompt = f"""
You are a helpful legal assistant. Generate a short, plain-language summary response for any legal document.

Document Type: {data.get('document_type', 'Legal Document')}
Parties: {data.get('parties', '')}
Key Dates: {data.get('effective_dates', '')}
Payment Terms: {data.get('payment_terms', '')}
Key Obligations: {data.get('key_obligations', '')}
Simplified Summary: {data.get('simplified_summary', 'This document contains important legal obligations.')}

Response:

Instructions:
Begin by clearly acknowledging the type of document and its purpose in one sentence.
Summarize the main obligations, rights, or risks in 2–3 sentences using simple, non-technical language.
Highlight any critical dates, payments, or deadlines explicitly.
End with 1–2 practical next steps (what the user should do or avoid).
Keep the entire response under 120 words.
Use respectful, concise, and neutral language.
Conclude with a brief list of "Must Do’s" and "Don’ts" (1–2 points each), written as simple actionable items.  
Avoid giving legal advice—stick only to summarizing and clarifying.
Generate only the response text as instructed.

"""
    
    return call_llm_api_fast(prompt, max_tokens=250, llm_api_url=llm_api_url)

# --- UI Configuration ---
st.set_page_config(page_title="Legal Document Extractor", layout="wide")
st.title("⚖️ Legal Document PDF & Image Extractor")
st.markdown("🔍 **Extract key details + generate plain-language summary from legal documents (PDF/Images) under 30s**")

# Session defaults
if "llm_api_url" not in st.session_state:
    st.session_state.llm_api_url = DEFAULT_LLM_API_URL

###
# API Configuration Sidebar
with st.sidebar:
    st.header("🔧 API Configuration")
    
    api_url = st.text_input("LLM Server URL:", value=st.session_state.llm_api_url)
    if api_url != st.session_state.llm_api_url:
        st.session_state.llm_api_url = api_url
        st.rerun()
    
    st.header("🔍 Connection Testing")
    
    if st.button("🔍 Detailed Connection Test"):
        test_api_connection(st.session_state.llm_api_url)
    
    if st.button("Quick Test"):
        test_response = call_llm_api_fast("Hello, respond with 'API working'", max_tokens=50,
                                        llm_api_url=st.session_state.llm_api_url)
        if "API working" in test_response or not test_response.startswith("API Error"):
            st.success("✅ LLM Server Connected")
        else:
            st.error("❌ API Connection Failed")
            st.error(test_response)
    
    st.info("**Server**: LLAMA API")
    st.info("**Format**: `{\"input\": {\"prompt\": \"...\"}}`")
    st.info("**Features**: Legal document analysis + Plain-language summary")
###
# --- Main Interface ---
dpi_setting = st.selectbox("PDF Image DPI", [200, 300, 400], index=0)
max_workers = st.slider("Parallel OCR Workers", min_value=1, max_value=5, value=4,
                       help="Number of CPU threads for OCR (higher = faster for multi-page PDFs)")

# Enhanced file upload with support for both PDFs and images
st.subheader("📤 Upload Legal Documents")
upload_type = st.radio("Choose upload type:", ["Single PDF", "Single Image", "Multiple Images"], horizontal=True)

if upload_type == "Single PDF":
    uploaded_file = st.file_uploader("Upload Legal Document PDF", type=["pdf"])
    file_type = "pdf"
elif upload_type == "Single Image":
    uploaded_file = st.file_uploader("Upload Legal Document Image", 
                                   type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"])
    file_type = "image"
else:  # Multiple Images
    uploaded_file = st.file_uploader("Upload Multiple Legal Document Images", 
                                   type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"],
                                   accept_multiple_files=True)
    file_type = "images"

if uploaded_file:
    # Reset progress bar and status for each upload
    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    try:
        start = time.time()
        
        if file_type == "pdf":
            # Handle PDF upload (existing code)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            st.session_state.status_text.text("📁 Saving and converting PDF...")
            st.session_state.progress_bar.progress(10)
            
            text = fast_pdf_to_text(tmp_path, dpi=dpi_setting, max_workers=max_workers)
            
            if text is None:
                st.info("This PDF is too long for automatic processing and has been marked for manual review.")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                st.stop()
        
        elif file_type == "image":
            # Handle single image upload
            st.session_state.status_text.text("🖼️ Processing single image...")
            text = process_single_image(uploaded_file)
        
        else:  # Multiple images
            # Handle multiple images upload
            if len(uploaded_file) > 20:
                st.warning("⚠️ More than 20 images uploaded. Processing first 20 to avoid excessive resource usage.")
                uploaded_file = uploaded_file[:20]
            
            st.session_state.status_text.text(f"🖼️ Processing {len(uploaded_file)} images...")
            text = process_multiple_images(uploaded_file, max_workers=max_workers)
        
        st.session_state.progress_bar.progress(80)
        st.session_state.status_text.text("🔎 Extracting legal document data...")
        
        data, raw_resp = extract_legal_document_data(text, st.session_state.llm_api_url)
        
        st.session_state.progress_bar.progress(90)
        
        # Generate summary response if we have valid data
        response = None
        if data and data.get("simplified_summary") and data.get("simplified_summary") != "MISSING":
            st.session_state.status_text.text("💬 Generating plain-language summary...")
            response = generate_legal_summary_response(data, st.session_state.llm_api_url)
        
        st.session_state.progress_bar.progress(100)
        st.session_state.status_text.text("✅ Done!")
        time.sleep(1)
        st.session_state.progress_bar.empty()
        st.session_state.status_text.empty()
        
        st.success(f"✅ Processed in {round(time.time() - start, 2)} seconds!")
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Document Stats")
            st.metric("Total Words", f"{len(text.split()):,}")
        
        with col2:
            st.subheader("📋 Extracted Legal Details")
            if data:
                st.json(data)
                
                # Create download filename based on upload type
                if file_type == "pdf":
                    filename_base = uploaded_file.name
                elif file_type == "image":
                    filename_base = uploaded_file.name
                else:  # multiple images
                    filename_base = f"{len(uploaded_file)}_images"
                
                st.download_button("📥 Download JSON", json.dumps(data, indent=2),
                                 f"{filename_base}_legal_data.json")
            else:
                st.warning("No valid data extracted. Check the content or LLM API response.")
        
        # Display document summary
        if data and "simplified_summary" in data and data["simplified_summary"] and data["simplified_summary"] != "MISSING":
            st.subheader("📝 Document Summary")
            st.info(data["simplified_summary"])
        
        # Display generated plain-language response
        if response:
            st.subheader("💡 Plain-Language Explanation")
            st.success(response)
            
            response_json = json.dumps({"plain_language_summary": response}, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 Download Summary (JSON)",
                response_json,
                file_name=f"{filename_base}_summary.json",
                mime="application/json"
            )
        
        # Optional raw text display
        if st.checkbox("📃 Show Raw Extracted Text"):
            st.text_area("Extracted Text", value=text[:2000] + "...", height=250)
            st.download_button("📥 Download Full Text", text, f"{filename_base}_full_text.txt")
        
        # Debug section for troubleshooting
        if not data:
            with st.expander("Show Raw LLM Response (Debug)"):
                st.text(raw_resp)
    
    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
        st.text(traceback.format_exc())
    
    finally:
        if file_type == "pdf" and 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("👆 Upload a legal document (PDF or Image) to begin analysis")

# Example usage section
with st.expander("📋 What types of documents can I analyze?"):
    st.markdown("""
**Supported Document Types:**
- Employment contracts and agreements
- Rental and lease agreements  
- Service agreements and contracts
- Terms of service and privacy policies
- Purchase agreements and sales contracts
- Non-disclosure agreements (NDAs)
- Partnership agreements
- Licensing agreements

**Supported File Formats:**
- **PDF**: Multi-page legal documents
- **Images**: PNG, JPG, JPEG, TIFF, BMP, WebP
- **Multiple Images**: Process several document images at once

**What you'll get:**
- Key parties and roles identified
- Important dates and deadlines extracted
- Payment terms and obligations summarized
- Plain-language explanation of main terms
- Downloadable JSON data for further analysis
""")
