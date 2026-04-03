import io
import json
import base64

import streamlit as st
from PIL import Image
import pdfplumber
import fitz  # PyMuPDF
from openai import OpenAI
import os

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Invoice Processing Assistant",
    page_icon="📄",
    layout="wide"
)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY n'a pas été trouvée dans les variables d'environnement.")

client = OpenAI(api_key=api_key)

PROMPT_INSTRUCTIONS = """
Tu es un assistant spécialisé dans l'extraction de données à partir de factures.

À partir du document fourni, extrait les informations suivantes :
- fournisseur
- date_facture
- montant_total
- montant_tva
- devise

Règles :
- Réponds uniquement en JSON valide
- Ne mets aucun texte avant ou après le JSON
- Si une information est absente, mets null
- Le montant_total et montant_tva doivent être des nombres (pas de texte)
- La date doit être au format YYYY-MM-DD si possible

Format attendu :
{
  "fournisseur": "...",
  "date_facture": "...",
  "montant_total": ...,
  "montant_tva": ...,
  "devise": "..."
}
"""

# =========================
# STYLE
# =========================

st.markdown("""
<style>
div[data-testid="stFileUploader"] button {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
}

div[data-testid="stFileUploader"] button:hover {
    background-color: #1D4ED8 !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #2563EB;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("📄 Invoice AI")

    st.markdown("## Comment ça fonctionne")
    st.markdown("""
Cette application permet d'extraire automatiquement les informations clés d'une facture grâce à l'IA.

### Étapes :
1. **Importer une facture**
   - Format accepté : PDF ou image

2. **Analyse du document**
   - Si le PDF contient du texte → extraction directe
   - Sinon → lecture visuelle par IA
   - Pour les images → lecture visuelle par IA

3. **Traitement par IA**
   - Le document est envoyé au modèle
   - Les données sont structurées en JSON

4. **Affichage des résultats**
   - La facture est affichée à gauche
   - Les champs extraits sont affichés à droite
    """)

    st.markdown("---")

    st.markdown("## Technologies utilisées")
    st.markdown("""
- PDF texte : pdfplumber  
- Rendu PDF en image : PyMuPDF  
- IA : OpenAI  
- Interface : Streamlit  
    """)

    st.markdown("---")

    st.markdown("## Cas d’usage")
    st.markdown("""
- Automatisation comptable  
- Extraction de données fournisseurs  
- Pré-traitement avant ERP  
- Analyse documentaire  
    """)

# =========================
# UI
# =========================

st.title("Assistant de traitement de factures")
st.subheader("Téléchargez une facture (image ou PDF), puis visualisez et structurez les données avec l'IA.")

uploaded_file = st.file_uploader(
    "Importer une facture",
    type=["png", "jpg", "jpeg", "pdf"]
)

with st.expander("Voir le prompt utilisé par l’IA"):
    st.code(PROMPT_INSTRUCTIONS, language="text")

# =========================
# HELPERS
# =========================

def clean_json_response(content: str) -> dict:
    """Nettoie une éventuelle réponse markdown et parse le JSON."""
    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()

    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_output": content}


def pil_image_to_base64(image: Image.Image) -> str:
    """Convertit une image PIL en base64 JPEG."""
    buffer = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extraction directe du texte d'un PDF texte."""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()


def pdf_bytes_to_images(pdf_bytes: bytes, zoom: float = 2.0, max_pages: int = 5):
    """
    Convertit un PDF en liste d'images PIL via PyMuPDF.
    max_pages limite le coût et la latence.
    """
    images = []
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_count = min(len(pdf_doc), max_pages)

    for page_num in range(page_count):
        page = pdf_doc.load_page(page_num)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    pdf_doc.close()
    return images


def display_pdf(pdf_bytes: bytes):
    """Affiche le PDF natif dans Streamlit."""
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="850"
            type="application/pdf"
            style="border: 1px solid #E5E7EB; border-radius: 12px;">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_pdf_as_images(pdf_bytes: bytes):
    """Affiche les pages du PDF sous forme d'images."""
    st.markdown("### 🖼️ Aperçu des pages PDF")
    images = pdf_bytes_to_images(pdf_bytes)
    for i, image in enumerate(images):
        st.image(image, caption=f"Page {i + 1}", use_container_width=True)
    return images


def render_field(label: str, value):
    display_value = "—" if value in [None, ""] else value
    st.markdown(
        f"""
        <div style="
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 14px 16px;
            background-color: white;
            margin-bottom: 12px;
        ">
            <div style="
                font-size: 12px;
                color: #6B7280;
                margin-bottom: 6px;
                font-weight: 500;
            ">
                {label}
            </div>
            <div style="
                font-size: 22px;
                color: #111827;
                font-weight: 600;
                line-height: 1.2;
            ">
                {display_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_invoice_fields(data: dict):
    fournisseur = data.get("fournisseur")
    date_facture = data.get("date_facture")
    montant_total = data.get("montant_total")
    montant_tva = data.get("montant_tva")
    devise = data.get("devise")

    st.subheader("Détails de la facture")

    col1, col2 = st.columns(2)

    with col1:
        render_field("Fournisseur", fournisseur)
        render_field("Date de facture", date_facture)
        render_field("Devise", devise)

    with col2:
        render_field("Montant total", montant_total)
        render_field("Montant TVA", montant_tva)


def extract_invoice_data_from_text(text: str) -> dict:
    """Extraction structurée à partir du texte brut."""
    prompt = PROMPT_INSTRUCTIONS + f"\n\nTexte de la facture :\n{text}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        content = response.choices[0].message.content
        return clean_json_response(content)

    except Exception as e:
        return {"error": f"Erreur OpenAI (texte) : {str(e)}"}


def extract_invoice_data_from_images(images: list[Image.Image]) -> dict:
    """
    Extraction structurée depuis une ou plusieurs images
    grâce au modèle vision.
    """
    try:
        content_blocks = [
            {
                "type": "text",
                "text": PROMPT_INSTRUCTIONS
            }
        ]

        for image in images:
            image_b64 = pil_image_to_base64(image)
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": content_blocks
                }
            ],
        )

        content = response.choices[0].message.content
        return clean_json_response(content)

    except Exception as e:
        return {"error": f"Erreur OpenAI (vision) : {str(e)}"}


# =========================
# MAIN LOGIC
# =========================

if uploaded_file:
    data = None

    st.write("Type de fichier :", uploaded_file.type)

    left_col, right_col = st.columns([1.15, 1], gap="large")

    with left_col:
        st.subheader("Facture")

        if uploaded_file.type == "application/pdf":
            st.info("📄 PDF détecté")
            pdf_bytes = uploaded_file.getvalue()

            display_pdf(pdf_bytes)

            with st.expander("Afficher aussi le PDF page par page"):
                preview_images = display_pdf_as_images(pdf_bytes)

            extracted_text = extract_text_from_pdf_bytes(pdf_bytes)

            if extracted_text.strip():
                st.success("✅ Texte extrait directement du PDF")
                extraction_mode = "text"
            else:
                st.warning("⚠️ Aucun texte détecté. Passage à l’analyse visuelle par IA.")
                extraction_mode = "vision"
                preview_images = pdf_bytes_to_images(pdf_bytes)

        else:
            st.info("🖼️ Image détectée")
            image = Image.open(uploaded_file)
            st.image(image, caption="Facture importée", use_container_width=True)
            extraction_mode = "vision"

    with right_col:
        with st.spinner("Analyse du document en cours..."):
            if uploaded_file.type == "application/pdf":
                if extraction_mode == "text":
                    data = extract_invoice_data_from_text(extracted_text)
                else:
                    data = extract_invoice_data_from_images(preview_images)
            else:
                image = Image.open(uploaded_file)
                data = extract_invoice_data_from_images([image])

        if data:
            if "error" not in data and "raw_output" not in data:
                display_invoice_fields(data)

                with st.expander("Voir le JSON extrait"):
                    st.json(data)
            else:
                st.warning("Impossible d'afficher les champs structurés.")
                st.json(data)