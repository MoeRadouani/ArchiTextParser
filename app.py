import streamlit as st
import spacy
from collections import defaultdict

# Load the trained spaCy model
nlp = spacy.load("my_custom_model") 

# Function to process the input text using the trained model
def process_text(paragraph, nlp):
    doc = nlp(paragraph)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return doc, entities

# Function to highlight entities in the paragraph
def highlight_entities(doc):
    colors = {
        "FEATURE": "#ADD8E6",   # Light Blue
        "MATERIAL": "#90EE90",  # Light Green
        "STYLE": "#DDA0DD",     # Plum
    }

    highlighted_text = ""
    last_end = 0

    for ent in doc.ents:
        highlighted_text += doc.text[last_end:ent.start_char]
        color = colors.get(ent.label_, "#DDD")
        highlighted_text += (
            f"<span style='background-color: {color}; color:black; "
            f"padding:2px 5px; border-radius:4px;'>{ent.text}</span>"
        )
        last_end = ent.end_char

    highlighted_text += doc.text[last_end:]
    return highlighted_text

# Function to display the extracted information as a table
def display_extracted_information(entities):
    extracted_info = []
    for label in ["FEATURE", "MATERIAL", "STYLE"]:
        extracted_info.append({
            "Category": f"{label.title()}s",
            "Extracted": ", ".join(set(entities.get(label, []))) or "—"
        })
    return extracted_info

# ✅ Set the browser tab title and icon at the very top of your script
st.set_page_config(
    page_title="ArchiParser - Building Text Analyzer",  # This sets the browser tab title
    page_icon="🏛️"  # Optional: sets the tab icon (favicon)
)

# Streamlit interface setup
st.title("🏛️ Historical Buildings Analyzer ")



# Get user input
paragraph = st.text_area(
    "Paste your paragraph here",
    height=180,
    placeholder="e.g. The Gothic cathedral features stained glass windows, marble floors, and tall spires."
)

# Analyze button
if st.button("🔍 Analyze"):
    if not paragraph.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        doc, entities = process_text(paragraph, nlp)
        extracted_info = display_extracted_information(entities)

        # Highlighted paragraph
        st.subheader("🖍️ Highlighted Text")
        st.markdown(highlight_entities(doc), unsafe_allow_html=True)

        # Legend for colors (moved below highlighted text)
        st.markdown(
    """
    <div style='text-align:center; padding:10px;'>
        <span style='background:LightSkyBlue; color:black; padding:5px 15px; margin-right:10px; display:inline-block; min-width:80px; text-align:center; border-radius: 6px;'>Feature</span>
        <span style='background:PaleGreen; color:black; padding:5px 15px; margin-right:10px; display:inline-block; min-width:80px; text-align:center; border-radius: 6px;'>Material</span>
        <span style='background:Plum; color:black; padding:5px 15px; display:inline-block; min-width:80px; text-align:center;border-radius: 6px;'>Style</span>
    </div>
    """, 
    unsafe_allow_html=True
)

        # Display extracted table
        st.subheader("📊 Extracted Information")
        st.table(extracted_info)
