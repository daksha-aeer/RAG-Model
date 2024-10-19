from PyPDF2 import PdfReader
import docx
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_file(file):
    """Extract text from uploaded file based on its type."""
    if file.name.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        text = file.read().decode('utf-8')
    print(f"Extracted Text from {file.name}:\n{text}")  # Debugging line
    return text


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        print(f"Extracted from page: {page_text}")  # Debugging line
        text += page_text
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    print(f"Extracted DOCX Text: {text}")  # Debugging line
    return text


def split_text_into_chunks(text, chunk_size=1500, overlap=100):
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    print(f"Generated {len(chunks)} chunks.")  # Debugging line
    return chunks
