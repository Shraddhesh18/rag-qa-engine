import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from pypdf import PdfReader
import pypdfium2 as pdfium
import pytesseract

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Point pytesseract to standard Windows installation path
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def process_pdf(pdf_path: str):
    load_dotenv(override=True)

    docs = []
    
    # 1. Parse PDF using pypdf (using context manager to release lock)
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(page_content=text, metadata={"page": page_num + 1}))

    # 2. Fallback to Tesseract OCR using pypdfium2
    if not docs:
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                image = page.render(scale=2).to_pil()
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text and ocr_text.strip():
                    docs.append(Document(page_content=ocr_text, metadata={"page": page_num + 1}))
            pdf.close()  # Close pdfium object to release file handle
        except Exception as e:
            raise ValueError(
                f"OCR processing failed. Ensure Tesseract-OCR is installed. Details: {e}"
            )

    if not docs:
        raise ValueError("The uploaded PDF contains no readable text, even after OCR analysis.")

    # 3. Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)

    # 4. Local Vector Store Setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    
    return vectorstore

def get_qa_chain(vectorstore):
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        temperature=0.2, 
        google_api_key=api_key
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you do not know the answer, say that you "
        "don't know based on the provided document. Do NOT make up an answer. "
        "Keep the response precise and factual.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain