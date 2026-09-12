import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def process_pdf(pdf_path: str):
    load_dotenv(override=True)

    # 1. Parse PDF using pypdf
    reader = PdfReader(pdf_path)
    docs = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(page_content=text, metadata={"page": page_num + 1}))

    if not docs:
        raise ValueError("The uploaded PDF contains no readable text. Make sure you upload a digital text-based PDF.")

    # 2. Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)

    # 3. Local Vector Store Setup
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

    # Updated model string to gemini-3.6-flash
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