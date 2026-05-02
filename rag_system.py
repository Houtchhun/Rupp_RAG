from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings_provider import LocalEmbeddings
from pypdf import PdfReader

import os
import importlib


def load_pdf_documents(source_pdf: str):
    reader = PdfReader(source_pdf)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": source_pdf, "page": page_number},
            )
        )
    return documents


def get_text_splitter_class():
    """Resolve RecursiveCharacterTextSplitter across LangChain package variants."""
    for module_name in ("langchain.text_splitter", "langchain_text_splitters"):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "RecursiveCharacterTextSplitter")
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "RecursiveCharacterTextSplitter is not available. Install langchain-text-splitters."
    )


def build_index(
    source_pdf: str = "data/RUPP Prospectus2018.pdf",
    output_dir: str = "rupp_index",
):
    # Load PDF
    documents = load_pdf_documents(source_pdf)

    # Split text
    RecursiveCharacterTextSplitter = get_text_splitter_class()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)

    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embeddings = LocalEmbeddings(model_name=model_name)

    # Create vector database and save locally
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(output_dir)
    print(f"Index built successfully at: {output_dir}")


if __name__ == "__main__":
    build_index()