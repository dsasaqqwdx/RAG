"""
Loads every PDF in data/papers/ into LangChain Document objects.
"""
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from app.config import PAPERS_DIR


def load_papers():
    loader = DirectoryLoader(
        PAPERS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs in {PAPERS_DIR}")
    return documents


if __name__ == "__main__":
    docs = load_papers()
    if docs:
        print("Example metadata:", docs[0].metadata)
        print("Example content snippet:", docs[0].page_content[:200])
    else:
        print("No PDFs found — add some to data/papers/ first.")