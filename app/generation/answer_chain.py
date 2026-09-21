"""
Takes retrieved chunks + a question, builds a strict prompt, and calls the LLM
via OpenRouter (OpenAI-compatible endpoint).
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)

SYSTEM_PROMPT = """You are a research assistant answering questions about a set of academic papers.

Rules:
- Answer ONLY using the provided context below. Do not use outside knowledge.
- If the context does not contain the answer, say "I don't have enough information in the provided papers to answer that."
- The context below is broken into labeled sections, each starting with a line like: [some_filename.pdf, page 8]
- When you cite a claim, copy that exact label from the section it came from — the real filename and real page number, never the literal placeholder words "source" or "page".
- Be precise with numbers, metrics, and model names — do not approximate or guess.

Context:
{context}
"""


def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=800,  
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def format_context(chunks):
    formatted = []
    for doc in chunks:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        page = int(page) + 1 if isinstance(page, (int, float)) else page 
        formatted.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def generate_answer(question: str, chunks: list) -> str:
    llm = get_llm()
    context = format_context(chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


if __name__ == "__main__":
    from app.retrieval.vector_retriever import get_retriever

    retriever = get_retriever()
    question = "What BLEU score did the base Transformer model achieve on the WMT 2014 English-to-German translation task?"
    chunks = retriever.invoke(question)

    print(f"\n--- Retrieved {len(chunks)} chunks ---")
    for i, doc in enumerate(chunks[:5]):
        print(f"\nChunk {i+1} (source: {doc.metadata.get('source')}, page: {doc.metadata.get('page')}):")
        print(doc.page_content[:200])
    print("\n--- End chunks ---\n")

    answer = generate_answer(question, chunks)
    print(answer)