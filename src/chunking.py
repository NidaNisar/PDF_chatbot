from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document(document,chunk_size=1000,chunk_overlap=200):
    """
    Chunk size is how much text you put into each piece (chunk) before sending it to an embedding model or LLM.
     Think of chunk overlap as repeating the last part of one chunk at the beginning of the next chunk. 
    """
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        length_function=len,
        separators=["\n\n","\n"," ",""]
    )
    split_docs=text_splitter.split_documents(document)
    print(f"split documents {len(document) } into the {len(split_docs)}")

    if split_docs:
        print(f"Example chunk ")
        print(f"Content {split_docs[0].page_content[:200]}...")
        print(f"Metadata {split_docs[1].metadata}...")
    return split_docs

"""
\n\n — Paragraph boundary

First, try splitting at blank lines between paragraphs.

\n — Line boundary

If paragraphs are too large, try splitting at individual line breaks.

" " — Word boundary

If the text is still too large, split at spaces between words.

"" — Character boundary

As a last resort, split at individual characters.
"""
        
