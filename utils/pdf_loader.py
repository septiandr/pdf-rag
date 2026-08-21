from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf_files(folder_path: Path):
    loader = PyPDFLoader(str(folder_path))
    documents = loader.load()

    # Gabungkan teks per halaman
    full_text = "\n".join([doc.page_content for doc in documents])

    # Split berdasarkan section (BAB, Pasal, dll)
    sections = []
    current_section = ""
    section_headers = ["BAB", "Pasal", "Bagian", "Ayat"]

    for line in full_text.split("\n"):
        is_header = any(line.strip().startswith(h) for h in section_headers)

        if is_header and current_section:
            sections.append(current_section.strip())
            current_section = line + "\n"
        else:
            current_section += line + "\n"

    if current_section.strip():
        sections.append(current_section.strip())

    # Jika tidak ada section, fallback ke recursive splitter
    if len(sections) <= 1:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        return chunks

    # Buat document objects dari sections
    from langchain_core.documents import Document
    chunks = []
    for section in sections:
        # Skip section terlalu pendek
        if len(section) < 50:
            continue
        chunks.append(Document(
            page_content=section,
            metadata={"source": str(folder_path)}
        ))

    # Sub-split untuk section terlalu panjang
    final_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for chunk in chunks:
        if len(chunk.page_content) > 800:
            sub_chunks = splitter.split_documents([chunk])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks