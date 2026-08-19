Chunking and text-splitter recommendations

Goal: avoid splitting clause-level information (e.g., contractual conditions) across chunks and increase overlap to preserve clause continuity.

Recommended parameters:
- chunk_size_tokens: 600 (range 500–800)
- chunk_overlap_tokens: 175 (range 150–200)
- prefer splitting on paragraph or sentence boundaries

LangChain-style example (Python):
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=600, chunk_overlap=175)
chunks = splitter.split_text(document_text)

Indexing tips:
- When indexing PDFs, preserve page numbers and paragraph offsets in metadata: {"source":"Offer Letter Mphasis.pdf","page":3,"para":2}
- If using OCR, run light clean-up so line breaks correspond to sentences/paragraphs.
- For legal/contract docs, consider larger chunk_size (700–800) to keep multi-clause paragraphs intact.

Validation check:
- After re-chunking, run an automated check that the phrase "Joining Bonus" and the surrounding 100 tokens appear together in at least one chunk for a sample Offer Letter.
