# Documents Folder

Place your PDF files in this directory to be processed by the RAG system.

## Supported Formats

- PDF files (`.pdf`)

## Example Structure

```
documents/
├── manual.pdf
├── research_paper.pdf
└── technical_docs.pdf
```

## Notes

- Files in this directory are automatically loaded when you run `rag_script.py`
- The system processes all PDFs recursively
- Large files (>10MB) may take longer to process
- Ensure PDFs are text-based (not scanned images) for best results

## Sample Document

A sample document (`Ora-Funky-Cat-owners-manual.pdf`) is included for testing purposes.

To use your own documents:
1. Add your PDF files to this folder
2. Delete or move the sample document (optional)
3. Run `rag_script.py` to rebuild the vector store
