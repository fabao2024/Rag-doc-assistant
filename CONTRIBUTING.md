# Contributing to RAG Document Q&A

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)
- Relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Possible implementation approach (optional)

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

### Testing

- Test your changes thoroughly
- Ensure existing tests still pass
- Add new tests for new features

## Development Setup

```bash
# Clone your fork
git clone https://github.com/fabao2024/Rag-doc-assistant.git
cd Rag-doc-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_imports.py
```

## Questions?

Feel free to open an issue for any questions or discussions!

Thank you for contributing! 🎉
