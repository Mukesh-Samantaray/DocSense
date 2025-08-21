
# DocSense

**DocSense** is a multimodal AI assistant that "understands" documents—extracting text, images, and graphs from PDFs. Ask questions, get answers backed by retrieved content, and visualize all images/graphs. Ideal for research, business reports, or educational materials; demonstrates PDF comprehension and multimodal reasoning.

---

## 🚀 Features

- **Multimodal Understanding**: Extracts and interprets text, images, and graphs from PDF documents.
- **Interactive Q&A**: Engage in natural language queries to retrieve specific information from documents.
- **Visual Insights**: Displays images and graphs embedded within PDFs for comprehensive understanding.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with AI-generated responses for accurate information delivery.

---

## 🎯 Use Cases

- **Academic Research**: Quickly analyze research papers, theses, and academic articles.
- **Business Analysis**: Extract key insights from financial reports, presentations, and business documents.
- **Education**: Assist students in understanding textbooks, lecture notes, and study materials.
- **Legal & Compliance**: Review and interpret contracts, legal documents, and compliance reports.

---

## 🛠 Technologies

- **Gemini API**: Advanced AI-driven document analysis.
- **LangChain**: Framework for building applications with LLMs.
- **Chroma**: Vector database for storing and retrieving document embeddings.
- **Streamlit**: Framework for creating interactive web applications.
- **PyMuPDF & pdfplumber**: Extract text and images from PDFs.
- **Pillow & pytesseract**: Image processing and OCR (Optional).

---

## 🧪 Test Case

**Objective**: Verify DocSense can accurately extract and display information from a sample PDF.

**Steps**:

1. Upload a PDF with text, images, and graphs.
2. Ask a question (e.g., "What is the revenue growth in Q2 2025?").
3. Check that the response includes relevant info and visuals.

**Expected Result**: DocSense provides a precise answer with visual context from the PDF.

---

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/Mukesh-Samantaray/DocSense.git
cd DocSense
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:

```
GEMINI_API_KEY=your_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

4. Run the app:

```bash
streamlit run app.py
```

---

## 🤝 Contributing

Contributions are welcome! Fork the repo, create a branch, and submit a pull request.

---

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📢 Acknowledgments

- **Gemini API**: AI capabilities.  
- **LangChain**: LLM integration.  
- **Chroma**: Document embedding storage.  
- **Streamlit**: Interactive web apps.  
- **PyMuPDF & pdfplumber**: PDF parsing.  
- **Pillow & pytesseract**: Image processing and OCR.
