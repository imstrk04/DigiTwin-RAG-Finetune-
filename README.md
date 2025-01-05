# 🧠 DocQuery AI – Intelligent document querying made simple
Welcome to the **DocQuery AI** repository! This project enables users to upload a PDF document and interact with it by asking questions. The app retrieves relevant information from the PDF using **Retrieval-Augmented Generation (RAG)** and provides meaningful answers using a **LLM (Large Language Model)**. Additionally, the repository includes a script to measure **semantic similarity** between a question and two possible answers.

## 📁 **Repository Structure**
```
📦YourRepoName
 ┣ 📂pdfs
 ┃ ┗ 📄 paper-rag.pdf
 ┣ 📄 data.json
 ┣ 📄 app.py
 ┣ 📄 similarity.py
 ┗ 📄 README.md
```

### **Files Overview:**
- **`app.py`**: The main Streamlit app that allows users to upload a PDF and interact with it through a chat interface.
- **`similarity.py`**: Contains the `semantic_similarity` function to calculate cosine similarity between a question and two answers.
- **`data.json`**: A sample dataset containing questions, answers, and context for RAG fine-tuning.
- **`paper-rag.pdf`**: A PDF document used to test the RAG app by asking context-based questions.

---

## 🚀 **App Workflow (RAG)**
### **Step 1: PDF Upload and Vector Embedding**
The app allows users to upload a PDF, which is loaded and split into chunks for vector embedding. Here's how it works:

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Load PDF
st.session_state.loader = PyPDFDirectoryLoader("pdfs")
st.session_state.docs = st.session_state.loader.load()

# Split text into chunks
st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

# Generate vector embeddings
st.session_state.vectors = FAISS.from_documents(
    st.session_state.final_documents,
    st.session_state.embeddings_model
)
```

### **Step 2: Document Summarization**
The uploaded PDF is summarized using the **Ollama LLM API**.

```python
def summarize_document():
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = summary_chain.run(text=full_text[:4000])
    return summary
```

### **Step 3: Question-Answer Interaction**
Users can ask questions, and the app retrieves relevant chunks from the PDF to generate context-aware responses.

```python
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question:
    <context>
    {context}
    </context>
    Questions: {input}
    """
)
```

---

## 📊 **Semantic Similarity Script**
The `semantic_similarity` function measures the similarity between a question and two answers using **Sentence-BERT (SBERT)**. This can help validate which answer is more relevant to a given question.

### **Code:**
```python
from sentence_transformers import SentenceTransformer, util

def semantic_similarity(question, answer1, answer2):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained Sentence-BERT model
    embeddings = model.encode([question, answer1, answer2], convert_to_tensor=True)

    similarity1 = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    similarity2 = util.pytorch_cos_sim(embeddings[0], embeddings[2]).item()

    return (similarity1, similarity2)
```

### **Example Usage:**
```python
question = "What is the capital of France?"
answer1 = "The capital of France is Paris."
answer2 = "Berlin is the capital of Germany."

similarity1, similarity2 = semantic_similarity(question, answer1, answer2)
print(f"Similarity with answer1: {similarity1:.2f}")
print(f"Similarity with answer2: {similarity2:.2f}")
```
**Output:**
```
Similarity with answer1: 0.95
Similarity with answer2: 0.45
```

---

## 📄 **Data Explanation (`data.json`)**
The `data.json` file contains structured data for RAG fine-tuning. It includes a list of questions, their corresponding answers, and the context from which the answers were derived.

### **Sample Format:**
```json
{
  "questions": [
    {
      "question": "What is the main topic of the paper?",
      "answer": "The paper discusses Retrieval-Augmented Generation (RAG) for document-based QA systems.",
      "context": "The paper explores how RAG combines the power of retrieval and generation for better document understanding."
    }
  ]
}
```

This file can be used to fine-tune the LLM to provide better, context-aware responses.

---

## 📘 **Test PDF (`paper-rag.pdf`)**
The app includes a sample PDF (`paper-rag.pdf`) that users can upload to test the functionality. This PDF contains technical content about RAG, and users can ask questions like:
- *"What is Retrieval-Augmented Generation?"*
- *"How does RAG improve document retrieval?"*

The app will retrieve relevant chunks from the PDF and generate answers using the LLM.

---

## 💻 **Running the App**
### **Step 1: Install Dependencies**
```bash
pip install streamlit langchain sentence-transformers faiss-cpu
```

### **Step 2: Run the Streamlit App**
```bash
streamlit run app.py
```

---

## 🔧 **Customization Options**
You can adjust the following parameters for better performance:
- **`chunk_size`**: Controls the size of document chunks for vector embedding (default: 700).
- **`chunk_overlap`**: Controls the overlap between chunks (default: 50).

Adjust these values based on your document size and LLM capabilities.

---

## 🧪 **Future Enhancements**
- Add support for multi-PDF uploads.
- Implement fine-tuning of the LLM using `data.json`.
- Enhance semantic similarity scoring with more advanced models.

---

## 🤖 **Contributions**
Contributions are welcome! Please open an issue or submit a pull request to suggest improvements.

---

## 📜 **License**
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

Happy Coding! 🚀

