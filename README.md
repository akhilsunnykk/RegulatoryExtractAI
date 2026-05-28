# RegulatoryExtractAI 🕵️‍♂️📜

RegulatoryExtractAI is a local, lightweight Retrieval-Augmented Generation (RAG) pipeline designed to evaluate messy, unstructured user documents against complex, dense regulatory manuals. 

Built with **Python**, **LangChain**, **Google Gemini**, and **ChromaDB**, this tool automates the compliance auditing process by extracting structured data from user inputs and semantically verifying it against official rulebooks.

## 🚀 Features

* **Semantic Search Indexing:** Ingests and chunks regulatory `.txt` manuals, embedding them into a local Chroma vector database using Google's `gemini-embedding-001`.
* **Structured Data Extraction:** Uses `gemini-2.5-flash` and Pydantic to parse messy, poorly-formatted user emails/applications and extract strict JSON data.
* **Automated Compliance Engine:** Orchestrates a final LLM evaluation, comparing the extracted user data against the exact retrieved regulatory rules to output a definitive Compliance Verdict (Pass/Fail + Rationale).

## 📁 Project Structure

```text
RegulatoryExtractAI/
│
├── data/
│   ├── messy_inputs/         # Put sample user applications here (.txt)
│   └── regulatory_docs/      # Put official compliance manuals here (.txt)
│
├── src/
│   ├── indexer.py            # Chunks and embeds regulations into ChromaDB
│   ├── extractor.py          # Extracts structured JSON from messy user files
│   └── evaluator.py          # The orchestrator: connects extraction to retrieval
│
├── chroma_db/                # Local vector database (auto-generated)
├── .env                      # API keys (not tracked in git)
├── .gitignore
└── README.md

🛠️ Installation & Setup (Windows / PowerShell)
1. Clone the repository

PowerShell
git clone [https://github.com/akhilsunnykk/RegulatoryExtractAI.git](https://github.com/akhilsunnykk/RegulatoryExtractAI.git)
cd RegulatoryExtractAI
2. Create and activate a virtual environment

PowerShell
python -m venv venv
.\venv\Scripts\activate
3. Install dependencies

PowerShell
pip install langchain langchain-google-genai langchain-community chromadb pydantic python-dotenv
4. Configure Environment Variables
Create a .env file in the root directory and add your Google Gemini API key:

Plaintext
GOOGLE_API_KEY=your_gemini_api_key_here
💻 Usage
Step 1: Populate your data folders
Add your ground-truth regulatory text files to data/regulatory_docs/ and the user applications to test to data/messy_inputs/.

Step 2: Build the Vector Index
Run the indexer to embed your regulatory rules. You only need to run this when your rules change.

PowerShell
python src/indexer.py
Step 3: Run the Compliance Evaluator
Run the orchestrator to extract data from the user document and test it against the rules.

PowerShell
python src/evaluator.py
🔧 Technology Stack
Language: Python

Orchestration: LangChain

LLM & Embeddings: Google Gemini (gemini-2.5-flash, gemini-embedding-001)

Vector Database: ChromaDB

Data Validation: Pydantic