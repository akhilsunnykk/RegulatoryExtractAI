import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import the extraction logic we built in the previous step
from extractor import extract_from_messy_doc

load_dotenv()

# 1. Define the Structured Output for the Final Evaluation
class ComplianceEvaluation(BaseModel):
    is_compliant: bool = Field(description="True if the application strictly meets all regulatory requirements, False otherwise.")
    violations: list[str] = Field(description="List of specific regulatory violations or missing required fields. Empty if compliant.")
    rationale: str = Field(description="Detailed explanation of the compliance decision citing the retrieved regulatory rules.")

def evaluate_application(messy_doc_path, chroma_db_dir="./chroma_db"):
    # Step 1: Extract the structured data
    print("1. Extracting data from messy document...")
    extracted_data = extract_from_messy_doc(messy_doc_path)
    print(f"   -> Found applicant: {extracted_data.applicant_name} (Risk: {extracted_data.risk_level})")

    # Step 2: Connect to the existing Chroma Vector Database
    print("2. Connecting to regulatory database...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)

    # Step 3: Formulate a targeted semantic search query
    # We use the extracted data to find the EXACT rules that apply to this application
    search_query = f"Compliance requirements, required fields, and rules for risk level: {extracted_data.risk_level}."
    
    print("3. Retrieving relevant regulatory rules...")
    retrieved_docs = vectorstore.similarity_search(search_query, k=3) # Fetch top 3 most relevant chunks
    regulatory_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 4: Evaluate against the rules using Gemini
    print("4. Generating compliance evaluation...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_evaluator = llm.with_structured_output(ComplianceEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict regulatory compliance officer. Compare the applicant's extracted data against the official regulatory documents provided. Determine if they are compliant. Do not make up rules; rely strictly on the provided regulatory context."),
        ("human", """
        *** Applicant Extracted Data ***
        {applicant_data}
        
        *** Official Regulatory Context ***
        {regulatory_context}
        """)
    ])

    # Execute the evaluation chain
    chain = prompt | structured_evaluator
    result = chain.invoke({
        "applicant_data": extracted_data.model_dump_json(),
        "regulatory_context": regulatory_context
    })

    return result

if __name__ == "__main__":
    sample_path = "../data/messy_inputs/sample_application.pdf"
    
    if os.path.exists(sample_path):
        final_report = evaluate_application(sample_path)
        print("\n=== FINAL COMPLIANCE REPORT ===")
        print(final_report.model_dump_json(indent=2))
    else:
        print("Please ensure a sample PDF exists at data/messy_inputs/sample_application.pdf")