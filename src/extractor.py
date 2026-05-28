import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 1. Define the Structured Output you want
class ExtractedData(BaseModel):
    applicant_name: str = Field(description="The full name of the applicant")
    compliance_id: str = Field(description="The alphanumeric compliance ID, if present")
    risk_level: str = Field(description="Assess as LOW, MEDIUM, or HIGH based on the text")
    missing_fields: list[str] = Field(description="Any required regulatory fields missing from the document")

def extract_from_messy_doc(file_path):
    # Load the messy document
    loader = PyPDFLoader(file_path)
    messy_text = "\n".join([doc.page_content for doc in loader.load()])

    # 2. Setup the Gemini LLM
    # Temperature 0 keeps the model deterministic for data extraction
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 
    
    # Force the model to output JSON matching our Pydantic schema
    structured_llm = llm.with_structured_output(ExtractedData)

    # 3. Prompt Design
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert regulatory compliance auditor. Extract the required information from the messy user text below. If a specific ID is not found, output 'NOT_FOUND'."),
        ("human", "Messy User Document:\n{document_text}")
    ])

    # 4. Execute the chain
    chain = prompt | structured_llm
    result = chain.invoke({"document_text": messy_text})
    
    return result

if __name__ == "__main__":
    sample_path = "../data/messy_inputs/sample_application.pdf"
    if os.path.exists(sample_path):
        extracted_json = extract_from_messy_doc(sample_path)
        print(extracted_json.model_dump_json(indent=2))
    else:
        print("Please add a sample PDF to data/messy_inputs/ to test extraction.")