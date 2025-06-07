import os
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import google.generativeai as genai

# === Gemini LLM Wrapper for LangChain with Pydantic fields ===
class GeminiLLM(LLM, BaseModel):
    model_name: str = Field(default="gemini-1.5-flash")
    api_key: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("API key must be provided")
        genai.configure(api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

# === Extract all text from PDFs in folder ===
def extract_text_from_pdfs(folder_path):
    all_text = ""
    for pdf_file in Path(folder_path).glob("*.pdf"):
        print(f"Reading: {pdf_file.name}")
        doc = fitz.open(pdf_file)
        for page in doc:
            all_text += page.get_text()
        all_text += "\n\n"
    return all_text.strip()

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables!")

    pdf_folder = r"/path/Receipt_Analysis_Agentic_AI_Project\receipt_pdfs"
    
    # Extract text from PDFs
    all_text = extract_text_from_pdfs(pdf_folder)
    print(f"\n✅ Extracted {len(all_text)} characters of text.\n")

    # Setup LangChain Gemini LLM
    gemini_llm = GeminiLLM(api_key=api_key)

    # Define prompt template
    prompt_template = """
Extract relevant data from these receipts:

{receipt_text}

Organize and consolidate the data by company and date, then generate a final list of clubbed data for analysis and reporting purposes. 
Include company names, dates, items, amounts, and totals. The output should be structured and easy to analyze.
"""
    prompt = PromptTemplate(
        input_variables=["receipt_text"],
        template=prompt_template
    )

    # Setup LLMChain
    chain = LLMChain(llm=gemini_llm, prompt=prompt)

    # Run the chain
    response = chain.run(receipt_text=all_text)

    print("\n=== Gemini's Response ===\n")
    print(response)

if __name__ == "__main__":
    main()
