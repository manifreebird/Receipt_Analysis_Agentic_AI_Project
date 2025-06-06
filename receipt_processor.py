import os
import json
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import PyPDF2
import pdfplumber
from typing import List, Dict, Any

# Set your Google AI API key
os.environ["GOOGLE_API_KEY"] = "your-google-api-key-here"

class ReceiptProcessingTools:
    @staticmethod
    @tool
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF file using pdfplumber for better OCR simulation"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    @staticmethod
    @tool
    def save_json_data(data: str, filename: str = "extracted_receipts.json") -> str:
        """Save extracted data to a JSON file"""
        try:
            # Parse the data if it's a string representation
            if isinstance(data, str):
                # Try to parse as JSON if it looks like JSON
                try:
                    parsed_data = json.loads(data)
                except:
                    # If not valid JSON, treat as raw data
                    parsed_data = data
            else:
                parsed_data = data
            
            with open(filename, 'w') as f:
                json.dump(parsed_data, f, indent=2)
            return f"Data saved successfully to {filename}"
        except Exception as e:
            return f"Error saving data: {str(e)}"
    
    @staticmethod
    @tool
    def read_json_file(filename: str) -> str:
        """Read data from a JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @staticmethod
    @tool
    def list_pdf_files(directory: str) -> List[str]:
        """List all PDF files in a directory"""
        try:
            pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
            return pdf_files
        except Exception as e:
            return [f"Error listing files: {str(e)}"]

class ReceiptProcessingCrew:
    def __init__(self, google_api_key: str = None):
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize Gemini 1.5 LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # You can also use "gemini-1.5-pro" for more complex tasks
            temperature=0.1,
            convert_system_message_to_human=True  # Important for Gemini compatibility
        )
        
        # Initialize tools
        self.tools = ReceiptProcessingTools()
        
        # Create agents
        self.extraction_agent = self._create_extraction_agent()
        self.aggregation_agent = self._create_aggregation_agent()
        
    def _create_extraction_agent(self):
        """Create the PDF extraction agent"""
        return Agent(
            role='Receipt Data Extractor',
            goal='Extract company name and total amount from receipt PDFs with high accuracy',
            backstory="""You are an expert at reading and parsing receipt documents. 
            You have years of experience in OCR and data extraction from various receipt formats.
            You always return data in the exact JSON format requested: 
            {"company_name": "name", "total_amount": "amount"}
            
            You are powered by Google's Gemini 1.5 model, which gives you excellent 
            text understanding and pattern recognition capabilities.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[
                self.tools.extract_text_from_pdf,
                self.tools.save_json_data,
                self.tools.list_pdf_files
            ]
        )
    
    def _create_aggregation_agent(self):
        """Create the data aggregation agent"""
        return Agent(
            role='Data Aggregation Specialist',
            goal='Aggregate receipt data by company name and calculate total amounts',
            backstory="""You are a data analysis expert who specializes in aggregating and 
            summarizing financial data. You excel at grouping data by categories and 
            performing accurate calculations. You always return clean, organized results.
            
            You are powered by Google's Gemini 1.5 model, giving you strong analytical 
            and mathematical reasoning capabilities.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[
                self.tools.read_json_file,
                self.tools.save_json_data
            ]
        )
    
    def create_extraction_task(self, pdf_directory: str):
        """Create task for PDF extraction"""
        return Task(
            description=f"""
            Extract receipt data from all PDF files in the directory: {pdf_directory}
            
            Step-by-step process:
            1. First, list all PDF files in the directory using the list_pdf_files tool
            2. For each PDF file found:
               - Use extract_text_from_pdf tool to get the text content
               - Analyze the text to identify the company/store name
               - Find the total amount (final bill amount, usually at the bottom)
               - Create JSON object: {{"company_name": "name", "total_amount": "number"}}
            3. Collect all extractions into a single list
            4. Save the complete list using save_json_data tool as 'extracted_receipts.json'
            
            Important rules:
            - Extract only the main company/store name (not addresses or other details)
            - For total_amount, extract only the number (remove currency symbols like $ or ₹)
            - If you can't find a field, use empty string ""
            - Look for keywords like "Total", "Amount", "Bill Total", "Grand Total"
            - Be careful to get the final total, not subtotals or tax amounts
            
            Return the final list of all extracted data in JSON format.
            """,
            agent=self.extraction_agent,
            expected_output="A JSON list containing extracted data from all receipts with company_name and total_amount fields"
        )
    
    def create_aggregation_task(self):
        """Create task for data aggregation"""
        return Task(
            description="""
            Read the extracted receipt data and create aggregated summary by company.
            
            Step-by-step process:
            1. Use read_json_file tool to read 'extracted_receipts.json'
            2. Parse the JSON data to get list of receipts
            3. Group all receipts by company_name (exact matches)
            4. For each company, sum all the total_amount values
            5. Create final aggregated object in format:
               {{
                 "Company Name 1": total_sum_as_number,
                 "Company Name 2": total_sum_as_number,
                 ...
               }}
            6. Save result using save_json_data tool as 'aggregated_receipts.json'
            
            Important rules:
            - Convert all total_amount strings to numbers before summing
            - Group by exact company name matches (case-sensitive)
            - Handle empty or invalid amounts by treating them as 0
            - Return only numeric values in the final result
            - If a company appears multiple times, sum all their amounts
            
            Return the final aggregated results showing total spending per company.
            """,
            agent=self.aggregation_agent,
            expected_output="A JSON object with company names as keys and total aggregated amounts as numeric values"
        )
    
    def process_receipts(self, pdf_directory: str):
        """Main method to process receipts using CrewAI with Gemini"""
        
        # Create tasks
        extraction_task = self.create_extraction_task(pdf_directory)
        aggregation_task = self.create_aggregation_task()
        
        # Create crew
        crew = Crew(
            agents=[self.extraction_agent, self.aggregation_agent],
            tasks=[extraction_task, aggregation_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute the crew
        print("🚀 Starting Receipt Processing with CrewAI and Gemini 1.5...")
        print("🤖 Agent 1: PDF Receipt Extractor (powered by Gemini)")
        print("🤖 Agent 2: Data Aggregator (powered by Gemini)")
        print("-" * 60)
        
        result = crew.kickoff()
        
        return result

# Usage Example
def main():
    # Initialize the system with Gemini
    processor = ReceiptProcessingCrew(google_api_key="your-google-api-key-here")
    
    # Specify the directory containing PDF receipts
    pdf_directory = "./receipt_pdfs"  # Change this to your PDF directory path
    
    # Create the directory if it doesn't exist (for testing)
    os.makedirs(pdf_directory, exist_ok=True)
    
    # Process receipts
    try:
        result = processor.process_receipts(pdf_directory)
        print("\n" + "="*60)
        print("🎉 FINAL RESULTS FROM GEMINI AGENTS:")
        print("="*60)
        print(result)
        
        # Display saved results
        try:
            with open('extracted_receipts.json', 'r') as f:
                extracted_data = json.load(f)
            print("\n📄 Extracted Data (Agent 1 Output):")
            print(json.dumps(extracted_data, indent=2))
        except FileNotFoundError:
            print("❌ extracted_receipts.json not found")
        except Exception as e:
            print(f"❌ Error reading extracted data: {e}")
        
        try:
            with open('aggregated_receipts.json', 'r') as f:
                aggregated_data = json.load(f)
            print("\n📊 Aggregated Data (Agent 2 Output):")
            print(json.dumps(aggregated_data, indent=2))
            
            # Show summary
            print("\n💰 SPENDING SUMMARY:")
            print("-" * 30)
            total_spending = 0
            for company, amount in aggregated_data.items():
                print(f"{company}: ${amount}")
                total_spending += float(amount)
            print("-" * 30)
            print(f"Total Spending: ${total_spending:.2f}")
            
        except FileNotFoundError:
            print("❌ aggregated_receipts.json not found")
        except Exception as e:
            print(f"❌ Error reading aggregated data: {e}")
            
    except Exception as e:
        print(f"❌ Error processing receipts: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your Google API key is valid")
        print("2. Check if PDF files exist in the specified directory")
        print("3. Ensure all required packages are installed")

if __name__ == "__main__":
    main()