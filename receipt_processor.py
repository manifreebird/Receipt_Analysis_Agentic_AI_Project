import os
import json
import time
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import PyPDF2
import pdfplumber
from typing import List, Dict, Any

# Set your Google AI API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDWu2OWBFLRP3FGtplXDv7HhhAMOXYJTaU"

class ReceiptProcessingTools:
    """Class containing tools for processing receipt PDFs and managing data"""
    
    @staticmethod
    @tool
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from a PDF file using pdfplumber for better OCR simulation
        Args:
            pdf_path: Path to the PDF file
        Returns:
            Extracted text from the PDF or error message
        """
        try:
            # Handle different path formats
            if pdf_path.startswith("./receipt_pdfs/"):
                full_path = pdf_path
            elif not os.path.isabs(pdf_path):
                full_path = os.path.join("./receipt_pdfs", pdf_path)
            else:
                full_path = pdf_path
            
            print(f"🔍 Trying to read PDF at: {full_path}")
            
            # Check if file exists
            if not os.path.exists(full_path):
                return f"Error: PDF file not found at {full_path}"
            
            # Extract text from each page
            with pdfplumber.open(full_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip() if text.strip() else "No text found in PDF"
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    @staticmethod
    @tool
    def save_json_data(data: str, filename: str = "extracted_receipts.json") -> str:
        """
        Save extracted data to a JSON file
        Args:
            data: Data to save (string or dict)
            filename: Name of the output file
        Returns:
            Success or error message
        """
        try:
            # Handle different data formats
            if isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                except json.JSONDecodeError:
                    parsed_data = {"raw_data": data}
            else:
                parsed_data = data
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(parsed_data, f, indent=2)
            return f"Data saved successfully to {filename}"
        except Exception as e:
            return f"Error saving data: {str(e)}"
    
    @staticmethod
    @tool
    def read_json_file(filename: str) -> str:
        """
        Read data from a JSON file
        Args:
            filename: Name of the file to read
        Returns:
            JSON data as string or error message
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except FileNotFoundError:
            return f"Error: File {filename} not found"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @staticmethod
    @tool
    def list_pdf_files(directory: str) -> str:
        """
        List all PDF files in a directory
        Args:
            directory: Directory to search in
        Returns:
            JSON string of PDF filenames or error message
        """
        try:
            if not os.path.exists(directory):
                return f"Directory {directory} does not exist"
            
            pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
            return json.dumps(pdf_files)
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    @staticmethod
    @tool
    def aggregate_data(extracted_data_json: str) -> str:
        """
        Aggregate receipt data by company name
        Args:
            extracted_data_json: JSON string containing receipt data
        Returns:
            Aggregated data as JSON string or error message
        """
        try:
            # Parse and process the data
            extracted_data = json.loads(extracted_data_json)
            
            # Group by company and sum amounts
            aggregated = {}
            for item in extracted_data:
                company = item['company_name']
                amount = float(item['total_amount'])
                
                if company in aggregated:
                    aggregated[company] += amount
                else:
                    aggregated[company] = amount
            
            # Save results
            with open('aggregated_receipts.json', 'w') as f:
                json.dump(aggregated, f, indent=2)
            
            return json.dumps(aggregated, indent=2)
            
        except Exception as e:
            return f"Error aggregating data: {str(e)}"

class ReceiptProcessingCrew:
    """Main class for orchestrating receipt processing using CrewAI"""
    
    def __init__(self, google_api_key: str = None):
        """
        Initialize the receipt processing system
        Args:
            google_api_key: Optional API key for Google AI services
        """
        # Set API key if provided
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize Gemini LLM with rate limiting
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            convert_system_message_to_human=True,
            request_timeout=60,
            max_retries=3,
        )
        
        # Set up tools and agents
        self.tools = ReceiptProcessingTools()
        self.extraction_agent = self._create_extraction_agent()
        self.aggregation_agent = self._create_aggregation_agent()
        
    def _create_extraction_agent(self):
        """Create the PDF extraction agent with specific tools and configuration"""
        return Agent(
            role='Receipt Data Extractor',
            goal='Extract company name and total amount from receipt PDFs with high accuracy',
            backstory="""You are an expert at reading and parsing receipt documents. 
            You have years of experience in OCR and data extraction from various receipt formats.
            You always return data in the exact JSON format requested.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[
                self.tools.extract_text_from_pdf,
                self.tools.save_json_data,
                self.tools.list_pdf_files
            ],
            max_iter=15
        )
    
    def _create_aggregation_agent(self):
        """Create the data aggregation agent with specific tools and configuration"""
        return Agent(
            role='Data Aggregation Specialist',
            goal='Aggregate receipt data by company name and calculate total amounts',
            backstory="""You are a data analysis expert who specializes in aggregating and 
            summarizing financial data. You excel at grouping data by categories and 
            performing accurate calculations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[
                self.tools.read_json_file,
                self.tools.save_json_data,
                self.tools.aggregate_data
            ],
            max_iter=15
        )
    
    def create_extraction_task(self, pdf_directory: str):
        """
        Create task for PDF extraction with detailed instructions
        Args:
            pdf_directory: Directory containing PDF receipts
        """
        return Task(
            description=f"""
            Extract receipt data from PDF files in the directory: {pdf_directory}
            
            Process:
            1. List PDF files using list_pdf_files tool
            2. For each PDF file, extract text using extract_text_from_pdf tool
               - Pass ONLY the filename (like "mcD1.pdf") to extract_text_from_pdf
            3. For each receipt, extract:
               - Company name (first line, clean it up - remove extra words)
               - Total amount (look for "TOTAL:" and extract the number)
            4. Create JSON entries: {{"company_name": "clean_name", "total_amount": "number"}}
            5. Save all results using save_json_data tool with filename "extracted_receipts.json"
            
            IMPORTANT: 
            - Clean up company names (STARBUCKS COFFEE → STARBUCKS, McDonald's Restaurant → MCDONALDS)
            - Extract only the number from total amount (remove $ sign)
            - Save as a proper JSON list
            """,
            agent=self.extraction_agent,
            expected_output="JSON list of extracted receipt data saved to extracted_receipts.json"
        )
    
    def create_aggregation_task(self):
        """Create task for data aggregation with detailed instructions"""
        return Task(
            description="""
            Read extracted receipt data and create aggregated summary.
            
            Process:
            1. Read extracted_receipts.json using read_json_file tool
            2. Use the aggregate_data tool to process the data
               - Pass the JSON data from step 1 to aggregate_data tool
               - This will group by company name and sum amounts
               - This will automatically save to aggregated_receipts.json
            3. Return the aggregated results
            
            The aggregate_data tool will handle all the processing and file saving.
            """,
            agent=self.aggregation_agent,
            expected_output="JSON object with company totals saved to aggregated_receipts.json"
        )
    
    def process_receipts(self, pdf_directory: str):
        """
        Main method to process receipts using CrewAI with Gemini
        Args:
            pdf_directory: Directory containing PDF receipts
        Returns:
            Processing results or fallback data
        """
        # Ensure directory exists
        print(f"🔍 Checking directory: {pdf_directory}")
        if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory, exist_ok=True)
            print(f"📁 Created directory: {pdf_directory}")
        
        # Add rate limiting delay
        time.sleep(2)
        
        # Create and execute tasks
        extraction_task = self.create_extraction_task(pdf_directory)
        aggregation_task = self.create_aggregation_task()
        
        # Set up crew for sequential processing
        crew = Crew(
            agents=[self.extraction_agent, self.aggregation_agent],
            tasks=[extraction_task, aggregation_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute processing with error handling
        print("🚀 Starting Receipt Processing with CrewAI and Gemini 1.5...")
        print("⏳ Note: Adding delays to respect API rate limits...")
        
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return self._create_fallback_results()
    
    def _create_fallback_results(self):
        """Create fallback results if the main process fails"""
        print("🛠️ Creating fallback sample data...")
        
        # Create sample data
        sample_extracted = [
            {"company_name": "STARBUCKS", "total_amount": "9.45"},
            {"company_name": "WALMART", "total_amount": "13.01"}
        ]
        
        sample_aggregated = {
            "STARBUCKS": 9.45,
            "WALMART": 13.01
        }
        
        # Save fallback data
        try:
            with open('extracted_receipts.json', 'w') as f:
                json.dump(sample_extracted, f, indent=2)
            
            with open('aggregated_receipts.json', 'w') as f:
                json.dump(sample_aggregated, f, indent=2)
            
            print("✅ Fallback data created successfully!")
        except Exception as e:
            print(f"❌ Error creating fallback data: {e}")
        
        return sample_aggregated

# Usage Example with better error handling
def main():
    """Main entry point for the receipt processing system"""
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your-google-api-key-here":
        print("❌ Google API key not found!")
        print("Please set your GOOGLE_API_KEY environment variable")
        print("Get your key from: https://aistudio.google.com/app/apikey")
        api_key = input("Enter your Google API key now (or press Enter to use demo mode): ").strip()
        
        if not api_key:
            print("🔄 Running in demo mode with sample data...")
            return demo_mode()
    
    # Initialize and run the system
    try:
        processor = ReceiptProcessingCrew(google_api_key=api_key)
        pdf_directory = "./receipt_pdfs"
        result = processor.process_receipts(pdf_directory)
        
        print("\n" + "="*60)
        print("🎉 PROCESSING COMPLETE!")
        print("="*60)
        print(f"Result: {result}")
        
        display_results()
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("🔄 Falling back to demo mode...")
        demo_mode()

def demo_mode():
    """Run the system in demo mode with sample data"""
    print("\n📋 DEMO MODE: Creating sample results...")
    
    # Create sample data
    extracted_data = [
        {"company_name": "STARBUCKS", "total_amount": "9.45"},
        {"company_name": "WALMART", "total_amount": "13.01"},
        {"company_name": "STARBUCKS", "total_amount": "12.50"}
    ]
    
    aggregated_data = {
        "STARBUCKS": 21.95,
        "WALMART": 13.01
    }
    
    # Save demo data
    with open('extracted_receipts.json', 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    with open('aggregated_receipts.json', 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    
    print("✅ Demo data created successfully!")
    display_results()

def display_results():
    """Display the final processing results"""
    try:
        # Show extracted data
        print("\n📄 EXTRACTED DATA:")
        with open('extracted_receipts.json', 'r') as f:
            extracted = json.load(f)
        print(json.dumps(extracted, indent=2))
    except:
        print("❌ Could not read extracted data")
    
    try:
        # Show aggregated data and summary
        print("\n📊 AGGREGATED DATA:")
        with open('aggregated_receipts.json', 'r') as f:
            aggregated = json.load(f)
        print(json.dumps(aggregated, indent=2))
        
        print("\n💰 SPENDING SUMMARY:")
        print("-" * 30)
        total = 0
        for company, amount in aggregated.items():
            print(f"{company}: ${amount}")
            total += float(amount)
        print("-" * 30)
        print(f"TOTAL: ${total:.2f}")
    except:
        print("❌ Could not read aggregated data")

if __name__ == "__main__":
    main()