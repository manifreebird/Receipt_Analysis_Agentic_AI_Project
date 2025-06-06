import os
import json
from receipt_processor import ReceiptProcessingCrew  # Import your main class

def create_sample_data():
    """Create sample receipt data for testing (simulates what would be in PDFs)"""
    
    # Create sample directory
    os.makedirs("./sample_receipts", exist_ok=True)
    
    # Create sample extracted data (for testing aggregation without actual PDFs)
    sample_extracted_data = [
        {"company_name": "Starbucks", "total_amount": "15.50"},
        {"company_name": "McDonald's", "total_amount": "8.75"},
        {"company_name": "Starbucks", "total_amount": "12.25"},
        {"company_name": "Walmart", "total_amount": "45.60"},
        {"company_name": "McDonald's", "total_amount": "6.50"},
        {"company_name": "Target", "total_amount": "23.80"},
        {"company_name": "Domino's Pizza", "total_amount": "18.90"},
        {"company_name": "Starbucks", "total_amount": "9.75"}
    ]
    
    # Save sample data
    with open("extracted_receipts.json", "w") as f:
        json.dump(sample_extracted_data, f, indent=2)
    
    print("✅ Sample data created!")
    print("📄 Sample extracted data:")
    print(json.dumps(sample_extracted_data, indent=2))

def test_aggregation_only():
    """Test only the aggregation agent with sample data using Gemini"""
    
    print("🧪 Testing Gemini-powered Aggregation Agent...")
    print("-" * 50)
    
    # Create sample data
    create_sample_data()
    
    # Get API key
    api_key = input("Enter your Google API key (or press Enter if set in environment): ").strip()
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")
    
    # Initialize processor with Gemini
    processor = ReceiptProcessingCrew(google_api_key=api_key)
    
    # Create and run only aggregation task
    aggregation_task = processor.create_aggregation_task()
    
    from crewai import Crew, Process
    
    crew = Crew(
        agents=[processor.aggregation_agent],
        tasks=[aggregation_task],
        process=Process.sequential,
        verbose=True
    )
    
    print("🤖 Starting Gemini 1.5 Aggregation Agent...")
    result = crew.kickoff()
    
    print("\n" + "="*60)
    print("🎉 GEMINI AGGREGATION TEST RESULTS:")
    print("="*60)
    print(result)
    
    # Show expected vs actual
    expected_totals = {
        "Starbucks": 37.50,  # 15.50 + 12.25 + 9.75
        "McDonald's": 15.25,  # 8.75 + 6.50
        "Walmart": 45.60,
        "Target": 23.80,
        "Domino's Pizza": 18.90
    }
    
    print("\n📊 Expected Results:")
    for company, total in expected_totals.items():
        print(f"  {company}: ${total}")

def test_full_system():
    """Test the complete system with Gemini"""
    
    print("🧪 Testing Full Gemini-powered System...")
    print("-" * 50)
    
    # Make sure you have PDF files in ./receipt_pdfs directory
    pdf_directory = "./receipt_pdfs"
    
    if not os.path.exists(pdf_directory):
        print(f"📁 Creating directory: {pdf_directory}")
        os.makedirs(pdf_directory, exist_ok=True)
        print("⚠️  Please add some PDF receipt files to the ./receipt_pdfs directory")
        print("   You can download sample receipts or create test PDFs")
        return
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        print("⚠️  No PDF files found in ./receipt_pdfs directory")
        print("Please add some PDF receipt files to test the full system")
        print("\nFor testing, you can:")
        print("1. Download sample receipts from the internet")
        print("2. Scan your own receipts")
        print("3. Create simple text-based PDFs with receipt-like content")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files: {pdf_files}")
    
    # Get API key
    api_key = input("Enter your Google API key (or press Enter if set in environment): ").strip()
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")
    
    # Initialize and run full system
    processor = ReceiptProcessingCrew(google_api_key=api_key)
    
    print("🤖 Starting Full Gemini 1.5 Processing...")
    result = processor.process_receipts(pdf_directory)
    
    print("\n" + "="*60)
    print("🎉 FULL GEMINI SYSTEM TEST RESULTS:")
    print("="*60)
    print(result)

def create_test_pdf():
    """Create a simple test PDF with receipt-like content"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create sample receipt PDF
        filename = "./receipt_pdfs/sample_receipt.pdf"
        os.makedirs("./receipt_pdfs", exist_ok=True)
        
        c = canvas.Canvas(filename, pagesize=letter)
        
        # Add receipt content
        c.drawString(100, 750, "STARBUCKS COFFEE")
        c.drawString(100, 730, "123 Main Street")
        c.drawString(100, 710, "City, State 12345")
        c.drawString(100, 680, "Date: 2024-01-15")
        c.drawString(100, 660, "Time: 10:30 AM")
        c.drawString(100, 630, "--------------------------------")
        c.drawString(100, 610, "Grande Latte          $5.25")
        c.drawString(100, 590, "Blueberry Muffin      $3.50")
        c.drawString(100, 570, "--------------------------------")
        c.drawString(100, 550, "Subtotal:             $8.75")
        c.drawString(100, 530, "Tax:                  $0.70")
        c.drawString(100, 510, "TOTAL:                $9.45")
        c.drawString(100, 480, "--------------------------------")
        c.drawString(100, 460, "Thank you for your business!")
        
        c.save()
        print(f"✅ Created test PDF: {filename}")
        return True
        
    except ImportError:
        print("❌ ReportLab not installed. Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ Error creating test PDF: {e}")
        return False

def main():
    print("🤖 CrewAI Receipt Processing with Google Gemini 1.5")
    print("=" * 60)
    print("Choose test mode:")
    print("1. Test aggregation only (with sample data)")
    print("2. Test full system (requires PDF files)")
    print("3. Create test PDF and run full system")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            test_aggregation_only()
            break
        elif choice == "2":
            test_full_system()
            break
        elif choice == "3":
            if create_test_pdf():
                print("\n🚀 Now running full system test...")
                test_full_system()
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()