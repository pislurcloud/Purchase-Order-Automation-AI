# config.py - Configuration management for different deployment scenarios
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class ProcessingConfig:
    """Configuration for document processing pipeline"""
    
    # LLM Settings
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_timeout: int = 30
    enable_llm_enhancement: bool = True
    
    # Processing Settings
    max_file_size_mb: int = 50
    ocr_dpi: int = 300
    pdf_page_limit: int = 10
    confidence_threshold: float = 0.7
    
    # Paths
    mock_files_dir: str = "data/mock_files"
    output_dir: str = "outputs"
    enhanced_output_dir: str = "outputs_enhanced"
    labels_dir: str = "labels"
    
    # Performance Settings
    enable_caching: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    
    # Validation Settings
    strict_schema_validation: bool = False
    require_order_total: bool = False
    require_items: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'ProcessingConfig':
        """Create configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            llm_model=os.getenv('LLM_MODEL', 'gpt-4'),
            llm_temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            enable_llm_enhancement=os.getenv('ENABLE_LLM', 'true').lower() == 'true',
            
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '50')),
            ocr_dpi=int(os.getenv('OCR_DPI', '300')),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')),
            
            mock_files_dir=os.getenv('MOCK_FILES_DIR', 'data/mock_files'),
            output_dir=os.getenv('OUTPUT_DIR', 'outputs'),
            enhanced_output_dir=os.getenv('ENHANCED_OUTPUT_DIR', 'outputs_enhanced'),
            
            parallel_processing=os.getenv('PARALLEL_PROCESSING', 'false').lower() == 'true',
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            
            strict_schema_validation=os.getenv('STRICT_VALIDATION', 'false').lower() == 'true',
            enable_metrics=os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
        )
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        issues = []
        
        if self.enable_llm_enhancement and not self.openai_api_key:
            issues.append("LLM enhancement enabled but OPENAI_API_KEY not provided")
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            issues.append("confidence_threshold must be between 0 and 1")
            
        if self.max_file_size_mb < 1:
            issues.append("max_file_size_mb must be >= 1")
            
        if self.max_workers < 1:
            issues.append("max_workers must be >= 1")
        
        if issues:
            print("Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True

# Mock data generator for testing
def create_mock_files():
    """Create mock files matching the assignment scenarios"""
    
    mock_dir = Path("data/mock_files")
    mock_dir.mkdir(parents=True, exist_ok=True)
    
    # Client A - Clean PDF content (we'll create a text version)
    client_a_content = """PURCHASE ORDER #PO-2024-1247
TechCorp Solutions
Date: March 15, 2024
Delivery Required: March 22, 2024

Item Code | Description | Qty | Unit Price | Total
TC-001 | Widget Pro | 50 | $25.00 | $1,250.00
TC-002 | Gadget Max | 25 | $45.00 | $1,125.00

TOTAL: $2,375.00
Special Notes: Rush delivery required"""
    
    with open(mock_dir / "client_a_techcorp.txt", "w") as f:
        f.write(client_a_content)
    
    # Client B - Excel structure (create actual Excel file)
    try:
        import pandas as pd
        
        # Sheet 1 - Order Info
        order_info = pd.DataFrame([{
            'Order#': 'GMI-2024-0892',
            'Client_Name': 'Global Manufacturing Inc',
            'Order_Created': '2024-03-16',
            'Needed_By': '2024-03-25'
        }])
        
        # Sheet 2 - Line Items
        line_items = pd.DataFrame([
            {'SKU': 'GMI-PUMP-X200', 'Item_Desc': 'Industrial Pump Model X200', 'Order_Qty': 3, 'Price_Each': 850.00},
            {'SKU': 'GMI-FILTER-SET', 'Item_Desc': 'Filter Cartridge Set', 'Order_Qty': 12, 'Price_Each': 45.00}
        ])
        
        # Sheet 3 - Notes
        notes = pd.DataFrame([{
            'Special_Requirements': 'Phoenix warehouse delivery',
            'Delivery_Instructions': 'Must be completed before 3 PM'
        }])
        
        with pd.ExcelWriter(mock_dir / "client_b_global_manufacturing.xlsx") as writer:
            order_info.to_excel(writer, sheet_name='Order', index=False)
            line_items.to_excel(writer, sheet_name='Items', index=False)
            notes.to_excel(writer, sheet_name='Notes', index=False)
            
    except ImportError:
        print("⚠ pandas not available, skipping Excel mock file creation")
    
    # Client C - Word document content (as text file)
    client_c_content = """Order Request - Regional Distributors
Order Number: RD-240815-A

We need the following items by August 22, 2024:

Product Name: Industrial Pump Model X200
Quantity Needed: 3 units
Expected Price: $850 per unit

Product Name: Filter Cartridge Set
Quantity Needed: 12 sets
Expected Price: $45 per set

Please note: This is for our Phoenix warehouse.
Delivery must be completed before 3 PM."""
    
    with open(mock_dir / "client_c_regional_distributors.txt", "w") as f:
        f.write(client_c_content)
    
    # Client D - CSV file
    csv_content = """Order_ID,Customer,Item_SKU,Product_Name,Qty_Ordered,Unit_Cost,Order_Date,Ship_Date,Notes
SCP-2024-0445,Supply Chain Partners,SKU-7789,Heavy Duty Clamp,100,12.50,2024-03-20,03/27/2024,Standard shipping
SCP-2024-0445,Supply Chain Partners,SKU-3421,Mounting Bracket,75,8.25,2024-03-20,03/27/2024,"""
    
    with open(mock_dir / "client_d_supply_chain.csv", "w") as f:
        f.write(csv_content)
    
    # Client E - Scanned form content (as text)
    client_e_content = """LOCAL HARDWARE CO - ORDER FORM
Order #: LHC-240318
Date: 3/18/2024
Need by: 3/25/2024

[ ] Standard Delivery [X] Rush Delivery

Item 1: Screws - Phillips Head #8 x 1"
Qty: 500 pcs
Price: $0.15 each

Item 2: Wood Stain - Oak Color  
Qty: 6 gallons
Price: $28.00 per gallon

Special Instructions: Call before delivery
Contact: Mike Johnson (555) 123-4567"""
    
    with open(mock_dir / "client_e_local_hardware.txt", "w") as f:
        f.write(client_e_content)
    
    # Additional test cases for edge cases
    edge_case_content = """URGENT ORDER - MEDICAL SUPPLIES INC
REF: MS-2024-URGENT-001
Date: Today
Required: ASAP

Items needed:
- Medical Masks (N95): 1000 units @ $2.50 each
- Hand Sanitizer: 50 bottles @ $8.99 each  
- Disposable Gloves: 200 boxes @ $15.75 each

Total: $6,094.50 USD

NOTE: This is a rush order for emergency supplies
Contact: Dr. Sarah Johnson, emergency@medical-supplies.com
Payment Terms: Net 15"""
    
    with open(mock_dir / "edge_case_urgent_medical.txt", "w") as f:
        f.write(edge_case_content)
    
    print(f"✅ Mock files created in {mock_dir}/")
    print("Files created:")
    for file in sorted(mock_dir.iterdir()):
        if file.is_file() and not file.name.startswith('.'):
            print(f"  - {file.name}")

# Environment-specific configurations
def get_development_config() -> ProcessingConfig:
    """Configuration for development environment"""
    config = ProcessingConfig.from_env()
    config.log_level = "DEBUG"
    config.enable_metrics = True
    config.strict_schema_validation = False
    return config

def get_production_config() -> ProcessingConfig:
    """Configuration for production environment"""
    config = ProcessingConfig.from_env()
    config.log_level = "INFO"
    config.enable_metrics = True
    config.strict_schema_validation = True
    config.parallel_processing = True
    return config

def get_testing_config() -> ProcessingConfig:
    """Configuration for testing environment"""
    config = ProcessingConfig()
    config.enable_llm_enhancement = False  # Avoid API calls in tests
    config.log_level = "WARNING"
    config.enable_metrics = False
    config.mock_files_dir = "tests/fixtures"
    config.output_dir = "tests/outputs"
    return config

# Configuration factory
def get_config(environment: str = None) -> ProcessingConfig:
    """Get configuration for specified environment"""
    env = environment or os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return get_production_config()
    elif env == 'testing':
        return get_testing_config()
    else:
        return get_development_config()

# Configuration validation script
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'create-mocks':
        create_mock_files()
    else:
        print("Testing configuration system...")
        
        # Test different environments
        for env in ['development', 'production', 'testing']:
            print(f"\n{env.upper()} Configuration:")
            config = get_config(env)
            print(f"  LLM Enhancement: {config.enable_llm_enhancement}")
            print(f"  Log Level: {config.log_level}")
            print(f"  Parallel Processing: {config.parallel_processing}")
            print(f"  Strict Validation: {config.strict_schema_validation}")
            
            valid = config.validate()
            print(f"  Valid: {valid}")
        
        print("\nTo create mock files, run: python config.py create-mocks")