
# enhanced_run_all_robust.py - Robust version with better LLM error handling
import json
import traceback
import os
from pathlib import Path
import pandas as pd
import time

# Configuration
repo_root = Path(__file__).resolve().parent
mock_dir = repo_root / "data" / "mock_files"
out_dir = repo_root / "outputs"
enhanced_out_dir = repo_root / "outputs_enhanced_robust"

out_dir.mkdir(parents=True, exist_ok=True)
enhanced_out_dir.mkdir(parents=True, exist_ok=True)

# Import existing extractors with fallbacks
try:
    from src.extract_pdf import extract_from_pdf
except Exception:
    def extract_from_pdf(path):
        return {'error': 'extract_from_pdf not available'}

try:
    from src.extract_ocr import extract_from_image
except Exception:
    def extract_from_image(path):
        return {'error': 'extract_from_image not available'}

try:
    from src.extract_excel import extract_excel as _extract_excel
    def extract_from_excel(path):
        return _extract_excel(path)
except Exception:
    def extract_from_excel(path):
        try:
            import pandas as pd
            df = pd.read_excel(path, sheet_name=0)
            items = []
            try:
                items_df = pd.read_excel(path, sheet_name='Items')
                items = items_df.to_dict(orient='records')
            except Exception:
                items = df.to_dict(orient='records') if not df.empty else []
            return {
                'order_id': None, 'client_name': None, 'order_date': None,
                'delivery_date': None, 'items': items,
                'order_total': None, 'currency': None, 'special_instructions': None,
                'confidence_score': 0.7
            }
        except Exception as e:
            return {'error': f'excel extraction failed: {e}'}

def extract_from_csv(path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return {
            'order_id': None, 'client_name': None, 'order_date': None,
            'delivery_date': None, 'items': df.to_dict(orient='records'),
            'order_total': None, 'currency': None, 'special_instructions': None,
            'confidence_score': 0.6
        }
    except Exception as e:
        return {'error': str(e)}

# Standard handlers
handlers = {
    '.pdf': lambda p: extract_from_pdf(str(p)),
    '.png': lambda p: extract_from_image(str(p)),
    '.jpg': lambda p: extract_from_image(str(p)),
    '.jpeg': lambda p: extract_from_image(str(p)),
    '.tiff': lambda p: extract_from_image(str(p)),
    '.bmp': lambda p: extract_from_image(str(p)),
    '.xlsx': lambda p: extract_from_excel(str(p)),
    '.xls': lambda p: extract_from_excel(str(p)),
    '.csv': lambda p: extract_from_csv(str(p)),
}

# Robust LLM Enhancement Class
class SimpleLLMEnhancer:
    def __init__(self, api_key):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4"
    
    def safe_llm_call(self, prompt, max_retries=2):
        """Make LLM call with retries and error handling"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500,
                    timeout=20
                )
                
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                    
            except Exception as e:
                print(f"      LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        return None
    
    def safe_json_parse(self, response_text, fallback=None):
        """Safely parse JSON response with fallback"""
        if not response_text:
            return fallback or {}
        
        # Try direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Manual extraction as last resort
        result = fallback or {}
        
        # Extract key patterns
        patterns = {
            'order_id': r'order[_\s]*id[:\s]*([A-Z0-9\-_]+)',
            'client_name': r'client[_\s]*name[:\s]*([^,\n]+)',
            'confidence_score': r'confidence[:\s]*(\d+\.?\d*)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key == 'confidence_score':
                    try:
                        result[key] = float(value)
                    except ValueError:
                        pass
                else:
                    result[key] = value
        
        return result
    
    def enhance_metadata_simple(self, raw_content, current_result):
        """Simple metadata enhancement with robust error handling"""
        
        # Limit content size
        content_sample = raw_content[:1500] if len(raw_content) > 1500 else raw_content
        
        prompt = f"""
        Extract purchase order information from this document. 
        Respond with ONLY a JSON object, no other text:
        
        Document: {content_sample}
        
        {{
            "order_id": "found_order_id_or_null",
            "client_name": "found_client_name_or_null",
            "order_date": "YYYY-MM-DD_or_null",
            "delivery_date": "YYYY-MM-DD_or_null", 
            "order_total": 123.45,
            "currency": "USD",
            "confidence_score": 0.8
        }}
        """
        
        response = self.safe_llm_call(prompt)
        enhanced = self.safe_json_parse(response, current_result.copy())
        
        return enhanced
    
    def enhance_excel_simple(self, raw_content, current_result):
        """Simple Excel enhancement"""
        
        items_sample = current_result.get('items', [])[:3]  # First 3 items
        
        prompt = f"""
        Process this Excel/CSV purchase order data.
        Respond with ONLY a JSON object, no other text:
        
        Current items: {json.dumps(items_sample, default=str)}
        
        {{
            "order_id": "extracted_from_items_or_existing",
            "client_name": "inferred_client_name",
            "order_total": 123.45,
            "confidence_score": 0.85
        }}
        """
        
        response = self.safe_llm_call(prompt)
        enhanced = self.safe_json_parse(response, current_result.copy())
        
        return enhanced
    
    def enhance_scanned_simple(self, raw_content, current_result):
        """Simple scanned document enhancement"""
        
        content_sample = raw_content[:1500] if len(raw_content) > 1500 else raw_content
        
        prompt = f"""
        Extract data from this scanned purchase order.
        Respond with ONLY a JSON object, no other text:
        
        OCR Text: {content_sample}
        
        {{
            "order_id": "corrected_order_id",
            "client_name": "extracted_client",
            "items": [
                {{
                    "product_code": "code",
                    "description": "description",
                    "quantity": 1,
                    "unit_price": 10.0,
                    "total_price": 10.0
                }}
            ],
            "confidence_score": 0.75
        }}
        """
        
        response = self.safe_llm_call(prompt)
        enhanced = self.safe_json_parse(response, current_result.copy())
        
        return enhanced

# Initialize LLM enhancer
llm_enhancer = None
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    try:
        llm_enhancer = SimpleLLMEnhancer(openai_api_key)
        print("✓ Robust LLM enhancer initialized")
    except Exception as e:
        print(f"⚠ LLM enhancer failed to initialize: {e}")
else:
    print("⚠ OPENAI_API_KEY not set")

def get_raw_content(file_path):
    """Extract raw content for LLM processing"""
    try:
        path = Path(file_path)
        
        if path.suffix.lower() in ['.xlsx', '.xls']:
            import pandas as pd
            try:
                # Try to read multiple sheets
                excel_file = pd.ExcelFile(str(path))
                content_parts = []
                for sheet_name in excel_file.sheet_names[:3]:  # First 3 sheets
                    df = pd.read_excel(str(path), sheet_name=sheet_name)
                    content_parts.append(f"Sheet {sheet_name}: {df.head(10).to_string()}")
                return "\n\n".join(content_parts)
            except Exception:
                return f"Excel file: {path.name}"
        
        elif path.suffix.lower() == '.csv':
            import pandas as pd
            try:
                df = pd.read_csv(str(path))
                return df.head(10).to_string()
            except Exception:
                return f"CSV file: {path.name}"
        
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(str(path))
                return pytesseract.image_to_string(img)
            except Exception:
                return f"Image file: {path.name}"
        
        elif path.suffix.lower() in ['.txt']:
            return path.read_text()[:2000]  # Limit size
        
        return f"File: {path.name}"
        
    except Exception as e:
        print(f"      Error extracting content from {file_path}: {e}")
        return ""

def should_enhance_robust(file_path, result):
    """Robust enhancement decision logic"""
    
    # Skip if error occurred
    if result.get('error'):
        return False
    
    # Always enhance if confidence < 0.8
    if result.get('confidence_score', 0) < 0.8:
        return True
    
    # Always enhance Excel/CSV files
    if file_path and Path(file_path).suffix.lower() in ['.xlsx', '.xls', '.csv']:
        return True
    
    # Always enhance image files
    if file_path and Path(file_path).suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return True
    
    # Enhance if missing critical data
    if not result.get('order_id') or not result.get('client_name'):
        return True
    
    # Enhance if no items
    if not result.get('items'):
        return True
    
    return False

def robust_enhance_extraction(enhancer, raw_content, current_result, file_path):
    """Robust enhancement with comprehensive error handling"""
    
    result = current_result.copy()
    strategies_applied = []
    
    print(f"    🛠️ Applying robust enhancement...")
    
    # Strategy 1: Metadata enhancement (always try)
    try:
        print(f"      📝 Enhancing metadata...")
        metadata_enhanced = enhancer.enhance_metadata_simple(raw_content, result)
        if metadata_enhanced:
            # Only update fields that were actually improved
            for key, value in metadata_enhanced.items():
                if value and (not result.get(key) or key == 'confidence_score'):
                    result[key] = value
            strategies_applied.append("metadata_enhanced")
            print(f"      ✓ Metadata enhancement applied")
    except Exception as e:
        print(f"      ⚠ Metadata enhancement failed: {e}")
        strategies_applied.append(f"metadata_failed: {str(e)[:30]}")
    
    # Strategy 2: File-type specific enhancement
    if file_path:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.xlsx', '.xls', '.csv']:
            try:
                print(f"      📊 Enhancing Excel/CSV...")
                excel_enhanced = enhancer.enhance_excel_simple(raw_content, result)
                if excel_enhanced:
                    for key, value in excel_enhanced.items():
                        if value and (not result.get(key) or key == 'confidence_score'):
                            result[key] = value
                    strategies_applied.append("excel_enhanced")
                    print(f"      ✓ Excel/CSV enhancement applied")
            except Exception as e:
                print(f"      ⚠ Excel/CSV enhancement failed: {e}")
                strategies_applied.append(f"excel_failed: {str(e)[:30]}")
        
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            try:
                print(f"      🖼️ Enhancing scanned document...")
                scanned_enhanced = enhancer.enhance_scanned_simple(raw_content, result)
                if scanned_enhanced:
                    for key, value in scanned_enhanced.items():
                        if value and (not result.get(key) or key in ['confidence_score', 'items']):
                            result[key] = value
                    strategies_applied.append("scanned_enhanced")
                    print(f"      ✓ Scanned document enhancement applied")
            except Exception as e:
                print(f"      ⚠ Scanned document enhancement failed: {e}")
                strategies_applied.append(f"scanned_failed: {str(e)[:30]}")
    
    # Ensure confidence score is reasonable
    if not result.get('confidence_score') or result['confidence_score'] < current_result.get('confidence_score', 0):
        result['confidence_score'] = max(current_result.get('confidence_score', 0.5), 0.6)
    
    # Add enhancement info
    result['enhancement_info'] = {
        'enhanced_by_llm': len([s for s in strategies_applied if 'enhanced' in s]) > 0,
        'enhancement_strategies': strategies_applied,
        'enhancement_timestamp': pd.Timestamp.now().isoformat(),
        'robust_mode': True,
        'original_confidence': current_result.get('confidence_score', 0),
        'final_confidence': result.get('confidence_score', 0)
    }
    
    return result

def process_file_robust(file_path):
    """Process a single file with robust enhancement"""
    
    print(f"\nProcessing: {file_path.name}")
    
    # Standard extraction
    ext = file_path.suffix.lower()
    handler = handlers.get(ext)
    
    if handler is None:
        result = {'error': f'unsupported file type: {ext}', 'confidence_score': 0.0}
    else:
        try:
            result = handler(file_path)
        except Exception as e:
            result = {'error': str(e), 'confidence_score': 0.0}
    
    print(f"  Original: {result.get('confidence_score', 0):.2f} confidence, {len(result.get('items', []))} items")
    
    # Robust enhancement
    enhanced_result = result.copy()
    
    if llm_enhancer and should_enhance_robust(file_path, result):
        try:
            raw_content = get_raw_content(file_path)
            if raw_content.strip():
                enhanced_result = robust_enhance_extraction(
                    llm_enhancer, raw_content, result, str(file_path)
                )
                
                orig_conf = result.get('confidence_score', 0)
                enh_conf = enhanced_result.get('confidence_score', 0)
                improvement = enh_conf - orig_conf
                
                strategies = enhanced_result.get('enhancement_info', {}).get('enhancement_strategies', [])
                success_count = len([s for s in strategies if 'enhanced' in s])
                
                print(f"  Enhanced: {enh_conf:.2f} confidence (Δ{improvement:+.2f}), {success_count} strategies applied")
            else:
                print(f"  No content available for enhancement")
        except Exception as e:
            print(f"  ❌ Enhancement completely failed: {e}")
            enhanced_result['enhancement_error'] = str(e)
    else:
        print(f"  Skipping enhancement (not needed or LLM unavailable)")
    
    # Add processing metadata
    enhanced_result['processing_metadata'] = {
        'file_name': file_path.name,
        'file_size_bytes': file_path.stat().st_size,
        'processing_timestamp': pd.Timestamp.now().isoformat(),
        'llm_enhanced': enhanced_result.get('enhancement_info', {}).get('enhanced_by_llm', False),
        'final_confidence': enhanced_result.get('confidence_score', 0.0),
        'needs_review': enhanced_result.get('confidence_score', 0.0) < 0.7,
        'robust_mode': True
    }
    
    return enhanced_result

def main():
    """Main robust processing pipeline"""
    print("🛠️ Starting ROBUST Enhanced Document Processing Pipeline")
    print("=" * 70)
    
    if not mock_dir.exists():
        print(f"❌ Mock files directory not found: {mock_dir}")
        return
    
    # Find files to process
    files_to_process = []
    for file_path in sorted(mock_dir.iterdir()):
        if file_path.name.startswith('.') or file_path.name.lower() == '.gitkeep':
            continue
        if not file_path.is_file():
            continue
        
        ext = file_path.suffix.lower()
        if ext in handlers.keys():
            files_to_process.append(file_path)
        else:
            print(f"⚠ Skipping unsupported format: {file_path.name}")
    
    if not files_to_process:
        print("❌ No files found to process")
        return
    
    print(f"📁 Found {len(files_to_process)} files to process")
    print(f"🤖 LLM Enhancement: {'✓ ROBUST MODE' if llm_enhancer else '✗ Disabled'}")
    
    # Process files
    processed_files = []
    successful_enhancements = 0
    
    for file_path in files_to_process:
        try:
            enhanced_result = process_file_robust(file_path)
            
            # Save result
            enhanced_name = file_path.stem + '_enhanced_robust.json'
            enhanced_path = enhanced_out_dir / enhanced_name
            
            with open(enhanced_path, 'w', encoding='utf-8') as fo:
                json.dump(enhanced_result, fo, indent=2, default=str)
            
            print(f"  💾 Saved: {enhanced_path.name}")
            
            if enhanced_result.get('enhancement_info', {}).get('enhanced_by_llm'):
                successful_enhancements += 1
            
            processed_files.append(enhanced_result)
            
        except Exception as e:
            print(f"❌ Failed processing {file_path.name}: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ROBUST Processing Summary")
    print("=" * 70)
    print(f"Total files processed: {len(processed_files)}")
    print(f"Successfully enhanced: {successful_enhancements}")
    if processed_files:
        print(f"Enhancement rate: {successful_enhancements/len(processed_files)*100:.1f}%")
        
        avg_confidence = sum(f.get('confidence_score', 0) for f in processed_files) / len(processed_files)
        print(f"Average confidence: {avg_confidence:.2f}")
        
        high_confidence = sum(1 for f in processed_files if f.get('confidence_score', 0) >= 0.8)
        print(f"High confidence files: {high_confidence}/{len(processed_files)}")
    
    print(f"\n💾 Results saved to: {enhanced_out_dir}/")
    print("\n✅ ROBUST processing complete!")

if __name__ == '__main__':
    main()
