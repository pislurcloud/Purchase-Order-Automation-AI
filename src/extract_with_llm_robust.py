# src/extract_with_llm_robust.py - Robust version with better error handling
"""
Robust LLM-enhanced extraction module with comprehensive error handling and response validation
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class RobustLLMEnhancedExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize with OpenAI API configuration"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            self.extraction_cache = {}
            self.max_retries = 3
            self.retry_delay = 1  # seconds
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _make_llm_call(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Optional[str]:
        """Make a robust LLM API call with retries and error handling"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30  # 30 second timeout
                )
                
                content = response.choices[0].message.content
                
                if not content or content.strip() == "":
                    logger.warning(f"Empty response from LLM on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                
                return content.strip()
                
            except Exception as e:
                logger.error(f"LLM API call failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                return None
        
        return None
    
    def _parse_json_response(self, response: str, fallback_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse JSON response with robust error handling"""
        
        if not response:
            logger.warning("Empty response to parse")
            return fallback_data or {}
        
        # Try to extract JSON from response (sometimes LLM adds extra text)
        json_patterns = [
            r'\{.*\}',  # Find JSON object
            r'\[.*\]',  # Find JSON array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Try parsing the entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response[:500]}...")
            
            # Try to extract key-value pairs manually as fallback
            fallback = self._extract_fallback_data(response)
            if fallback:
                return fallback
            
            return fallback_data or {}
    
    def _extract_fallback_data(self, response: str) -> Dict[str, Any]:
        """Extract data from non-JSON response as fallback"""
        
        fallback = {}
        
        # Try to extract key-value patterns
        patterns = {
            'order_id': r'order[_\s]*id[:\s]*([A-Z0-9\-_]+)',
            'client_name': r'client[_\s]*name[:\s]*([^,\n]+)',
            'confidence_score': r'confidence[:\s]*(\d+\.?\d*)',
            'order_total': r'total[:\s]*(\d+\.?\d*)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key == 'confidence_score':
                    try:
                        fallback[key] = float(value)
                    except ValueError:
                        pass
                elif key == 'order_total':
                    try:
                        fallback[key] = float(value)
                    except ValueError:
                        pass
                else:
                    fallback[key] = value
        
        return fallback
    
    def classify_document_structure(self, text_content: str, filename: str) -> Dict[str, Any]:
        """Use LLM to classify document structure with robust error handling"""
        
        prompt = f"""
        Analyze this document and return ONLY a JSON object with no additional text:
        
        Document: {filename}
        Content: {text_content[:1500]}...
        
        Return this exact JSON structure:
        {{
            "document_type": "excel_multisheet|csv_tabular|scanned_document|structured_table|form_based",
            "confidence": 0.8,
            "extraction_hints": {{
                "has_clear_headers": true,
                "table_structure": "multiple",
                "suggested_approach": "excel_csv_enhancement"
            }}
        }}
        """
        
        response = self._make_llm_call(prompt, max_tokens=500)
        result = self._parse_json_response(response, {
            "document_type": "unknown",
            "confidence": 0.5,
            "extraction_hints": {"suggested_approach": "fallback"}
        })
        
        return result
    
    def enhance_metadata_extraction(self, raw_text: str, current_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to validate and improve metadata extraction with robust error handling"""
        
        # Limit input size to prevent token limit issues
        text_sample = raw_text[:2000] if len(raw_text) > 2000 else raw_text
        
        prompt = f"""
        Extract purchase order metadata from this document. Return ONLY valid JSON with no additional text:
        
        Document: {text_sample}
        
        Current data: {json.dumps(current_extraction, default=str)[:500]}
        
        Return this exact JSON structure:
        {{
            "order_id": "extracted_order_id_or_null",
            "client_name": "extracted_client_name_or_null", 
            "order_date": "YYYY-MM-DD_or_null",
            "delivery_date": "YYYY-MM-DD_or_null",
            "order_total": 123.45,
            "currency": "USD",
            "special_instructions": "extracted_instructions_or_null",
            "field_confidences": {{
                "order_id_confidence": 0.9,
                "client_name_confidence": 0.8
            }}
        }}
        """
        
        response = self._make_llm_call(prompt, max_tokens=1000)
        result = self._parse_json_response(response, current_extraction.copy())
        
        # Ensure field_confidences exists
        if 'field_confidences' not in result:
            result['field_confidences'] = {
                f"{k}_confidence": 0.7 for k in ['order_id', 'client_name', 'order_date', 'delivery_date'] 
                if k in result
            }
        
        return result
    
    def enhance_excel_csv_extraction(self, raw_data: Dict[str, Any], raw_content: str) -> Dict[str, Any]:
        """Enhanced processing for Excel/CSV data with robust error handling"""
        
        # Limit content size
        content_sample = raw_content[:3000] if len(raw_content) > 3000 else raw_content
        
        prompt = f"""
        Process this Excel/CSV purchase order data. Return ONLY valid JSON with no additional text:
        
        Data: {json.dumps(raw_data, default=str)[:1000]}
        
        Content: {content_sample}
        
        Extract and return this exact JSON structure:
        {{
            "order_id": "primary_order_from_data",
            "client_name": "inferred_client_name",
            "order_date": "YYYY-MM-DD_if_found",
            "delivery_date": "YYYY-MM-DD_if_found",
            "items": [
                {{
                    "product_code": "standardized_code",
                    "description": "product_description", 
                    "quantity": 10,
                    "unit_price": 25.50,
                    "total_price": 255.00
                }}
            ],
            "order_total": 255.00,
            "currency": "USD",
            "confidence_score": 0.85,
            "processing_notes": "brief_explanation"
        }}
        """
        
        response = self._make_llm_call(prompt, max_tokens=2000)
        result = self._parse_json_response(response, raw_data.copy())
        
        # Add enhancement info
        result['enhancement_info'] = {
            'enhanced_by_llm': True,
            'enhancement_type': 'excel_csv_processing',
            'enhancement_timestamp': datetime.utcnow().isoformat()
        }
        
        return result
    
    def enhance_scanned_document_extraction(self, raw_data: Dict[str, Any], raw_content: str) -> Dict[str, Any]:
        """Enhanced processing for scanned documents with robust error handling"""
        
        # Limit content size
        content_sample = raw_content[:3000] if len(raw_content) > 3000 else raw_content
        
        prompt = f"""
        Extract data from this scanned purchase order. Return ONLY valid JSON with no additional text:
        
        Current: {json.dumps(raw_data, default=str)[:500]}
        
        OCR Text: {content_sample}
        
        Extract and return this exact JSON structure:
        {{
            "order_id": "corrected_order_id",
            "client_name": "extracted_client",
            "order_date": "YYYY-MM-DD_if_found",
            "delivery_date": "YYYY-MM-DD_if_found", 
            "items": [
                {{
                    "product_code": "extracted_code",
                    "description": "extracted_description",
                    "quantity": 5,
                    "unit_price": 100.00,
                    "total_price": 500.00
                }}
            ],
            "order_total": 500.00,
            "currency": "USD",
            "confidence_score": 0.75,
            "ocr_corrections": ["list_of_corrections_made"]
        }}
        """
        
        response = self._make_llm_call(prompt, max_tokens=2000)
        result = self._parse_json_response(response, raw_data.copy())
        
        # Add enhancement info
        result['enhancement_info'] = {
            'enhanced_by_llm': True,
            'enhancement_type': 'scanned_document_processing',
            'enhancement_timestamp': datetime.utcnow().isoformat()
        }
        
        return result
    
    def parse_items_intelligently(self, text_content: str, current_items: List[Dict]) -> List[Dict]:
        """Use LLM to intelligently parse item tables with robust error handling"""
        
        # Limit content size
        content_sample = text_content[:2000] if len(text_content) > 2000 else text_content
        
        prompt = f"""
        Extract line items from this document. Return ONLY a valid JSON array with no additional text:
        
        Content: {content_sample}
        
        Current items: {json.dumps(current_items[:3], default=str)}
        
        Return this exact JSON array structure:
        [
            {{
                "product_code": "item_code",
                "description": "item_description",
                "quantity": 10,
                "unit_price": 25.50,
                "total_price": 255.00,
                "confidence": 0.9
            }}
        ]
        """
        
        response = self._make_llm_call(prompt, max_tokens=1500)
        
        # Try to parse as JSON array
        if response:
            try:
                # Look for JSON array in response
                array_match = re.search(r'\[.*\]', response, re.DOTALL)
                if array_match:
                    items = json.loads(array_match.group(0))
                    if isinstance(items, list):
                        # Validate and clean items
                        validated_items = []
                        for item in items:
                            if isinstance(item, dict):
                                # Ensure numeric fields
                                try:
                                    if 'quantity' in item:
                                        item['quantity'] = int(float(str(item['quantity'])))
                                    if 'unit_price' in item:
                                        item['unit_price'] = float(str(item['unit_price']))
                                    if 'total_price' in item:
                                        item['total_price'] = float(str(item['total_price']))
                                    validated_items.append(item)
                                except (ValueError, TypeError):
                                    item['confidence'] = 0.3
                                    validated_items.append(item)
                        return validated_items
            except json.JSONDecodeError:
                logger.error("Failed to parse items JSON")
        
        # Return original items with confidence scores if LLM parsing failed
        for item in current_items:
            if 'confidence' not in item:
                item['confidence'] = 0.5
        
        return current_items
    
    def calculate_overall_confidence(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        
        factors = []
        
        # Field completeness
        required_fields = ['order_id', 'client_name', 'items']
        filled_fields = sum(1 for field in required_fields if extraction_result.get(field))
        completeness = filled_fields / len(required_fields)
        factors.append(completeness * 0.4)
        
        # Items quality
        items = extraction_result.get('items', [])
        if items:
            item_confidence = sum(item.get('confidence', 0.5) for item in items) / len(items)
            factors.append(item_confidence * 0.3)
        else:
            factors.append(0.0)
        
        # Field confidences
        field_confidences = extraction_result.get('field_confidences', {})
        if field_confidences:
            avg_field_conf = sum(field_confidences.values()) / len(field_confidences)
            factors.append(avg_field_conf * 0.3)
        else:
            factors.append(0.5 * 0.3)
        
        return min(1.0, max(0.0, sum(factors)))


# Factory function
def create_robust_enhanced_extractor(api_key: str) -> RobustLLMEnhancedExtractor:
    """Factory function to create robust LLM extractor"""
    return RobustLLMEnhancedExtractor(api_key=api_key, model="gpt-4")


def robust_enhance_existing_extraction(extractor: RobustLLMEnhancedExtractor, 
                                     raw_content: str, 
                                     current_result: Dict[str, Any],
                                     file_path: str = None) -> Dict[str, Any]:
    """Enhanced extraction with comprehensive error handling"""
    
    try:
        # Always start with a copy
        result = current_result.copy()
        enhancement_log = []
        
        print(f"    🔍 Starting robust enhancement...")
        
        # Strategy 1: Metadata enhancement
        try:
            print(f"    📝 Enhancing metadata...")
            metadata_enhanced = extractor.enhance_metadata_extraction(raw_content, result)
            if metadata_enhanced and len(metadata_enhanced) > len(result):
                result.update(metadata_enhanced)
                enhancement_log.append("metadata_enhancement_success")
                print(f"    ✓ Metadata enhanced")
            else:
                enhancement_log.append("metadata_enhancement_minimal")
        except Exception as e:
            enhancement_log.append(f"metadata_enhancement_failed: {str(e)[:50]}")
            print(f"    ⚠ Metadata enhancement failed: {e}")
        
        # Strategy 2: Items enhancement
        try:
            print(f"    📋 Enhancing items...")
            current_items = result.get('items', [])
            items_enhanced = extractor.parse_items_intelligently(raw_content, current_items)
            if items_enhanced and len(items_enhanced) >= len(current_items):
                result['items'] = items_enhanced
                enhancement_log.append("items_enhancement_success")
                print(f"    ✓ Items enhanced: {len(items_enhanced)} items")
            else:
                enhancement_log.append("items_enhancement_minimal")
        except Exception as e:
            enhancement_log.append(f"items_enhancement_failed: {str(e)[:50]}")
            print(f"    ⚠ Items enhancement failed: {e}")
        
        # Strategy 3: Document type specific enhancement
        if file_path:
            file_ext = Path(file_path).suffix.lower() if isinstance(file_path, str) else file_path.suffix.lower()
            
            if file_ext in ['.xlsx', '.xls', '.csv']:
                try:
                    print(f"    📊 Applying Excel/CSV enhancement...")
                    excel_enhanced = extractor.enhance_excel_csv_extraction(result, raw_content)
                    if excel_enhanced.get('confidence_score', 0) > result.get('confidence_score', 0):
                        result.update(excel_enhanced)
                        enhancement_log.append("excel_csv_enhancement_success")
                        print(f"    ✓ Excel/CSV enhancement applied")
                except Exception as e:
                    enhancement_log.append(f"excel_csv_enhancement_failed: {str(e)[:50]}")
                    print(f"    ⚠ Excel/CSV enhancement failed: {e}")
            
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                try:
                    print(f"    🖼️ Applying scanned document enhancement...")
                    ocr_enhanced = extractor.enhance_scanned_document_extraction(result, raw_content)
                    if ocr_enhanced.get('confidence_score', 0) > result.get('confidence_score', 0):
                        result.update(ocr_enhanced)
                        enhancement_log.append("scanned_document_enhancement_success")
                        print(f"    ✓ Scanned document enhancement applied")
                except Exception as e:
                    enhancement_log.append(f"scanned_document_enhancement_failed: {str(e)[:50]}")
                    print(f"    ⚠ Scanned document enhancement failed: {e}")
        
        # Calculate final confidence
        try:
            final_confidence = extractor.calculate_overall_confidence(result)
            result['confidence_score'] = final_confidence
        except Exception as e:
            print(f"    ⚠ Confidence calculation failed: {e}")
        
        # Add comprehensive enhancement info
        result['enhancement_info'] = {
            'enhanced_by_llm': True,
            'enhancement_strategies': enhancement_log,
            'enhancement_timestamp': datetime.utcnow().isoformat(),
            'robust_mode': True,
            'original_confidence': current_result.get('confidence_score', 0),
            'final_confidence': result.get('confidence_score', 0),
            'success_rate': len([s for s in enhancement_log if 'success' in s]) / max(1, len(enhancement_log))
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Robust enhancement completely failed: {e}")
        current_result['enhancement_error'] = str(e)
        current_result['enhancement_timestamp'] = datetime.utcnow().isoformat()
        return current_result