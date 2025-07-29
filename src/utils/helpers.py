"""
Utility functions for the AI Help Desk System.
"""
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import re

logger = logging.getLogger(__name__)


def generate_request_id(user_id: str = None) -> str:
    """
    Generate a unique request ID.
    
    Args:
        user_id: Optional user identifier
        
    Returns:
        Unique request ID string
    """
    timestamp = int(time.time() * 1000)  # milliseconds
    if user_id:
        return f"req_{user_id}_{timestamp}"
    else:
        return f"req_{timestamp}"


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
    
    return text


def extract_email_addresses(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Text to search for email addresses
        
    Returns:
        List of email addresses found
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text.
    
    Args:
        text: Text to search for phone numbers
        
    Returns:
        List of phone numbers found
    """
    # Simple phone number patterns
    patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        r'\b\d{10}\b'  # 1234567890
    ]
    
    phone_numbers = []
    for pattern in patterns:
        phone_numbers.extend(re.findall(pattern, text))
    
    return phone_numbers


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity based on common words.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def hash_text(text: str) -> str:
    """
    Generate a hash for text content.
    
    Args:
        text: Text to hash
        
    Returns:
        SHA-256 hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()


def format_response_for_display(response: str, max_width: int = 80) -> str:
    """
    Format response text for better display.
    
    Args:
        response: Response text to format
        max_width: Maximum line width
        
    Returns:
        Formatted response text
    """
    if not response:
        return ""
    
    # Simple word wrapping
    words = response.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


def parse_json_safely(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        # Clean common JSON formatting issues
        json_str = json_str.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {str(e)}")
        return None


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
    return bool(re.match(pattern, email))


def time_ago(timestamp: datetime) -> str:
    """
    Get human-readable time difference.
    
    Args:
        timestamp: Timestamp to compare
        
    Returns:
        Human-readable time difference string
    """
    now = datetime.now()
    if timestamp.tzinfo and not now.tzinfo:
        now = now.replace(tzinfo=timestamp.tzinfo)
    elif not timestamp.tzinfo and now.tzinfo:
        timestamp = timestamp.replace(tzinfo=now.tzinfo)
    
    diff = now - timestamp
    
    if diff.days > 7:
        return timestamp.strftime('%Y-%m-%d')
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"


def create_directory_structure(base_path: str) -> bool:
    """
    Create the required directory structure for the help desk system.
    
    Args:
        base_path: Base path for the application
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directories = [
            'data/knowledge_base/faqs',
            'data/knowledge_base/procedures',
            'data/knowledge_base/policies',
            'data/vector_store',
            'data/test_requests',
            'logs'
        ]
        
        for directory in directories:
            full_path = Path(base_path) / directory
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")
        return False


def load_test_requests(file_path: str = None) -> List[Dict[str, Any]]:
    """
    Load test requests from JSON file.
    
    Args:
        file_path: Path to test requests file
        
    Returns:
        List of test request dictionaries
    """
    if not file_path:
        file_path = "data/test_requests/sample_requests.json"
    
    try:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Test requests file not found: {file_path}")
            return []
    except Exception as e:
        logger.error(f"Error loading test requests: {str(e)}")
        return []


def save_response_log(
    request_id: str,
    request_text: str,
    response: str,
    metadata: Dict[str, Any] = None
) -> bool:
    """
    Save request and response to log file.
    
    Args:
        request_id: Unique request identifier
        request_text: Original request text
        response: Generated response
        metadata: Additional metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "request": request_text,
            "response": response,
            "metadata": metadata or {}
        }
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Append to daily log file
        log_file = f"logs/responses_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving response log: {str(e)}")
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    try:
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}


def benchmark_processing_time(func, *args, **kwargs) -> tuple:
    """
    Benchmark function execution time.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in benchmarked function: {str(e)}")
        return None, execution_time


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe file system usage.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-.')


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later ones taking precedence.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result