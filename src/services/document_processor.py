"""
Document processor for loading and processing knowledge base files.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import markdown
from langchain.document_loaders import (
    TextLoader, 
    JSONLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain.schema import Document
from src.models.schemas import KnowledgeDocument, RequestCategory
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing and loading knowledge base documents."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.knowledge_base_path = settings.system_config.knowledge_base_path
        self.supported_extensions = {'.md', '.json', '.txt', '.pdf'}
        
        # Common field names for content extraction (case insensitive)
        self.content_fields = {
            'content', 'text', 'body', 'description', 'message', 'data', 
            'value', 'details', 'info', 'summary', 'notes', 'document',
            'article', 'page_content', 'raw_text', 'full_text'
        }
        
        # Common field names for titles (case insensitive)
        self.title_fields = {
            'title', 'name', 'subject', 'heading', 'header', 'topic',
            'label', 'filename', 'file_name', 'document_name'
        }
        
        # Common field names for categories (case insensitive)
        self.category_fields = {
            'category', 'type', 'classification', 'class', 'kind',
            'section', 'department', 'group', 'tag_category'
        }
        
        # Common field names for tags (case insensitive)
        self.tag_fields = {
            'tags', 'keywords', 'labels', 'categories', 'topics',
            'metadata', 'properties', 'attributes'
        }
    
    def load_all_documents(self) -> List[KnowledgeDocument]:
        """
        Load all documents from the knowledge base directory.
        
        Returns:
            List of processed KnowledgeDocument objects
        """
        try:
            documents = []
            
            if not os.path.exists(self.knowledge_base_path):
                logger.warning(f"Knowledge base path does not exist: {self.knowledge_base_path}")
                return documents
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(self.knowledge_base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = Path(file).suffix.lower()
                    
                    if file_ext in self.supported_extensions:
                        try:
                            processed_docs = self._process_file(file_path)
                            documents.extend(processed_docs)
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {str(e)}")
                            continue
            
            logger.info(f"Loaded {len(documents)} documents from knowledge base")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return []
    
    def _process_file(self, file_path: str) -> List[KnowledgeDocument]:
        """
        Process a single file and convert to KnowledgeDocument objects.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of KnowledgeDocument objects
        """
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).stem
        
        try:
            if file_ext == '.md':
                return self._process_markdown(file_path)
            elif file_ext == '.json':
                return self._process_json(file_path)
            elif file_ext == '.txt':
                return self._process_text(file_path)
            elif file_ext == '.pdf':
                return self._process_pdf(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def _process_markdown(self, file_path: str) -> List[KnowledgeDocument]:
        """Process a Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert markdown to HTML and extract text
            html = markdown.markdown(content)
            
            # Extract metadata from filename and path
            category = self._extract_category_from_path(file_path)
            tags = self._extract_tags_from_content(content)
            
            document = KnowledgeDocument(
                id=self._generate_doc_id(file_path),
                title=Path(file_path).stem.replace('_', ' ').title(),
                content=content,
                category=category,
                tags=tags,
                source=file_path,
                last_updated=datetime.fromtimestamp(os.path.getmtime(file_path))
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"Error processing markdown file {file_path}: {str(e)}")
            return []
    
    def _process_json(self, file_path: str) -> List[KnowledgeDocument]:
        """Process a JSON file with flexible structure detection."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Recursively extract all potential documents from the JSON structure
            extracted_items = self._extract_json_items(data, file_path)
            
            for i, item in enumerate(extracted_items):
                doc = self._json_item_to_document(item, file_path, i)
                if doc:
                    documents.append(doc)
            
            # If no documents were extracted, treat the entire JSON as a single document
            if not documents:
                fallback_doc = self._create_fallback_document(data, file_path)
                if fallback_doc:
                    documents.append(fallback_doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {str(e)}")
            return []
    
    def _extract_json_items(self, data: Any, file_path: str, path: str = "") -> List[Dict[str, Any]]:
        """
        Recursively extract potential document items from JSON data.
        
        Args:
            data: JSON data to process
            file_path: Source file path
            path: Current path in the JSON structure
            
        Returns:
            List of potential document items
        """
        items = []
        
        if isinstance(data, dict):
            # Check if this dict looks like a document
            if self._is_document_like(data):
                items.append(data)
            else:
                # Look for arrays that might contain documents
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if isinstance(value, list):
                        # Process array items
                        for i, item in enumerate(value):
                            item_path = f"{current_path}[{i}]"
                            if isinstance(item, dict) and self._is_document_like(item):
                                items.append(item)
                            elif isinstance(item, (dict, list)):
                                items.extend(self._extract_json_items(item, file_path, item_path))
                    elif isinstance(value, dict):
                        # Recursively process nested objects
                        items.extend(self._extract_json_items(value, file_path, current_path))
        
        elif isinstance(data, list):
            # Process each item in the array
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                if isinstance(item, dict) and self._is_document_like(item):
                    items.append(item)
                elif isinstance(item, (dict, list)):
                    items.extend(self._extract_json_items(item, file_path, item_path))
        
        return items
    
    def _is_document_like(self, item: Dict[str, Any]) -> bool:
        """
        Check if a dictionary looks like a document.
        
        Args:
            item: Dictionary to check
            
        Returns:
            True if the item looks like a document
        """
        if not isinstance(item, dict) or not item:
            return False
        
        # Check for content fields
        has_content = any(
            key.lower() in self.content_fields 
            for key in item.keys()
        )
        
        # Check for meaningful text content (at least 10 characters)
        has_meaningful_content = any(
            isinstance(value, str) and len(value.strip()) >= 10
            for value in item.values()
        )
        
        # Must have either identifiable content fields or meaningful text
        return has_content or has_meaningful_content
    
    def _find_field_value(self, item: Dict[str, Any], field_names: set) -> Optional[str]:
        """
        Find a field value by checking multiple possible field names (case insensitive).
        
        Args:
            item: Dictionary to search
            field_names: Set of possible field names
            
        Returns:
            Field value if found, None otherwise
        """
        for key, value in item.items():
            if key.lower() in field_names and isinstance(value, str):
                return value.strip()
        return None
    
    def _find_best_content(self, item: Dict[str, Any]) -> str:
        """
        Find the best content field from a JSON item.
        
        Args:
            item: Dictionary to search
            
        Returns:
            Content string (may be empty)
        """
        # First, try to find explicit content fields
        content = self._find_field_value(item, self.content_fields)
        if content:
            return content
        
        # If no explicit content field, find the longest string value
        longest_text = ""
        for key, value in item.items():
            if isinstance(value, str) and len(value.strip()) > len(longest_text):
                longest_text = value.strip()
        
        # If still no content, create a summary of all fields
        if not longest_text:
            content_parts = []
            for key, value in item.items():
                if isinstance(value, (str, int, float, bool)):
                    content_parts.append(f"{key}: {value}")
            longest_text = "\n".join(content_parts)
        
        return longest_text
    
    def _extract_tags_from_item(self, item: Dict[str, Any]) -> List[str]:
        """
        Extract tags from a JSON item.
        
        Args:
            item: Dictionary to search
            
        Returns:
            List of tags
        """
        tags = []
        
        # Look for explicit tag fields
        for key, value in item.items():
            if key.lower() in self.tag_fields:
                if isinstance(value, list):
                    tags.extend([str(tag) for tag in value if tag])
                elif isinstance(value, str):
                    # Split by common delimiters
                    tag_parts = value.replace(',', ' ').replace(';', ' ').replace('|', ' ').split()
                    tags.extend(tag_parts)
        
        # Add content-based tags
        content = self._find_best_content(item)
        if content:
            content_tags = self._extract_tags_from_content(content)
            tags.extend(content_tags)
        
        # Remove duplicates and limit
        unique_tags = list(dict.fromkeys(tags))  # Preserve order
        return unique_tags[:15]  # Limit to 15 tags
    
    def _create_fallback_document(self, data: Any, file_path: str) -> Optional[KnowledgeDocument]:
        """
        Create a fallback document when no structured documents are found.
        
        Args:
            data: Original JSON data
            file_path: Source file path
            
        Returns:
            KnowledgeDocument or None
        """
        try:
            # Convert the entire JSON to a readable string
            content = json.dumps(data, indent=2, ensure_ascii=False)
            
            if len(content.strip()) < 10:  # Skip if too short
                return None
            
            category = self._extract_category_from_path(file_path)
            tags = self._extract_tags_from_content(content)
            
            document = KnowledgeDocument(
                id=self._generate_doc_id(file_path),
                title=Path(file_path).stem.replace('_', ' ').title(),
                content=content,
                category=category,
                tags=tags,
                source=file_path,
                last_updated=datetime.fromtimestamp(os.path.getmtime(file_path))
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating fallback document: {str(e)}")
            return None
    
    def _process_text(self, file_path: str) -> List[KnowledgeDocument]:
        """Process a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            category = self._extract_category_from_path(file_path)
            tags = self._extract_tags_from_content(content)
            
            document = KnowledgeDocument(
                id=self._generate_doc_id(file_path),
                title=Path(file_path).stem.replace('_', ' ').title(),
                content=content,
                category=category,
                tags=tags,
                source=file_path,
                last_updated=datetime.fromtimestamp(os.path.getmtime(file_path))
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def _process_pdf(self, file_path: str) -> List[KnowledgeDocument]:
        """Process a PDF file."""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Combine all pages into a single document
            content = "\n\n".join([page.page_content for page in pages])
            
            category = self._extract_category_from_path(file_path)
            tags = self._extract_tags_from_content(content)
            
            document = KnowledgeDocument(
                id=self._generate_doc_id(file_path),
                title=Path(file_path).stem.replace('_', ' ').title(),
                content=content,
                category=category,
                tags=tags,
                source=file_path,
                last_updated=datetime.fromtimestamp(os.path.getmtime(file_path))
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            return []
    
    def _json_item_to_document(
        self, 
        item: Dict[str, Any], 
        file_path: str, 
        index: int
    ) -> Optional[KnowledgeDocument]:
        """Convert a JSON item to a KnowledgeDocument with flexible field detection."""
        try:
            # Generate document ID
            item_id = item.get('id') or item.get('_id') or item.get('doc_id')
            if not item_id:
                item_id = f"{self._generate_doc_id(file_path)}_{index}"
            
            # Find title using flexible matching
            title = self._find_field_value(item, self.title_fields)
            if not title:
                title = f"Document {index + 1} from {Path(file_path).stem}"
            
            # Find content using flexible matching
            content = self._find_best_content(item)
            
            if not content or len(content.strip()) < 5:
                logger.warning(f"No meaningful content found in JSON item {index} from {file_path}")
                return None
            
            # Find category using flexible matching
            category = None
            category_str = self._find_field_value(item, self.category_fields)
            if category_str:
                try:
                    category = RequestCategory(category_str.lower())
                except ValueError:
                    logger.debug(f"Category '{category_str}' not in enum, will extract from path")
            
            if not category:
                category = self._extract_category_from_path(file_path)
            
            # Extract tags
            tags = self._extract_tags_from_item(item)
            
            return KnowledgeDocument(
                id=str(item_id),
                title=title,
                content=content,
                category=category,
                tags=tags,
                source=file_path,
                last_updated=datetime.fromtimestamp(os.path.getmtime(file_path))
            )
            
        except Exception as e:
            logger.error(f"Error converting JSON item to document: {str(e)}")
            return None
    
    def _extract_category_from_path(self, file_path: str) -> Optional[RequestCategory]:
        """Extract category from file path."""
        path_parts = Path(file_path).parts
        
        # Look for category in directory names
        for part in path_parts:
            part_lower = part.lower().replace(' ', '_').replace('-', '_')
            for category in RequestCategory:
                if category.value in part_lower or part_lower in category.value:
                    return category
        
        return None
    
    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract potential tags from content."""
        tags = []
        content_lower = content.lower()
        
        # Common IT terms that could be tags
        potential_tags = [
            'password', 'email', 'outlook', 'vpn', 'wifi', 'network',
            'printer', 'monitor', 'keyboard', 'mouse', 'software',
            'hardware', 'installation', 'configuration', 'troubleshooting',
            'security', 'policy', 'procedure', 'faq', 'guide'
        ]
        
        for tag in potential_tags:
            if tag in content_lower:
                tags.append(tag)
        
        return tags[:10]  # Limit to 10 tags
    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID from file path."""
        # Use relative path from knowledge base and replace separators
        rel_path = os.path.relpath(file_path, self.knowledge_base_path)
        doc_id = rel_path.replace(os.sep, '_').replace('.', '_')
        return doc_id
    
    def load_documents_by_category(self, category: RequestCategory) -> List[KnowledgeDocument]:
        """
        Load documents filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of documents in the specified category
        """
        all_documents = self.load_all_documents()
        return [doc for doc in all_documents if doc.category == category]
    
    def refresh_document(self, file_path: str) -> List[KnowledgeDocument]:
        """
        Refresh a specific document from file.
        
        Args:
            file_path: Path to the file to refresh
            
        Returns:
            List of refreshed documents
        """
        if os.path.exists(file_path):
            return self._process_file(file_path)
        else:
            logger.warning(f"File not found for refresh: {file_path}")
            return []


# Global document processor instance
document_processor = DocumentProcessor()