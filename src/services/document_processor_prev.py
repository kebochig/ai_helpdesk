"""
Document processor for loading and processing knowledge base files.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
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
        """Process a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of documents
                for i, item in enumerate(data):
                    doc = self._json_item_to_document(item, file_path, i)
                    if doc:
                        documents.append(doc)
            elif isinstance(data, dict):
                # Single document or structured data
                if 'documents' in data and isinstance(data['documents'], list):
                    # Structured format with documents array
                    for i, item in enumerate(data['documents']):
                        doc = self._json_item_to_document(item, file_path, i)
                        if doc:
                            documents.append(doc)
                else:
                    # Single document
                    doc = self._json_item_to_document(data, file_path, 0)
                    if doc:
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {str(e)}")
            return []
    
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
        """Convert a JSON item to a KnowledgeDocument."""
        try:
            # Extract required fields
            doc_id = item.get('id', f"{self._generate_doc_id(file_path)}_{index}")
            title = item.get('title', f"Document {index + 1}")
            content = item.get('content', item.get('text', ''))
            
            if not content:
                logger.warning(f"No content found in JSON item {index} from {file_path}")
                return None
            
            # Extract optional fields
            category_str = item.get('category')
            category = None
            if category_str:
                try:
                    category = RequestCategory(category_str)
                except ValueError:
                    logger.warning(f"Invalid category '{category_str}' in {file_path}")
            
            tags = item.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            
            return KnowledgeDocument(
                id=doc_id,
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