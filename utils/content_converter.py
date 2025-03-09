import os
from typing import Dict, Any, Optional
import logging
from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from pytube import YouTube
from models.advanced_ai import ai_model
from models.text_processor import TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentConverter:
    def __init__(self):
        self.text_processor = TextProcessor()
        
    def convert_content(self, file_path: str, content_type: str = None) -> Dict[str, Any]:
        """Convert various content types into standardized format."""
        try:
            if not content_type:
                content_type = self._detect_content_type(file_path)
            
            content = self._extract_content(file_path, content_type)
            
            # Process the extracted content
            processed_content = self.text_processor.process_text(content)
            
            # Store in vector database for similarity search
            content_id = f"{content_type}_{os.path.basename(file_path)}"
            ai_model.store_content_vector(
                content_id=content_id,
                content=content,
                content_type=content_type
            )
            
            return {
                "content": processed_content.simplified,
                "topics": processed_content.topics,
                "highlighted_terms": processed_content.highlighted_terms,
                "tags": processed_content.tags,
                "content_type": content_type,
                "content_id": content_id
            }
            
        except Exception as e:
            logger.error(f"Error converting content: {str(e)}")
            raise
            
    def _detect_content_type(self, file_path: str) -> str:
        """Detect content type from file extension or URL."""
        if file_path.startswith(('http://', 'https://')):
            if 'youtube.com' in file_path or 'youtu.be' in file_path:
                return 'video'
            return 'article'
            
        ext = file_path.lower().split('.')[-1]
        content_types = {
            'pdf': 'pdf',
            'docx': 'document',
            'doc': 'document',
            'pptx': 'presentation',
            'ppt': 'presentation',
            'txt': 'text'
        }
        return content_types.get(ext, 'unknown')
        
    def _extract_content(self, file_path: str, content_type: str) -> str:
        """Extract text content from various file types."""
        try:
            if content_type == 'pdf':
                return self._extract_from_pdf(file_path)
            elif content_type == 'document':
                return self._extract_from_docx(file_path)
            elif content_type == 'presentation':
                return self._extract_from_pptx(file_path)
            elif content_type == 'article':
                return self._extract_from_url(file_path)
            elif content_type == 'video':
                return self._extract_from_video(file_path)
            elif content_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
                
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            raise
            
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        text_content = []
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text_content.append(page.extract_text())
        return '\n'.join(text_content)
        
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from Word documents."""
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
    def _extract_from_pptx(self, file_path: str) -> str:
        """Extract text from PowerPoint presentations."""
        pptx = Presentation(file_path)
        text_content = []
        
        for slide in pptx.slides:
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text)
            text_content.append('\n'.join(slide_text))
            
        return '\n\n'.join(text_content)
        
    def _extract_from_url(self, url: str) -> str:
        """Extract text from web articles."""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        return '\n'.join(chunk for chunk in chunks if chunk)
        
    async def _extract_from_video(self, url: str) -> str:
        """Extract text content from video using AssemblyAI transcription."""
        try:
            # For YouTube videos, get the audio URL first
            if 'youtube.com' in url or 'youtu.be' in url:
                yt = YouTube(url)
                audio_stream = yt.streams.filter(only_audio=True).first()
                url = audio_stream.url
            
            # Use advanced AI model's audio processing
            audio_result = await ai_model.process_audio_content(url)
            return audio_result.get('text', '')
            
        except Exception as e:
            logger.error(f"Error extracting video content: {str(e)}")
            raise

# Create singleton instance
content_converter = ContentConverter()
