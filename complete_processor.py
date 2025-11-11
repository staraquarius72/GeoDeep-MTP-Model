import ssl
import tempfile
import torch
import flask
from PyPDF2 import PdfReader
from docling.datamodel import pipeline_options
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import io
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import hashlib
import time
import re
import gc
from pathlib import Path

# LangChain Docling imports
from langchain_docling import DoclingLoader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, FormatOption, InputFormat

from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    EasyOcrOptions,
    RapidOcrOptions,
)


# Configuration Class
class Settings:
    def __init__(self):
        self.api_title = "PDF Summarization API with Llama and Docling (Flask Version)"
        self.api_version = "2.1.0"
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.default_model = os.getenv("LLAMA_MODEL", "llama3.2:latest")
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB
        self.max_text_length = int(os.getenv("MAX_TEXT_LENGTH", "50000"))  # characters
        self.model_timeout = int(os.getenv("MODEL_TIMEOUT", "300"))  # 5 minutes
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.cors_origins = ["*"]  # Or specify your frontend origins
        # Llama-specific parameters
        self.model_temperature = 0.3
        self.model_top_p = 0.9
        self.max_tokens = 2048
        self.chunk_size = 4000  # For handling large PDFs
        self.chunk_overlap = 200  # Overlap for chunking
        # Docling specific settings
        self.enable_ocr = True  # Enable OCR for images
        self.enable_table_extraction = True  # Enable table extraction


settings = Settings()

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/api/*": {"origins": settings.cors_origins}})


# Utility Functions
def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def validate_pdf_content(content: bytes) -> bool:
    """Validate PDF file format"""
    return content.startswith(b'%PDF-')


def remove_page_numbers(text: str) -> str:
    """Remove page numbers from text while preserving content"""
    # Remove standalone page numbers (numbers on their own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove page numbers at the beginning or end of lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove common page number patterns
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*\d+\s*/\s*\d+\s*\n', '\n', text)  # "1/10" style

    return text


def chunk_text(text: str, chunk_size: int = settings.chunk_size, overlap: int = settings.chunk_overlap) -> List[str]:
    """Split text into overlapping chunks for large documents"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            sentence_end = text.rfind('.', start + chunk_size - 200, end)
            if sentence_end != -1 and sentence_end > start:  # Ensure sentence_end is found and is forward
                end = sentence_end + 1

        chunks.append(text[start:end])
        start = end - overlap

        if start >= len(text):  # Ensure we don't go into an infinite loop if overlap is too large
            break
        if start < 0:  # Ensure start is not negative
            start = 0

    return chunks


class DoclingPDFProcessor:
    """PDF Processor using DoclingLoader for text, tables, and images"""

    def __init__(self):
        """Initialize Docling converter with optimal settings"""
        # Disable SSL verification for EasyOCR
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        # Configure pipeline options for better extraction - FIXED
        pipeline_options = PdfPipelineOptions(
            do_ocr=settings.enable_ocr,
            do_table_structure=True,
            do_image_analysis=True
        )

        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)

        # Add table structure options if available
        if hasattr(pipeline_options, 'table_structure_options'):
            pipeline_options.table_structure_options = {
                "do_cell_matching": True
            }

        format_option = FormatOption(
            backend=PyPdfiumDocumentBackend,
            pipeline_cls=StandardPdfPipeline,
            options=pipeline_options
        )

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: format_option}
        )

        # Try to initialize OCR with reduced languages
        try:
            from easyocr import Reader
            self.reader = Reader(['en'], gpu=True, download_enabled=True)
        except:
            logger.warning("Could not initialize EasyOCR")
            self.reader = None
            self.enable_ocr = False

    def generate_file_hash(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def _extract_title_from_content(self, content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "Untitled Document"
        for line in lines:
            if 5 < len(line) <= 120:
                return line
        return lines[0][:120]

    def process_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Process a single PDF and return extracted content and metadata"""

        if not validate_pdf_content(pdf_bytes):
            abort(400, description="Invalid PDF file format")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name
            logger.info("Temporary file created")

        try:
            loader = DoclingLoader(
                file_path=temp_file_path,
                converter=self.converter
            )
            doc = loader.load()
        except Exception as e:
            logger.error(f"Docling loading failed: {e}")
            abort(500, description="Failed to load document using Docling")

        finally:
            page_texts = [page.page_content for page in doc]
            full_text = "\n".join(page_texts)
            # Cleanup the temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info("Removed the temporary file")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {cleanup_error}")

        metadata = {
            "pages": len(page_texts),
            "title": None,
            "author": None,
            "creation_date": None,
            "tables_found": 0,
            "images_found": 0
        }

        return {
            "text": full_text,
            "metadata": metadata,
            "pages": page_texts,
            "tables": [],  # optional enhancement
            "images": [],
            "total_characters": len(full_text),
            "total_words": len(full_text.split()),
            "file_hash": self.generate_file_hash(pdf_bytes),
            "extraction_method": "docling"
        }

    def process_folder(self, folder_path: str) -> Dict[str, Any]:
        """Process all PDFs in a folder and return combined content"""

        combined_text = ""
        page_texts: List[str] = []
        total_pages = 0
        total_words = 0
        total_characters = 0

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
                result = self.process_pdf(pdf_bytes)

            combined_text += f"\n\n--- Content from {file_name} ---\n\n{result['text']}"
            page_texts.extend(result["pages"])
            total_pages += result["metadata"]["pages"]
            total_words += result["total_words"]
            total_characters += result["total_characters"]

        metadata = {
            "pages": total_pages,
            "title": "Combined PDF Folder Content",
            "author": None,
            "creation_date": None,
            "tables_found": 0,
            "images_found": 0
        }

        return {
            "text": combined_text.strip(),
            "metadata": metadata,
            "pages": page_texts,
            "tables": [],
            "images": [],
            "total_characters": total_characters,
            "total_words": total_words,
            "file_hash": None,
            "extraction_method": "docling-folder"
        }


# Llama Service Class (same as before but with enhanced PDF data)
class LlamaService:
    """LLM service with error handling for Flask"""

    def __init__(self):
        self.base_url = settings.ollama_url
        self.model = settings.default_model
        self.session = requests.Session()
        self.temperature = settings.model_temperature
        self.top_p = settings.model_top_p
        self.max_tokens = settings.max_tokens
        self.model_timeout = settings.model_timeout
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def check_model_availability(self) -> Dict[str, Any]:
        """Check if Llama model is available (synchronous)"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                available_models = [model['name'] for model in models_data]

                return {
                    "status": "healthy",
                    "available_models": available_models,
                    "model_ready": self.model in available_models or any(
                        self.model.split(':')[0] in model_name for model_name in available_models)
                }
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}

    def _summarize_large_document(self, pdf_data: Dict[str, Any], summary_type: str, language: str,
                                  custom_prompt: Optional[str]) -> Dict[str, Any]:
        """Summarize large documents by chunking (synchronous)"""
        text = pdf_data["text"]
        metadata = pdf_data["metadata"]
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        logger.info(f"Document too large, splitting into {len(chunks)} chunks.")

        chunk_summaries = []
        total_processing_time = 0

        for i, chunk_text_content in enumerate(chunks):
            logger.info(f"Summarizing chunk {i + 1}/{len(chunks)}")
            try:
                chunk_data = {
                    "text": chunk_text_content,
                    "tables": [],  # Not combined for now
                    "images": [],  # Not combined for now
                    "extraction_method": "combined_folder"
                }
                # Build prompt for the chunk with table/image context
                chunk_context = self._build_chunk_context(pdf_data, i, len(chunks))
                prompt = self._build_prompt_with_context_with_context(chunk_data, "brief", language,
                                                                      f"This is chunk {i + 1} of a larger document. "
                                                                      f"{chunk_context} Summarize this part concisely.")

                start_time = time.time()
                response = self._call_llama_api(prompt)
                processing_time = time.time() - start_time
                total_processing_time += processing_time

                summary_content = self._parse_llama_response(response)
                chunk_summaries.append(summary_content)
            except Exception as e:
                logger.error(f"Error summarizing chunk {i + 1}: {str(e)}")
                chunk_summaries.append(f"[Error summarizing chunk {i + 1}: {str(e)}]")

        # Combine chunk summaries
        combined_summary_text = "\n\n---\n\n".join(chunk_summaries)

        # Final summarization with document context
        logger.info("Generating final summary from combined chunk summaries...")
        doc_context = self._build_document_context(pdf_data)
        final_prompt_text = f"The following text consists of summaries from different parts of a large document. " \
                            f"{doc_context} Please synthesize these into a coherent '{summary_type}' " \
                            f"summary:\n\n{combined_summary_text}"
        final_data = {
            "text": final_prompt_text,
            "tables": [],  # Not combined for now
            "images": [],  # Not combined for now
            "extraction_method": "combined_folder"
        }
        final_prompt = self._build_prompt_with_context_with_context(final_data, summary_type, language, custom_prompt)

        start_time = time.time()
        final_response = self._call_llama_api(final_prompt)
        total_processing_time += (time.time() - start_time)
        final_summary_content = self._parse_llama_response(final_response)

        return {
            "title": "Document Summary (from chunks)",
            "summary_type": summary_type,
            "content": final_summary_content,
            "model_used": self.model,
            "processing_time": total_processing_time,
            "notes": f"Document was processed in {len(chunks)} chunks. Found {metadata.get('tables_found', 0)} tables and {metadata.get('images_found', 0)} images."
        }

    def _build_document_context(self, pdf_data: Dict[str, Any]) -> str:
        """Build context about document structure for better summarization"""
        context_parts = []

        metadata = pdf_data.get("metadata", {})

        if metadata.get("tables_found", 0) > 0:
            context_parts.append(f"This document contains {metadata['tables_found']} tables with structured data.")

        if metadata.get("images_found", 0) > 0:
            context_parts.append(f"This document contains {metadata['images_found']} images/figures.")

        if pdf_data.get("tables"):
            context_parts.append("Pay attention to tabular data and structured information.")

        return " ".join(context_parts)

    def _build_chunk_context(self, pdf_data: Dict[str, Any], chunk_index: int, total_chunks: int) -> str:
        """Build context for individual chunks"""
        context = f"Document has {pdf_data['metadata'].get('pages', 'unknown')} pages total."

        if pdf_data.get("tables"):
            context += f" Contains {len(pdf_data['tables'])} tables."

        if pdf_data.get("images"):
            context += f" Contains {len(pdf_data['images'])} images/figures."

        return context

    def summarize_pdf_text(self, pdf_data: Dict[str, Any], summary_type: str = "comprehensive",
                           language: str = "english", custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate summary from PDF text using Llama with enhanced context"""
        try:
            text = pdf_data["text"]
            metadata = pdf_data["metadata"]

            if len(text) > settings.max_text_length:
                logger.info(
                    f"Text length ({len(text)}) exceeds max_text_length ({settings.max_text_length}). Using chunking.")
                return self._summarize_large_document(pdf_data, summary_type, language, custom_prompt)
            print("custom_prompt", custom_prompt, language, summary_type)
            prompt = self._build_prompt_with_context_with_context(pdf_data, summary_type, language, custom_prompt)

            start_time = time.time()
            response = self._call_llama_api(prompt)
            processing_time = time.time() - start_time

            summary_content = self._parse_llama_response(response)

            return {
                "title": "Document Summary",
                "summary_type": summary_type,
                "content": summary_content,
                "model_used": self.model,
                "processing_time": processing_time,
                "extraction_info": {
                    "method": pdf_data.get("extraction_method", "unknown"),
                    "tables_found": metadata.get("tables_found", 0),
                    "images_found": metadata.get("images_found", 0)
                }
            }

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            abort(500, description=json.dumps({
                "error": "Failed to generate summary",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }))

    def _build_prompt_with_context_with_context(self, pdf_data: Dict[str, Any], summary_type: str, language: str,
                                                custom_prompt: Optional[str] = None) -> str:
        """Build prompt with enhanced context about document structure"""
        text = pdf_data["text"]
        doc_context = self._build_document_context(pdf_data)

        if custom_prompt:
            return f"""{custom_prompt}

{doc_context}

Document Content (in {language}):
{text}

Please provide your analysis based on the custom instructions:"""

        prompt_templates = {
            "comprehensive": f"""You are an expert document analyst. Please analyze the following document (in {language}) and create a comprehensive summary with these sections:

{doc_context}
Summarize the following text extracted from a PDF in 1–2 clear and concise paragraphs. Your summary must focus on extracting and highlighting critical financial and contractual details, including:
Seller quotations with all relevant financial figures
Discounts or special pricing offered by vendors
Comparison across multiple vendors in terms of pricing, products, and services
Key financial tables, numerical data, and any structured information
Terms and conditions that impact pricing, obligations, or the overall agreement
Roles and identities of buyers and sellers
Detailed descriptions of products and services
Even if information is spread across the document or embedded in tables, ensure no critical financial or contractual point is missed. Where possible, quantify differences between vendor offers and highlight implications for decision-making.
Make your summary detailed, well-structured, and professional. If there are tables or structured data, make sure to highlight key information from them. Ensure the summary is in {language}.""",

        }

        base_prompt = prompt_templates.get(summary_type, prompt_templates["comprehensive"])

        return f"""{base_prompt}

Document Content:
{text}

Please provide your analysis:"""

    def _call_llama_api(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API with Llama model (synchronous)"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.model_timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"Llama model timeout after {self.model_timeout}s for prompt: {prompt[:100]}...")
            abort(408, description="Llama model timeout - request took too long or document too complex")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama service at {self.base_url}")
            abort(503, description="Cannot connect to Ollama service")
        except requests.exceptions.HTTPError as e:
            error_msg = f"Llama API error: HTTP {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            abort(e.response.status_code, description=error_msg)
        except Exception as e:
            logger.error(f"Unexpected error calling Llama API: {str(e)}")
            abort(500, description=f"Unexpected error calling Llama API: {str(e)}")

    def _parse_llama_response(self, response: Dict[str, Any]) -> str:
        """Parse and format Llama model response"""
        content = response.get("response", "").strip()

        if not content:
            return "Summary could not be generated from the document."

        # Basic formatting
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)

        # Simple list formatting
        lines = content.split('\n')
        formatted_lines = []
        in_list = False
        for line in lines:
            stripped_line = line.strip()
            if (stripped_line.startswith('- ') or stripped_line.startswith('• ')):
                if not in_list:
                    formatted_lines.append('<ul>')
                    in_list = True
                formatted_lines.append(f'<li>{stripped_line[2:]}</li>')
            elif re.match(r'^\d+\.\s', stripped_line):
                if not in_list:
                    formatted_lines.append('<ol>')
                    in_list = True
                item_text = re.sub(r'^\d+\.\s', '', stripped_line)
                formatted_lines.append(f'<li>{item_text}</li>')
            else:
                if in_list:
                    if formatted_lines[-1].startswith('<ul>'):
                        formatted_lines.append('</ul>')
                    elif formatted_lines[-1].startswith('<ol>'):
                        formatted_lines.append('</ol>')
                    in_list = False
                if stripped_line:
                    formatted_lines.append(f'<p>{stripped_line}</p>')

        if in_list:
            if formatted_lines[-1].startswith('<ul>'):
                formatted_lines.append('</ul>')
            elif formatted_lines[-1].startswith('<ol>'):
                formatted_lines.append('</ol>')

        return '\n'.join(formatted_lines) if formatted_lines else content

    def summarize_folder(self, folder_path: str, extractor, summary_type: str = "comprehensive",
                         language: str = "english", custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all PDFs in the folder and generate a combined summary.
        :param folder_path: Folder path containing PDFs
        :param extractor: Instance of your PDFProcessor class
        """
        combined_text = ""
        total_metadata = {
            "pages": 0,
            "tables_found": 0,
            "images_found": 0
        }

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                    extracted_data = extractor.process_pdf(pdf_bytes)

                    combined_text += f"\n\n--- Content from: {file_name} ---\n\n{extracted_data['text']}\n"

                    meta = extracted_data.get("metadata", {})
                    total_metadata["pages"] += meta.get("pages", 0)
                    total_metadata["tables_found"] += meta.get("tables_found", 0)
                    total_metadata["images_found"] += meta.get("images_found", 0)

        pdf_data = {
            "text": combined_text,
            "metadata": total_metadata,
            "tables": [],  # Not combined for now
            "images": [],  # Not combined for now
            "extraction_method": "combined_folder"
        }

        return self.summarize_pdf_text(pdf_data, summary_type, language, custom_prompt)

    def summarize_docling_output(self, docling_result: Dict[str, Any], summary_type: str = "comprehensive",
                                 language: str = "english", custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize the result dictionary produced by DoclingPDFProcessor.process_folder().
        This method assumes all PDF content is already merged in 'docling_result'.
        """
        try:
            text = docling_result["text"]
            metadata = docling_result.get("metadata", {})

            if len(text) > settings.max_text_length:
                logger.info(
                    f"Text length ({len(text)}) exceeds max_text_length ({settings.max_text_length}). Using chunking.")
                return self._summarize_large_document(docling_result, summary_type, language, custom_prompt)

            prompt = self._build_prompt_with_context_with_context(docling_result, summary_type, language, custom_prompt)

            start_time = time.time()
            response = self._call_llama_api(prompt)
            processing_time = time.time() - start_time

            summary_content = self._parse_llama_response(response)

            return {
                "title": "Folder Summary",
                "summary_type": summary_type,
                "content": summary_content,
                "model_used": self.model,
                "processing_time": processing_time,
                "extraction_info": {
                    "method": docling_result.get("extraction_method", "unknown"),
                    "tables_found": metadata.get("tables_found", 0),
                    "images_found": metadata.get("images_found", 0)
                }
            }

        except Exception as e:
            logger.error(f"Error generating summary from folder result: {str(e)}")
            abort(500, description=json.dumps({
                "error": "Failed to generate summary from folder result",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }))


pdf_processor = DoclingPDFProcessor()
llama_service = LlamaService()


# Error Handling
@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(408)
@app.errorhandler(413)
@app.errorhandler(500)
@app.errorhandler(503)
def handle_error(error):
    response_data = {
        "success": False,
        "error": error.name,
        "message": error.description,
        "timestamp": datetime.now().isoformat()
    }
    try:
        if isinstance(error.description, str) and error.description.startswith("{"):
            description_json = json.loads(error.description)
            response_data["message"] = description_json.get("message", error.description)
            if "error" in description_json:
                response_data["error_details"] = description_json.get("error")
    except json.JSONDecodeError:
        pass

    return jsonify(response_data), error.code


# API Endpoints (same as before but updated info)
@app.route('/')
def serve_frontend_root():
    """Serves the main index.html from the static folder."""
    if not os.path.exists(os.path.join(app.static_folder, 'index.html')):
        logger.warning("static/index.html not found. Serving API info instead.")
        return jsonify({
            "message": f"Welcome to {settings.api_title}. Frontend not found at /static/index.html.",
            "version": settings.api_version,
        })
    return send_from_directory(app.static_folder, 'index.html')


# Folder-based PDF Processing
@app.route('/api/process-folder', methods=['POST'])
def process_folder():
    try:
        uploaded_files = request.files.getlist('files')
        folder_path = request.form.get("folder_path", "unknown-folder")
        summary_type = request.form.get("summary_type", "comprehensive")
        language = request.form.get("language", "english")
        custom_prompt = request.form.get("custom_prompt")

        logger.info("Data received")
        logger.info(f"{len(uploaded_files)} files received for processing.")
        if not uploaded_files:
            abort(400, description="No PDF files uploaded.")

        results = []
        total_metadata = {
            "pages": 0,
            "tables_found": 0,
            "images_found": 0
        }
        combined_text =""
        for file_storage in uploaded_files:
            filename = file_storage.filename

            if not filename.lower().endswith(".pdf"):
                continue

            try:
                content = file_storage.read()
                file_info = pdf_processor.process_pdf(content)

                if not file_info["text"].strip():
                    logger.warning(f"No readable text in {filename}") 
                    continue

                combined_text += f"\n\n--- Content from: {filename} ---\n\n{file_info['text']}\n"

                meta = file_info.get("metadata", {})
                total_metadata["pages"] += meta.get("pages", 0)
                total_metadata["tables_found"] += meta.get("tables_found", 0)
                total_metadata["images_found"] += meta.get("images_found", 0)

                results.append({
                    "filename": filename,
                    "metadata": {
                        "pages": meta.get("pages", 0),
                        "words": file_info.get("total_words", 0),
                        "tables": meta.get("tables_found", 0),
                        "images": meta.get("images_found", 0),
                        "file_hash": file_info.get("file_hash"),
                    }
                })

            except Exception as pdf_err:
                logger.error(f"Error processing {filename}: {str(pdf_err)}")
        pdf_data = {
            "text": combined_text,
            "metadata": total_metadata,
            "tables": [],  # Not combined for now
            "images": [],  # Not combined for now
            "extraction_method": "combined_folder"
        }
        logger.info("Combined text obtained. Generating Summary......")
        combined_summary = llama_service.summarize_pdf_text(
                            pdf_data, summary_type, language, custom_prompt
                        )
        logger.info("Summary generated. Process over.")
        gc.collect()
        logger.info("Garbage Collected.")
        return jsonify({
            "success": True,
            "processed_files": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "model_used": combined_summary.get("model_used", settings.default_model),
            "processing_time": combined_summary.get("processing_time"),
            "summary": combined_summary.get("content")
        })

    except Exception as e:
        logger.error(f"Unhandled folder processing error: {str(e)}", exc_info=True)
        abort(500, description=f"Unexpected error: {str(e)}")


@app.route('/frontend')
def serve_frontend_direct():
    if not os.path.exists(os.path.join(app.static_folder, 'index.html')):
        abort(404, description="Frontend index.html not found.")
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/api')
def api_root():
    """API information"""
    return jsonify({
        "message": f"Welcome to {settings.api_title}",
        "version": settings.api_version,
        "model": settings.default_model,
        "endpoints": {
            "health": "/api/health",
            "summarize": "/api/process-pdf",
            "info": "/api/pdf-info"
        },
        "summary_types": ["comprehensive", "executive", "key_points", "brief"],
        "max_file_size": format_file_size(settings.max_file_size),
        "features": {
            "table_extraction": settings.enable_table_extraction,
            "ocr_enabled": settings.enable_ocr,
            "page_number_removal": True,
            "enhanced_pdf_processing": "Docling"
        }
    })


# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    model_status = llama_service.check_model_availability()
    return jsonify({
        "status": "healthy" if model_status.get("status") == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ollama_status": model_status,
        "docling_status": "enabled",
    })


# Startup Checks
def initial_startup_checks():
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    try:
        from docling.document_converter import DocumentConverter
        logger.info("Docling successfully imported")
    except ImportError as e:
        logger.error(f"Docling import failed: {str(e)}")
    model_status = llama_service.check_model_availability()
    if model_status.get("status") == "healthy":
        logger.info(f"Ollama service healthy. Models: {model_status.get('available_models')}")


if __name__ == "__main__":
    initial_startup_checks()
    # For development, Flask's built-in server is fine.
    # For production, use a WSGI server like Gunicorn or uWSGI.
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 your_flask_app_file:app
    app.run(host="0.0.0.0", port=8000, debug=True)
