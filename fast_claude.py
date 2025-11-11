import os
import time
import json
import asyncio
import base64
import fitz  # PyMuPDF
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from langflow.base.data import BaseFileComponent
from langflow.io import BoolInput, IntInput, MessageTextInput, SecretStrInput
from langflow.schema import Data
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from functools import lru_cache
import io  # Import io for in-memory PDF handling
from langflow.custom import Component


class Component(Component):
    display_name = "Claude_Vision_PDF_S3"
    description = "Ultra-fast PDF processing with Claude Vision from S3, with performance optimizations."
    icon = "bot"
    name = "ClaudeVisionPDFS3"
    filepath = ""
    VALID_EXTENSIONS = ["pdf"]
    silent_errors = False
    file_path = False
    inputs = [
        # Dummy inputs required by BaseFileComponent, not actively used for S3 download
        # MessageTextInput(name="file_path", display_name="Dummy File Path (Not Used for S3)", value="", advanced=True, info="This input is required by the base component but not used for S3 downloads. Can be left empty."),

        # Inputs for S3 file location - these will be used by the 'invoke' method primarily
        MessageTextInput(name="s3_bucket_name", display_name="S3 Bucket Name", value="langflow", required=True,
                         info="The name of the S3 bucket where the PDF is stored."),
        MessageTextInput(name="s3_object_key", display_name="S3 Object Key", value="Final.pdf", required=True,
                         info="The S3 object key (full path) to the PDF file within the bucket."),

        # S3 (MinIO) specific endpoint URL
        SecretStrInput(name="minio_endpoint_url", display_name="MinIO/Custom S3 Endpoint URL", value="", required=True,
                       info="Specify the full URL for your MinIO or other S3-compatible storage. Example: http://localhost:9000. Leave blank if using AWS S3."),
        SecretStrInput(name="minio_access_key_idl", display_name="MinIO/Custom S3 access_key", value="", required=True,
                       info="Specify the full URL for your MinIO or other S3-compatible storage. Example: http://localhost:9000. Leave blank if using AWS S3."),
        SecretStrInput(name="minio_secret_access_key", display_name="MinIO/Custom S3 secret_access_key", value="",
                       required=True,
                       info="Specify the full URL for your MinIO or other S3-compatible storage. Example: http://localhost:9000. Leave blank if using AWS S3."),
        # Claude Vision specific inputs
        MessageTextInput(name="prompt", display_name="Prompt",
                         value="Summarize the content of this document page by page in detail, highlighting key information and sections.",
                         info="The prompt to send to Claude Vision for each page or chunk."),
        MessageTextInput(name="model_id", display_name="Claude Model ID",
                         value="anthropic.claude-3-sonnet-20240229-v1:0",
                         info="The ID of the Claude Vision model to use (e.g., 'anthropic.claude-3-haiku-20240307-v1:0' for faster processing)."),

        # AWS Region and Credentials (These are for Bedrock)
        MessageTextInput(name="aws_region", display_name="AWS Region (for Bedrock)", value="us-east-1",
                         info="The AWS region where your Bedrock service is located."),
        SecretStrInput(name="aws_access_key_id", display_name="AWS Access Key ID",
                       info="The access key for your AWS account with Bedrock and S3 (if you were using AWS S3) permissions.",
                       value="AWS_ACCESS_KEY_ID", required=True),
        SecretStrInput(name="aws_secret_access_key", display_name="AWS Secret Access Key",
                       info="The secret key for your AWS account.", value="AWS_SECRET_ACCESS_KEY", required=True),

        # Performance optimization inputs
        IntInput(name="concurrency", display_name="Processing Concurrency", value=10, advanced=True,
                 info="Number of page chunks to process in parallel. Adjust based on your environment and API limits."),
        IntInput(name="dpi", display_name="Image DPI", value=72, advanced=True,
                 info="Image resolution (Dots Per Inch). Lower values (e.g., 72-100) are faster and often sufficient for LLMs. Higher values increase detail but slow down processing."),
        IntInput(name="jpeg_quality", display_name="JPEG Quality", value=85, advanced=True,
                 info="JPEG compression quality (1-100). Lower values (e.g., 75-80) result in smaller files and faster upload, but more compression artifacts."),
        BoolInput(name="use_jpeg", display_name="Use JPEG Instead of PNG", value=True, advanced=True,
                  info="Highly recommended to use JPEG for smaller file sizes and faster processing. PNG is lossless but much larger."),
        IntInput(name="chunk_size", display_name="Pages per Claude Request", value=2, advanced=True,
                 info="Number of PDF pages to combine into a single Claude Vision API request. Larger chunks reduce API call overhead but might hit Claude's image/token limits for very complex pages."),
    ]
    from langflow.template import Output
    from langflow.schema import Data

    outputs = [
        Output(name="data", method="process_pdf_from_bytes", type=Data)
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = ""
        self._bedrock_runtime_client = None
        self._s3_client = None
        self.doc = None
        file_path = ""

    def safe_log(self, message: str):
        try:
            self.log(message)
        except Exception:
            print(f"[SAFE_LOG] {message}")

    @lru_cache(maxsize=2)
    def get_aws_clients(self):
        # We need to ensure credentials are read even if only one client is initialized
        aws_access_key_id = self.aws_access_key_id
        aws_secret_access_key = self.aws_secret_access_key

        # Initialize S3 Client (for MinIO)
        if self._s3_client is None:
            self.safe_log(f"[INFO] Initializing S3 client.")
            # Determine which credentials to use based on whether we're using MinIO or AWS S3
            if self.minio_endpoint_url:  # Using MinIO
                s3_client_params = {
                    "region_name": self.aws_region,  # Region might still be needed by boto3 even for MinIO
                    "aws_access_key_id": "miniorootadmin",
                    "aws_secret_access_key": "m1n10@r00t@Psw",
                    "endpoint_url": "http://10.88.0.39:9000",
                    "verify": False  # Disable SSL verification for local MinIO
                }
                self.safe_log(f"[INFO] Using MinIO/Custom S3 endpoint: {self.minio_endpoint_url}")
            else:  # Using AWS S3
                s3_client_params = {
                    "region_name": self.aws_region,
                    "aws_access_key_id": self.aws_access_key_id,
                    "aws_secret_access_key": self.aws_secret_access_key
                }
                self.safe_log(f"[INFO] Using AWS S3 with region: {self.aws_region}")

            try:
                self._s3_client = boto3.client("s3", **s3_client_params)
                self.safe_log("[INFO] S3 client initialized successfully.")
            except Exception as e:
                self.safe_log(f"[ERROR] Failed to initialize S3 client: {e}")
                raise

        # Initialize Bedrock Runtime Client (for AWS)
        if self._bedrock_runtime_client is None:
            self.safe_log(f"[INFO] Initializing Bedrock Runtime client for region: {self.aws_region}")
            bedrock_client_params = {
                "region_name": self.aws_region,
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            }
            # NO endpoint_url for Bedrock unless you explicitly have a proxy

            try:
                self._bedrock_runtime_client = boto3.client("bedrock-runtime", **bedrock_client_params)
                self.safe_log("[INFO] Bedrock Runtime client initialized successfully.")
            except Exception as e:
                self.safe_log(f"[ERROR] Failed to initialize Bedrock Runtime client: {e}")
                raise

        return self._bedrock_runtime_client, self._s3_client

    def convert_pdf_page_to_base64_optimized(self, page_num: int) -> Tuple[Optional[str], Optional[str]]:
        try:
            if not self.doc:
                self.safe_log(
                    f"[ERROR] PDF document not opened for page {page_num + 1}. This indicates a prior failure in document loading.")
                return None, None

            page = self.doc.load_page(page_num)

            pix = page.get_pixmap(dpi=self.dpi, alpha=False, colorspace=fitz.csRGB)

            img_bytes = None
            media_type = None

            if self.use_jpeg:
                try:
                    img_bytes = pix.tobytes("jpeg", jpg_quality=self.jpeg_quality)
                    media_type = "image/jpeg"
                    if len(img_bytes) < 100:
                        self.safe_log(
                            f"[WARN] Generated JPEG for page {page_num + 1} is unusually small ({len(img_bytes)} bytes). Check page content or DPI settings.")
                except Exception as jpeg_e:
                    self.safe_log(
                        f"[WARN] Failed to convert page {page_num + 1} to JPEG ({jpeg_e}). Falling back to PNG for robustness.")
                    img_bytes = pix.tobytes("png")
                    media_type = "image/png"
            else:
                img_bytes = pix.tobytes("png")
                media_type = "image/png"

            if img_bytes:
                return base64.b64encode(img_bytes).decode("utf-8"), media_type
            else:
                self.safe_log(
                    f"[ERROR] No image bytes generated for page {page_num + 1}. Image conversion likely failed due to internal PyMuPDF error or empty page.")
                return None, None

        except Exception as e:
            self.safe_log(f"[ERROR] `convert_pdf_page_to_base64_optimized` failed for page {page_num + 1}: {e}")
            return None, None

    def process_page_chunk(self, page_chunk: List[int]) -> List[Tuple[int, str, Dict]]:
        results = []
        try:
            bedrock_runtime, _ = self.get_aws_clients()

            api_content_blocks = [{"type": "text", "text": self.prompt}]
            pages_included_in_api_call = []

            for page_num in page_chunk:
                self.safe_log(f"[INFO] Converting page {page_num + 1} to image for Claude request.")
                image_b64, media_type = self.convert_pdf_page_to_base64_optimized(page_num)

                if image_b64 and media_type:
                    api_content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        }
                    })
                    pages_included_in_api_call.append(page_num)
                else:
                    self.safe_log(
                        f"[ERROR] Could not generate image for page {page_num + 1}. Marking this page as failed within the chunk.")
                    results.append(
                        (page_num, f"[ERROR] Could not process page {page_num + 1} (image conversion failed).", {}))

            if not pages_included_in_api_call:
                self.safe_log(
                    f"[WARN] No valid images generated for chunk: {page_chunk}. Skipping Claude API call for this chunk.")
                return results

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{
                    "role": "user",
                    "content": api_content_blocks,
                }],
                "max_tokens": 8000,
            }

            self.safe_log(f"[INFO] Invoking Claude model {self.model_id} for chunk of pages: {page_chunk}")

            response = bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8"),
            )

            result = json.loads(response["body"].read().decode("utf-8"))
            content = result["content"][0]["text"]
            usage = result.get("usage", {})

            for page_num_from_api_call in pages_included_in_api_call:
                results.append((page_num_from_api_call,
                                f"\n\n=== Page {page_num_from_api_call + 1} Analysis ===\n\n{content}", usage))

        except bedrock_runtime.exceptions.ValidationException as ve:
            self.safe_log(f"[ERROR] Claude API Validation Error for chunk {page_chunk}: {ve}. Retrying page-by-page.")
            for page_num in page_chunk:
                try:
                    image_b64, media_type = self.convert_pdf_page_to_base64_optimized(page_num)
                    if not image_b64 or not media_type:
                        raise ValueError("Image conversion failed.")

                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_b64,
                                    }
                                }
                            ],
                        }],
                        "max_tokens": 8000,
                    }

                    response = bedrock_runtime.invoke_model(
                        modelId=self.model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body).encode("utf-8"),
                    )

                    result = json.loads(response["body"].read().decode("utf-8"))
                    content = result["content"][0]["text"]
                    usage = result.get("usage", {})

                    results.append((page_num, f"\n\n=== Page {page_num + 1} Analysis ===\n\n{content}", usage))

                except Exception as single_e:
                    self.safe_log(f"[ERROR] Failed processing individual page {page_num + 1}: {single_e}")
                    if not any(r[0] == page_num for r in results):
                        results.append(
                            (page_num, f"[ERROR] Validation failed for page {page_num + 1}: {str(single_e)}", {}))

        except bedrock_runtime.exceptions.ThrottlingException as te:
            self.safe_log(
                f"[ERROR] Claude API Throttling Error for chunk {page_chunk}. Consider reducing 'concurrency' or requesting higher Bedrock service quotas: {te}")
            for page_num in page_chunk:
                if not any(r[0] == page_num for r in results):
                    results.append((page_num, f"[ERROR] Throttled for page {page_num + 1} in chunk: {str(te)}", {}))
            time.sleep(2)

        except Exception as e:
            self.safe_log(f"[ERROR] Unhandled exception processing chunk {page_chunk}: {e}. Retrying page-by-page.")
            for page_num in page_chunk:
                try:
                    image_b64, media_type = self.convert_pdf_page_to_base64_optimized(page_num)
                    if not image_b64 or not media_type:
                        raise ValueError("Image conversion failed.")

                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_b64,
                                    }
                                }
                            ],
                        }],
                        "max_tokens": 8000,
                    }

                    response = bedrock_runtime.invoke_model(
                        modelId=self.model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body).encode("utf-8"),
                    )

                    result = json.loads(response["body"].read().decode("utf-8"))
                    content = result["content"][0]["text"]
                    usage = result.get("usage", {})

                    results.append((page_num, f"\n\n=== Page {page_num + 1} Analysis ===\n\n{content}", usage))

                except Exception as single_e:
                    self.safe_log(f"[ERROR] Failed processing individual page {page_num + 1}: {single_e}")
                    if not any(r[0] == page_num for r in results):
                        results.append(
                            (page_num, f"[ERROR] Chunk processing failed for page {page_num + 1}: {str(single_e)}", {}))

        return results

    def _extract_title_from_content(self, content: str) -> str:
        lines = content.split('\n', 5)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('==='):
                return line[:100]
        return "Untitled Document"

    def process_pdf_from_bytes(self) -> Data:
        bedrock_runtime_client, s3_client = self.get_aws_clients()

        self.safe_log(
            f"[INFO] Initiating S3 PDF processing for bucket: '{self.s3_bucket_name}', key: '{self.s3_object_key}'")

        pdf_bytes = None
        try:
            s3_response = s3_client.get_object(Bucket=self.s3_bucket_name, Key=self.s3_object_key)
            pdf_bytes = s3_response['Body'].read()
            self.safe_log(
                f"[INFO] Successfully downloaded s3://{self.s3_bucket_name}/{self.s3_object_key} ({len(pdf_bytes)} bytes).")
        except s3_client.exceptions.NoSuchKey:
            error_msg = f"[ERROR] S3 object not found: s3://{self.s3_bucket_name}/{self.s3_object_key}. Please verify bucket name and object key."
            self.safe_log(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"[ERROR] Failed to download S3 object s3://{self.s3_bucket_name}/{self.s3_object_key}: {e}. Check network, bucket policy, or credentials."
            self.safe_log(error_msg)
            raise RuntimeError(error_msg)

        original_file_name = os.path.basename(self.s3_object_key)
        start_time = time.time()

        try:
            self.doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            num_pages = len(self.doc)
            self.safe_log(f"[INFO] PDF '{original_file_name}' (downloaded from S3) has {num_pages} pages.")
            self.safe_log(f"[INFO] Processing with 'chunk_size': {self.chunk_size}, 'concurrency': {self.concurrency}.")
        except Exception as e:
            msg = f"[ERROR] Could not open PDF from bytes for '{original_file_name}': {e}"
            self.safe_log(msg)
            raise RuntimeError(msg)

        page_chunks = [list(range(i, min(i + self.chunk_size, num_pages))) for i in
                       range(0, num_pages, self.chunk_size)]
        self.safe_log(f"[INFO] Prepared {len(page_chunks)} chunks for parallel processing.")

        all_content = [None] * num_pages
        total_usage = {}
        processed_pages_count = 0

        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(page_chunks)),
                                thread_name_prefix="PDFChunkProcessor") as executor:
            futures = {
                executor.submit(self.process_page_chunk, chunk): chunk
                for chunk in page_chunks
            }
            self.safe_log(f"[INFO] Submitted {len(futures)} concurrent tasks for PDF chunks.")
            for future in as_completed(futures):
                original_chunk = futures[future]
                try:
                    chunk_results = future.result()
                    for page_num, content, usage in chunk_results:
                        if 0 <= page_num < num_pages:
                            all_content[page_num] = content
                            processed_pages_count += 1
                            for key, value in usage.items():
                                total_usage[key] = total_usage.get(key, 0) + value
                        else:
                            self.safe_log(
                                f"[WARN] Received result for invalid page number {page_num} from chunk {original_chunk}. Ignoring this specific page's result.")
                except Exception as e:
                    self.safe_log(
                        f"[ERROR] Unhandled exception occurred for chunk {original_chunk} (this should ideally be caught in process_page_chunk): {e}")
                    for page_num_in_failed_chunk in original_chunk:
                        if all_content[page_num_in_failed_chunk] is None:
                            all_content[
                                page_num_in_failed_chunk] = f"[ERROR] Processing of page {page_num_in_failed_chunk + 1} failed due to chunk error. Check logs for details."

        for i in range(num_pages):
            if all_content[i] is None:
                self.safe_log(
                    f"[WARN] Page {i + 1} has no content after all processing steps. Marking as unprocessed/failed.")
                all_content[i] = f"[ERROR] No content generated for page {i + 1} - possible processing failure."

        processing_time = time.time() - start_time
        pages_per_second = num_pages / processing_time if processing_time > 0 else 0

        try:
            self.doc.close()
            self.safe_log("[INFO] PDF document (from S3) closed successfully.")
        except Exception as e:
            self.safe_log(f"[WARN] Error closing PDF document (from S3): {e}")

        self.safe_log(
            f"[INFO] Total processing completed for {processed_pages_count}/{num_pages} pages in {processing_time:.2f} seconds ({pages_per_second:.1f} pages/sec).")
        if total_usage:
            self.safe_log("[INFO] Aggregated Claude API token usage:")
            for key, value in total_usage.items():
                self.safe_log(f"  {key}: {value}")

        file_hash = hashlib.md5(f"{self.s3_bucket_name}/{self.s3_object_key}".encode()).hexdigest()

        return Data(
            text="\n".join(all_content),
            pages=num_pages,
            tables_found=0,
            images_found=num_pages,
            file_path=f"s3://{self.s3_bucket_name}/{self.s3_object_key}",
            file_hash=file_hash,
            extraction_method="claude_vision_optimized",
            cuda_used=False,
            total_chunks=len(page_chunks),
            metadata={
                "usage": total_usage,
                "processing_time": processing_time,
                "pages_per_second": pages_per_second,
                "chunks_processed": len(page_chunks),
                "optimization_settings": {
                    "dpi": self.dpi,
                    "use_jpeg": self.use_jpeg,
                    "jpeg_quality": self.jpeg_quality if self.use_jpeg else None,
                    "concurrency": self.concurrency,
                },
                "original_file_name": original_file_name
            },
        )

