from pathlib import Path
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    EasyOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from my_timer import my_timer


def process_single_pdf(input_doc: Path, output_type="markdown") -> str:
    if torch.cuda.is_available():
        acc_device = AcceleratorDevice.CUDA
    else:
        acc_device = AcceleratorDevice.CPU

    accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=acc_device,
        cuda_use_flash_attention2=True
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    doc = converter.convert(input_doc).document

    table_count = len(doc.tables) if hasattr(doc, "tables") else 0
    output = doc.export_to_markdown() if output_type == "markdown" else doc.export_to_dict()
    image_count = str(output).count("<!-- image -->")

    print(f"📄 {input_doc.name}")
    print(f"  Tables: {table_count}")
    print(f"  Images: {image_count}")
    print(f"{input_doc.name} done\n")

    header = f"\n\n===== {input_doc.name} =====\n\n"
    return header + str(output)


@my_timer
def main(folder_path: Path, output_type="markdown", combined_output_file="CombinedDoclingOutput.txt"):
    if not folder_path.exists() or not folder_path.is_dir():
        print("Folder path is invalid.")
        return

    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    combined_output = ""

    for pdf_file in pdf_files:
        content = process_single_pdf(pdf_file, output_type=output_type)
        combined_output += content

    # Write all output to one file
    try:
        with open(combined_output_file, "w", encoding="utf-8") as f:
            f.write(combined_output)
        print(f"\n All PDF outputs combined and saved to '{combined_output_file}'")
    except UnicodeEncodeError as e:
        print(f"Encoding error: {e}")


if __name__ == "__main__":
    # Replace this with your PDF folder path
    folder = Path("folder1")
    main(folder_path=folder, output_type="markdown")
