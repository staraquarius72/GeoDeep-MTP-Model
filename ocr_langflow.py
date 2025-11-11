from pathlib import Path
from typing import Any, Dict
from langflow.base.data.utils import TEXT_FILE_TYPES, parse_text_file_to_record
from langflow.interface.custom.custom_component import CustomComponent
from langflow.schema import Record
from langchain_core.prompts import PromptTemplate
from langflow.field_typing import Prompt, TemplateField, Text
from PIL import Image
import pytesseract


class FileComponent(CustomComponent):
    display_name = "Files"
    description = "A generic file loader."

    def build_config(self) -> Dict[str, Any]:
        return {
            "path": {
                "display_name": "Path",
                "field_type": "file",
                "file_types": TEXT_FILE_TYPES,
                "info": f"Supported file types: {', '.join(TEXT_FILE_TYPES)}",
            },
            "silent_errors": {
                "display_name": "Silent Errors",
                "advanced": True,
                "info": "If true, errors will not raise an exception.",
            },
        }

    def load_file(self, path: str, silent_errors: bool = False) -> Record:
        resolved_path = self.resolve_path(path)
        path_obj = Path(resolved_path)
        extension = path_obj.suffix[1:].lower()
        if extension == "doc":
            raise ValueError("doc files are not supported. Please save as .docx")
        if extension not in TEXT_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {extension}")
        record = parse_text_file_to_record(resolved_path, silent_errors)
        self.status = record if record else "No data"
        return record or Record()

    def build(self, path: str, silent_errors: bool = False) -> Record:
        record = self.load_file(path, silent_errors)
        self.status = record
        return record


class ImageAnalysisComponent(CustomComponent):
    display_name = "Image Analysis"
    description = "Analyze images to extract labeled components."

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        # Implement custom logic to identify and extract labeled components
        components = self.extract_components(text)
        return components

    def extract_components(self, text: str) -> Dict[str, Any]:
        # Custom logic to parse text and extract components
        components = {}
        lines = text.split('\n')
        for line in lines:
            if "Component" in line:
                key, value = line.split(':')
                components[key.strip()] = value.strip()
        return components


class PromptComponent(CustomComponent):
    display_name: str = "Prompt"
    description: str = "Create a prompt template with dynamic variables."
    icon = "prompts"

    def build_config(self):
        return {
            "template": TemplateField(display_name="Template"),
            "code": TemplateField(advanced=True),
        }

    def build(self, template: Prompt, **kwargs) -> Text:
        from langflow.base.prompts.utils import dict_values_to_string

        prompt_template = PromptTemplate.from_template(Text(template))
        kwargs = dict_values_to_string(kwargs)
        kwargs = {k: "\n".join(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        try:
            formated_prompt = prompt_template.format(**kwargs)
        except Exception as exc:
            raise ValueError(f"Error formatting prompt: {exc}") from exc
        self.status = f'Prompt:\n"{formated_prompt}"'
        return formated_prompt


# Example usage
file_component = FileComponent()
record = file_component.build(path="path/to/your/forklift.pdf")

image_analysis_component = ImageAnalysisComponent()
components_info = image_analysis_component.analyze_image(image_path="path/to/extracted/image.png")

prompt_component = PromptComponent()
prompt = prompt_component.build(template="Identify the labeled components of the forklift:\n{components_info}",
                                components_info=components_info)
print(prompt)
