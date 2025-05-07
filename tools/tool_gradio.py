import os
import sys
import tempfile
import subprocess
import gradio as gr
import logging
import shutil
import huggingface_hub as hf
import re
from typing import List, Optional, Dict, Tuple
from tool_auto import get_hf_valid_files, normalize_hf_path

# Import the list of valid extensions directly from tool_auto.py
try:
    from tool_auto import VALID_SRC_EXTS
except ImportError:
    # Default fallback
    VALID_SRC_EXTS = [".safetensors", ] # ".pt", ".ckpt", ]

# Configure logging - set to DEBUG for troubleshooting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)  # Root logger

# Configuration
IS_HF_SPACE = False  # Set to True for Huggingface Space deployment. If false, it's expected to run locally.
MAX_FORMAT_SELECTIONS = 1 if IS_HF_SPACE else None  # None means unlimited
MAX_HF_MODEL_SELECTIONS = 1 if IS_HF_SPACE else None  # Limit to 1 in HF Space mode, unlimited in local mode

# Global reference to components that need to be accessed across functions
model_path = None

# Format Settings
FORMAT_GROUPS = {
    "Q2": ["Q2_K"],
    "Q3": ["Q3_K_S", "Q3_K_M"],
    "Q4": ["Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M"],
    "Q5": ["Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M"],
    "Q6": ["Q6_K"],
    "Q8": ["Q8_0"],
    "Other": ["base"],
}

def get_all_formats():
    """Get a flat list of all available formats"""
    all_formats = []
    for formats in FORMAT_GROUPS.values():
        all_formats.extend(formats)
    return all_formats

def resolve_hf_models(repo_path: str) -> List[Tuple[str, str]]:
    """Get list of valid models from a HF repo"""
    try:
        normalized_path = normalize_hf_path(repo_path)
        if not normalized_path:
            return []
        valid_files = get_hf_valid_files(normalized_path)
        return [(path, f"{path} ({arch})") for path, arch in valid_files.items()]
    except Exception as e:
        print(f"Error resolving HF models: {e}")
        return []

def search_hf_models(hf_path: str, status_callback=None) -> Dict[str, str]:
    """Search for valid model files in HF repository with status updates"""
    if status_callback:
        status_callback("Searching for model files...")

    try:
        valid_files = get_hf_valid_files(hf_path)
        if not valid_files:
            if status_callback:
                status_callback("No valid model files found.")
            return {}

        if status_callback:
            files_str = "\n".join([f"- {path} ({arch})" for path, arch in valid_files.items()])
            status_callback(f"Found {len(valid_files)} valid model file(s):\n{files_str}")

        return valid_files
    except Exception as e:
        if status_callback:
            status_callback(f"Error searching for models: {str(e)}")
        return {}

def update_visibility(upload_value, path_value) -> Dict[str, bool]:
    """Update component visibility based on input values"""
    return {
        "upload": not bool(path_value),
        "path": not bool(upload_value),
        "model_select": bool(path_value) and len(resolve_hf_models(path_value)) > 0
    }

def run_conversion_single(input_path: str, output_dir: str, selected_formats: List[str]) -> str:
    """Run conversion for a single model"""
    if not input_path:
        return "No input provided"

    if not selected_formats:
        return "No formats selected"

    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        # If it's a HF path, normalize it
        if '/' in input_path or input_path.startswith(('http://', 'https://')):
            normalized_path = normalize_hf_path(input_path)
            if not normalized_path:
                return f"Invalid HuggingFace path: {input_path}"

            valid_files = get_hf_valid_files(normalized_path)
            if not valid_files:
                return "No valid models found in HuggingFace repository"

        # Determine output directory
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)

        # Clean temp directory if it exists
        temp_output = os.path.join(temp_dir, "temp_output")
        if os.path.exists(temp_output):
            for item in os.listdir(temp_output):
                item_path = os.path.join(temp_output, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error cleaning temp directory: {e}")

        # Build command
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tool_auto_path = os.path.join(script_dir, "tool_auto.py")

        cmd = [
            sys.executable,
            tool_auto_path,
            "--src", input_path,
            "--quants", *selected_formats,
            "--output-dir", output_dir,
            "--temp-dir", temp_output
        ]

        print(f"Running command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                return f"Conversion failed for {input_path}:\n{stderr}"

            return f"Successfully converted {input_path}"
        except Exception as e:
            return f"Error converting {input_path}: {str(e)}"

def run_conversion(model_input, output_dir: str, selected_formats: List[str], hf_models: List[str] = None) -> str:
    """Run conversion for one or multiple models"""
    if not selected_formats:
        return "No formats selected"

    # Log parameters
    logging.info(f"Running conversion with: model_input={model_input}, formats={selected_formats}, hf_models={hf_models}")

    # Create a temporary directory for output in HF Space mode
    if IS_HF_SPACE:
        output_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary output directory: {output_dir}")
    elif not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Using output directory: {output_dir}")

    output_files = []  # To track created files for downloading
    results = []

    if IS_HF_SPACE:
        # In HF Space mode, handle either file upload or selected HF models
        if hf_models and len(hf_models) > 0:
            status_text = []

            def status_callback(msg):
                status_text.append(msg)
                logging.info(f"HF download: {msg}")

            # Get repository path from the model_path input
            repo_path = model_path.value if hasattr(model_path, 'value') else None
            if not repo_path:
                return "No repository path provided."

            # Normalize the repository path
            normalized_repo = normalize_hf_path(repo_path)
            if not normalized_repo:
                return f"Invalid HuggingFace repository path: {repo_path}"

            results.append(f"Using HuggingFace repository: {normalized_repo}")

            for model_file in hf_models:
                # Clean up the model name if needed
                if isinstance(model_file, tuple) and len(model_file) > 0:
                    model_file = model_file[0]
                elif isinstance(model_file, str) and " (" in model_file:
                    model_file = model_file.split(" (")[0]

                results.append(f"Processing model: {model_file}")

                try:
                    # Download the model file
                    results.append(f"Downloading {model_file} from {normalized_repo}...")

                    # Use huggingface_hub to download the file
                    downloaded_path = hf.hf_hub_download(
                        repo_id=normalized_repo,
                        filename=model_file,
                        local_dir=output_dir,
                        local_dir_use_symlinks=False
                    )

                    results.append(f"Successfully downloaded to {downloaded_path}")

                    # Run conversion on the downloaded file
                    result = run_conversion_single(downloaded_path, output_dir, selected_formats)
                    results.append(result)

                    # Track output files
                    if "Successfully converted" in result:
                        for file in os.listdir(output_dir):
                            file_path = os.path.join(output_dir, file)
                            if os.path.isfile(file_path) and file_path not in output_files and file_path != downloaded_path:
                                output_files.append(file_path)
                except Exception as e:
                    logging.exception(f"Error processing {model_file}")
                    results.append(f"Error processing {model_file}: {str(e)}")

            result_text = "\n".join(results)
        elif isinstance(model_input, gr.File) or (hasattr(model_input, 'name') and model_input):
            # Handle uploaded file
            file_path = model_input.name if hasattr(model_input, 'name') else str(model_input)
            logging.info(f"Processing uploaded file: {file_path}")

            result_text = run_conversion_single(file_path, output_dir, selected_formats)

            # Add download links for files
            if "Successfully converted" in result_text:
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        output_files.append(file_path)
        else:
            return "No input provided. Please upload a file or select a HuggingFace model."
    else:
        # In local mode, process multiple models sequentially
        if not model_input:
            return "No input provided. Please provide model paths or select models from HuggingFace."

        # Split input into lines and filter out empty lines
        model_paths = [path.strip() for path in model_input.split("\n") if path.strip()]

        if not model_paths:
            return "No valid model paths found. Please check your input."

        # Process each model
        for path in model_paths:
            if not os.path.exists(path):
                results.append(f"File not found: {path}")
                continue

            result = run_conversion_single(path, output_dir, selected_formats)
            results.append(f"Model {path}:\n{result}")

            # Add output files to the list
            if "Successfully converted" in result:
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path) and file_path not in output_files:
                        output_files.append(file_path)

        if not results:
            return "No models were processed successfully."

        result_text = "\n\n".join(results)

    # Add download links to the output
    if output_files:
        download_section = "\n\nDownload converted files:\n"
        for file_path in output_files:
            file_name = os.path.basename(file_path)
            download_section += f"- {file_name}: {file_path}\n"
        result_text += download_section

    return result_text

def main():
    # Setup initial interface if running offline
    setup_utils()

    # Create interface based on mode
    title = "GGUF Converter - Local" if not IS_HF_SPACE else "GGUF Converter - Spaces"

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")

        # Make model_path accessible to other functions
        global model_path

        # Use tabs for input selection in both modes
        with gr.Tabs() as input_tabs:
            if IS_HF_SPACE:
                # HF Space mode tabs
                with gr.TabItem("Upload Model") as upload_tab:
                    model_input = gr.File(
                        label="Upload Model File",
                        file_types=VALID_SRC_EXTS,
                        interactive=True
                    )

                    # Format selection dropdown for upload tab
                    formats_upload = gr.Dropdown(
                        choices=get_all_formats(),
                        label="Select Output Formats",
                        multiselect=True,
                        max_choices=MAX_FORMAT_SELECTIONS,
                        value=[]
                    )

                    # Upload convert button
                    upload_convert_btn = gr.Button("Convert")

                with gr.TabItem("HuggingFace Model") as hf_tab:
                    model_path = gr.Textbox(
                        label="Enter HuggingFace Model Path or URL",
                        placeholder=f"username/modelname or direct URL to {', '.join(VALID_SRC_EXTS)} file",
                        lines=1,
                        interactive=True
                    )
                    scan_btn = gr.Button("Scan for Models")
                    scan_status = gr.Textbox(
                        label="Scan Status",
                        lines=3,
                        value=f"Click 'Scan for Models' to search for supported files ({', '.join(VALID_SRC_EXTS)})."
                    )
                    model_select = gr.Dropdown(
                        label="Select Models",
                        choices=[],
                        multiselect=True,
                        max_choices=MAX_HF_MODEL_SELECTIONS,
                        visible=False,
                        allow_custom_value=False,
                        scale=1,
                        min_width=600,
                        # In local mode, make the dropdown taller to show more options
                        info="Select the model files to convert. Multiple selections are allowed in local mode." if not IS_HF_SPACE else None
                    )

                    # Format selection dropdown for HF tab
                    formats_hf = gr.Dropdown(
                        choices=get_all_formats(),
                        label="Select Output Formats",
                        multiselect=True,
                        max_choices=MAX_FORMAT_SELECTIONS,
                        value=[]
                    )

                    # HF convert button
                    hf_convert_btn = gr.Button("Convert")
            else:
                # Local mode tabs - all three options
                with gr.TabItem("Model Paths") as local_tab:
                    model_input_local = gr.Textbox(
                        label="Model Paths (one per line)",
                        placeholder="Enter local file paths",
                        lines=5
                    )

                    # Output settings for Model Paths tab
                    gr.Markdown("## Output Settings")
                    output_dir_local = gr.Textbox(
                        label="Output Directory",
                        placeholder="Leave empty to use default (/output)",
                        lines=1
                    )
                    formats_local = gr.Dropdown(
                        choices=get_all_formats(),
                        label="Select Output Formats",
                        multiselect=True,
                        max_choices=MAX_FORMAT_SELECTIONS,
                        value=[]
                    )
                    local_convert_btn = gr.Button("Convert")

                with gr.TabItem("Upload Model") as upload_tab:
                    model_input_upload = gr.File(
                        label="Upload Model File",
                        file_types=VALID_SRC_EXTS,
                        interactive=True
                    )

                    # Output settings for Upload tab
                    gr.Markdown("## Output Settings")
                    output_dir_upload = gr.Textbox(
                        label="Output Directory",
                        placeholder="Leave empty to use default (/output)",
                        lines=1
                    )
                    formats_upload = gr.Dropdown(
                        choices=get_all_formats(),
                        label="Select Output Formats",
                        multiselect=True,
                        max_choices=MAX_FORMAT_SELECTIONS,
                        value=[]
                    )
                    upload_convert_btn = gr.Button("Convert")

                with gr.TabItem("HuggingFace Model") as hf_tab:
                    model_path = gr.Textbox(
                        label="Enter HuggingFace Model Path or URL",
                        placeholder=f"username/modelname or direct URL to {', '.join(VALID_SRC_EXTS)} file",
                        lines=1,
                        interactive=True
                    )
                    scan_btn = gr.Button("Scan for Models")
                    scan_status = gr.Textbox(
                        label="Scan Status",
                        lines=3,
                        value=f"Click 'Scan for Models' to search for supported files ({', '.join(VALID_SRC_EXTS)})."
                    )
                    model_select = gr.Dropdown(
                        label="Select Models",
                        choices=[],
                        multiselect=True,
                        max_choices=MAX_HF_MODEL_SELECTIONS,
                        visible=False,
                        allow_custom_value=False,
                        scale=1,
                        min_width=600,
                        # In local mode, make the dropdown taller to show more options
                        info="Select the model files to convert. Multiple selections are allowed in local mode." if not IS_HF_SPACE else None
                    )

                    # Output settings for HF tab
                    gr.Markdown("## Output Settings")
                    output_dir_hf = gr.Textbox(
                        label="Output Directory",
                        placeholder="Leave empty to use default (/output)",
                        lines=1
                    )
                    formats_hf = gr.Dropdown(
                        choices=get_all_formats(),
                        label="Select Output Formats",
                        multiselect=True,
                        max_choices=MAX_FORMAT_SELECTIONS,
                        value=[]
                    )
                    hf_convert_btn = gr.Button("Convert")

        # Status output (common for both modes)
        with gr.Group():
            with gr.Column(scale=1):
                gr.Markdown("## Status")
                status = gr.Textbox(label="Conversion Status", lines=10)

                # Add file download component
                download_files = gr.File(
                    label="Download Converted Files",
                    file_count="multiple",
                    type="filepath",
                    interactive=False,
                    visible=False
                )

        # Set up event handlers
        def scan_models(path):
            """Scan for models in the HF repository or handle direct URL"""
            if not path:
                return [
                    gr.update(visible=False),
                    "Please enter a valid HuggingFace model path or direct URL."
                ]

            try:
                # Check if this is a direct URL to a safetensors file
                is_direct_url = False
                for ext in VALID_SRC_EXTS:
                    if path.lower().endswith(ext):
                        is_direct_url = True
                        break

                if is_direct_url:
                    logging.info(f"Direct file URL detected: {path}")
                    # Extract filename from URL
                    filename = os.path.basename(path)
                    # Always use tuples for consistency (path, display_text)
                    return [
                        gr.update(choices=[(path, filename)], visible=True, value=[path]),
                        f"Direct file URL detected: {filename}\nYou can click Convert to proceed."
                    ]

                # Regular HF repo scanning
                status_text = []
                def status_callback(msg):
                    status_text.append(msg)
                    logging.info(f"Scan status: {msg}")

                models = search_hf_models(path, status_callback)
                if not models:
                    return [
                        gr.update(visible=False),
                        "\n".join(status_text) or "No valid models found in repository."
                    ]

                # Create proper model choices for the dropdown - only include supported file types
                model_choices = []
                for file_path, arch in models.items():
                    # Check if file has a supported extension
                    is_valid = False
                    for ext in VALID_SRC_EXTS:
                        if file_path.lower().endswith(ext):
                            is_valid = True
                            break

                    if is_valid:
                        display_text = f"{file_path} ({arch})"
                        # Always store as tuples (path, display_text) for consistency
                        model_choices.append((file_path, display_text))

                if not model_choices:
                    return [
                        gr.update(visible=False),
                        f"No files with supported extensions ({', '.join(VALID_SRC_EXTS)}) found in repository."
                    ]

                logging.info(f"Found models: {model_choices}")

                # In local mode, pre-select all models to make multiple selection easier
                if not IS_HF_SPACE and MAX_HF_MODEL_SELECTIONS is None:
                    # Use file paths as values for consistency
                    preselected_values = [choice[0] for choice in model_choices]
                    logging.info(f"Pre-selecting all models in local mode: {preselected_values}")
                    return [
                        gr.update(choices=model_choices, visible=True, value=preselected_values),
                        f"Found {len(model_choices)} valid model file(s) with supported extensions ({', '.join(VALID_SRC_EXTS)}). All models are pre-selected for conversion."
                    ]
                else:
                    # In HF Space mode or if selection limit is set, don't pre-select
                    return [
                        gr.update(choices=model_choices, visible=True, value=[]),
                        f"Found {len(model_choices)} valid model file(s) with supported extensions ({', '.join(VALID_SRC_EXTS)})."
                    ]
            except Exception as e:
                logging.exception("Error in scan_models")
                return [
                    gr.update(visible=False),
                    f"Error: {str(e)}"
                ]

        # Connect scan buttons to handler in both modes
        scan_btn.click(
            fn=scan_models,
            inputs=[model_path],
            outputs=[model_select, scan_status]
        )

        if IS_HF_SPACE:
            # Process file upload
            def process_input_upload(file_upload, formats):
                """Process file upload input for conversion"""
                logging.info(f"Processing input: file={file_upload}, formats={formats}")

                # Check if we have formats selected
                if not formats:
                    return "No formats selected. Please select at least one format.", gr.update(value=None, visible=False)

                # Using file upload
                if file_upload:
                    logging.info(f"Using uploaded file: {file_upload}")
                    result_text = run_conversion(file_upload, None, formats)

                    # Check for downloads
                    files_to_download = []
                    lines = result_text.split("\n")
                    for line in lines:
                        if line.startswith("- ") and ":" in line:
                            file_path = line.split(":", 1)[1].strip()
                            if os.path.isfile(file_path):
                                files_to_download.append(file_path)

                    if files_to_download:
                        return result_text, gr.update(value=files_to_download, visible=True)
                    else:
                        return result_text, gr.update(value=None, visible=False)
                else:
                    return "No file uploaded. Please upload a model file.", gr.update(value=None, visible=False)

            # Process HF models
            def process_input_hf(hf_models, formats):
                """Process HuggingFace model input for conversion"""
                logging.info(f"Processing input: models={hf_models}, formats={formats}")

                # Check if we have formats selected
                if not formats:
                    return "No formats selected. Please select at least one format.", gr.update(value=None, visible=False)

                # Using HuggingFace models
                if hf_models and len(hf_models) > 0:
                    logging.info(f"Using selected HF models: {hf_models}")

                    # Add status update
                    status_text = f"Starting conversion of {len(hf_models)} HuggingFace model(s)...\n"

                    try:
                        # Process models one by one
                        repo_name = model_path.value
                        if not repo_name:
                            return "Repository path is empty. Please enter a valid HuggingFace repo.", gr.update(value=None, visible=False)

                        normalized_repo = normalize_hf_path(repo_name)
                        if not normalized_repo:
                            return f"Could not normalize repository path: {repo_name}", gr.update(value=None, visible=False)

                        logging.info(f"Using repo: {normalized_repo}")
                        status_text += f"Using repository: {normalized_repo}\n"

                        # Create list of models to process - clean model names
                        models_to_process = []
                        for model in hf_models:
                            # Extract just the model filename without architecture info
                            if isinstance(model, tuple) and len(model) > 0:
                                clean_model = model[0]
                            elif isinstance(model, str) and " (" in model:
                                clean_model = model.split(" (")[0]
                            else:
                                clean_model = model
                            models_to_process.append(clean_model)

                        logging.info(f"Models to process: {models_to_process}")
                        status_text += f"Models to process: {models_to_process}\n"

                        # Call run_conversion directly
                        try:
                            result_text = run_conversion(None, None, formats, models_to_process)

                            # Check for downloads
                            files_to_download = []
                            lines = result_text.split("\n")
                            for line in lines:
                                if line.startswith("- ") and ":" in line:
                                    file_path = line.split(":", 1)[1].strip()
                                    if os.path.isfile(file_path):
                                        files_to_download.append(file_path)

                            # Return results
                            if files_to_download:
                                return status_text + result_text, gr.update(value=files_to_download, visible=True)
                            else:
                                return status_text + result_text, gr.update(value=None, visible=False)
                        except Exception as e:
                            logging.exception(f"Error in run_conversion: {e}")
                            return status_text + f"\nError in conversion: {str(e)}", gr.update(value=None, visible=False)
                    except Exception as e:
                        logging.exception("Error processing HuggingFace models")
                        return f"Error processing models: {str(e)}", gr.update(value=None, visible=False)
                else:
                    return "No models selected. Please scan for models and select at least one.", gr.update(value=None, visible=False)

            # Connect buttons to handlers
            upload_convert_btn.click(
                fn=process_input_upload,
                inputs=[model_input, formats_upload],
                outputs=[status, download_files]
            )

            hf_convert_btn.click(
                fn=process_input_hf,
                inputs=[model_select, formats_hf],
                outputs=[status, download_files]
            )
        else:
            # Local mode handlers for each tab
            def process_local_paths(input, out_dir, formats):
                """Process local file paths input"""
                if not input:
                    return "No input provided. Please enter file paths.", gr.update(value=None, visible=False)
                if not formats:
                    return "No formats selected. Please select at least one format.", gr.update(value=None, visible=False)

                result_text = run_conversion(input, out_dir, formats)

                # Check for downloads
                files_to_download = []
                lines = result_text.split("\n")
                for line in lines:
                    if line.startswith("- ") and ":" in line:
                        file_path = line.split(":", 1)[1].strip()
                        if os.path.isfile(file_path):
                            files_to_download.append(file_path)

                if files_to_download:
                    return result_text, gr.update(value=files_to_download, visible=True)
                else:
                    return result_text, gr.update(value=None, visible=False)

            def process_upload(file_upload, out_dir, formats):
                """Process file upload in local mode"""
                if not file_upload:
                    return "No file uploaded. Please upload a model file.", gr.update(value=None, visible=False)
                if not formats:
                    return "No formats selected. Please select at least one format.", gr.update(value=None, visible=False)

                file_path = file_upload.name if hasattr(file_upload, 'name') else str(file_upload)
                result_text = run_conversion_single(file_path, out_dir, formats)

                # Check for downloads
                files_to_download = []
                lines = result_text.split("\n")
                for line in lines:
                    if line.startswith("- ") and ":" in line:
                        file_path = line.split(":", 1)[1].strip()
                        if os.path.isfile(file_path):
                            files_to_download.append(file_path)

                if files_to_download:
                    return result_text, gr.update(value=files_to_download, visible=True)
                else:
                    return result_text, gr.update(value=None, visible=False)

            def process_hf_models(hf_models, out_dir, formats):
                """Process HuggingFace models in local mode"""
                if not hf_models:
                    return "No models selected. Please scan for models and select at least one.", gr.update(value=None, visible=False)
                if not formats:
                    return "No formats selected. Please select at least one format.", gr.update(value=None, visible=False)

                # Add status update
                status_text = f"Starting conversion of {len(hf_models)} HuggingFace model(s)...\n"

                try:
                    # Process HuggingFace models by creating input paths for each model
                    repo_name = model_path.value
                    if not repo_name:
                        return "Repository path is empty. Please enter a valid HuggingFace repo.", gr.update(value=None, visible=False)

                    normalized_repo = normalize_hf_path(repo_name)
                    if not normalized_repo:
                        return f"Could not normalize repository path: {repo_name}", gr.update(value=None, visible=False)

                    logging.info(f"Using repo: {normalized_repo}")
                    status_text += f"Using repository: {normalized_repo}\n"

                    # Create a temporary directory to store downloaded models
                    temp_dir = tempfile.mkdtemp(prefix="hf_download_")

                    # Create input paths for run_conversion
                    input_paths = []

                    # Download each model file
                    for model in hf_models:
                        # Extract just the model filename without architecture info
                        if isinstance(model, tuple) and len(model) > 0:
                            clean_model = model[0]
                        elif isinstance(model, str) and " (" in model:
                            clean_model = model.split(" (")[0]
                        else:
                            clean_model = model

                        try:
                            status_text += f"Downloading {clean_model}...\n"
                            downloaded_path = hf.hf_hub_download(
                                repo_id=normalized_repo,
                                filename=clean_model,
                                local_dir=temp_dir,
                                local_dir_use_symlinks=False
                            )
                            input_paths.append(downloaded_path)
                            status_text += f"Successfully downloaded {clean_model}\n"
                        except Exception as e:
                            status_text += f"Error downloading {clean_model}: {str(e)}\n"
                            logging.exception(f"Error downloading {clean_model}")

                    if not input_paths:
                        return status_text + "No models could be downloaded successfully.", gr.update(value=None, visible=False)

                    # Create a single input string with one path per line for run_conversion
                    input_string = "\n".join(input_paths)
                    logging.info(f"Prepared input paths: {input_string}")

                    # Call run_conversion with the local paths
                    result_text = run_conversion(input_string, out_dir, formats)

                    # Check for downloads
                    files_to_download = []
                    lines = result_text.split("\n")
                    for line in lines:
                        if line.startswith("- ") and ":" in line:
                            file_path = line.split(":", 1)[1].strip()
                            if os.path.isfile(file_path):
                                files_to_download.append(file_path)

                    # Clean up temp directory
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logging.warning(f"Failed to clean up temp directory: {e}")

                    # Return results
                    if files_to_download:
                        return status_text + result_text, gr.update(value=files_to_download, visible=True)
                    else:
                        return status_text + result_text, gr.update(value=None, visible=False)
                except Exception as e:
                    logging.exception("Error processing HuggingFace models")
                    return f"Error processing models: {str(e)}", gr.update(value=None, visible=False)

            # Connect buttons to handlers
            local_convert_btn.click(
                fn=process_local_paths,
                inputs=[model_input_local, output_dir_local, formats_local],
                outputs=[status, download_files]
            )

            upload_convert_btn.click(
                fn=process_upload,
                inputs=[model_input_upload, output_dir_upload, formats_upload],
                outputs=[status, download_files]
            )

            hf_convert_btn.click(
                fn=process_hf_models,
                inputs=[model_select, output_dir_hf, formats_hf],
                outputs=[status, download_files]
            )

    # Launch the interface
    demo.queue().launch(show_api=False, share=False)

if __name__ == "__main__":
    main() 