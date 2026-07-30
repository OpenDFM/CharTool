import logging
import time
from typing import Any, Optional, Dict
import sys
import os
import io
import base64
from uuid import uuid4

from sandbox_fusion import run_code_async, RunCodeRequest, set_endpoint
from qwen_vl_utils import fetch_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)

INPUT_IMAGE_NAME = "chart.png"
OUTPUT_IMAGE_NAME = "output.png"


class CustomSandboxFusionTool(BaseTool):
    """
    A tool for chartool.
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        set_endpoint(config.get("sandbox_fusion_url", ""))
        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        logger.info(f"Init SandboxFusionTool with config: {config}")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)
        
        # Get image from kwargs
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        img = fetch_image({"image": image})
        image_base64 = encode_image_to_base64(img)
        self._instance_dict[instance_id] = {
            "image": image_base64,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)
        
        instance_data = self._instance_dict[instance_id]
        image_base64 = instance_data["image"]
        result = await execute_code(code, image_base64=image_base64, fetch_output_image=True, timeout=timeout)
        text_content = f"Stdout:\n{result['stdout']}\nStderr:\n{result['stderr']}"
        image_content = []
        if result["images"]:
            if result['stdout'] == "" and result['stderr'] == "":
                text_content = ""
            for img_base64 in result["images"]:
                image_url = f"data:image/png;base64,{img_base64}"
                img = fetch_image({"image": image_url})
                image_content.append(img)
        return ToolResponse(text=text_content, image=image_content), None, None
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


def encode_image_to_base64(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def execute_code( 
    code: str, 
    image_base64: Optional[str] = None, 
    fetch_output_image: bool = True,
    timeout: int = 10) -> Dict[str, Any]:
    """
    Executes the code in the sandbox.
    
    Args:
        code: Python code string (no preprocessing, model writes complete code).
        image_base64: Base64 encoded image to be available as 'chart.png' in sandbox.
        fetch_output_image: Whether to fetch the output image from sandbox.
        timeout: Execution timeout in seconds.
        
    Returns:
        Dict containing stdout, stderr, and images (list of base64 strings).
    """
        
    # Prepare files to upload to sandbox
    files_to_upload = {}
    if image_base64:
        files_to_upload[INPUT_IMAGE_NAME] = image_base64
    
    fetch_files = [OUTPUT_IMAGE_NAME] if fetch_output_image else []
    request = RunCodeRequest(
        code=code,
        language="python",
        run_timeout=timeout,
        files=files_to_upload,
        fetch_files=fetch_files
    )
    
    try:
        result = await run_code_async(request)
        
        output = {
            "stdout": result.run_result.stdout if result.run_result else "",
            "stderr": result.run_result.stderr if result.run_result else "",
            "images": [],
            "status": result.status
        }
        
        if result.status == "Success" and result.files:
            for filename, b64_content in result.files.items():
                if b64_content:
                    output["images"].append(b64_content)
        
        return output
        
    except Exception as e:
        logger.error(f"Sandbox execution failed: {str(e)}")
        return {
            "stdout": "",
            "stderr": str(e),
            "images": [],
            "status": "SystemError"
        }