"""Image generation tool for LangGraph agents."""

import base64
import json
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.image import image_generation_service
from app.core.logging import get_logger
from app.services.file_storage import file_storage_service

logger = get_logger(__name__)


class GenerateImageInput(BaseModel):
    """Input schema for image generation tool."""

    prompt: str = Field(
        description=(
            "Detailed description of the image to generate. "
            "Be specific about style, composition, colors, and subject."
        )
    )
    size: Literal[
        "512x512", "768x768", "512x768", "768x512",
        "1024x1024", "1792x1024", "1024x1792"
    ] = Field(
        default="1024x1024",
        description=(
            "Image size. Square (1024x1024), landscape (1792x1024), "
            "or portrait (1024x1792). Smaller sizes (512x, 768x) also available."
        ),
    )
    n: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4). More images take longer."
    )
    model: str | None = Field(
        default=None,
        description=(
            "Image generation model. Leave as None to use the server-configured default. "
            "Only specify a model if the user explicitly requests a specific one."
        ),
    )
    quality: Literal["standard", "hd"] = Field(
        default="standard",
        description=(
            "Image quality: 'standard' (default) or 'hd' for more detail "
            "(provider support varies)."
        ),
    )
    # Context fields (injected by agent, not provided by LLM)
    user_id: str | None = Field(
        default=None,
        description="User ID for storage (internal use only)",
        json_schema_extra={"exclude": True},
    )


@tool(args_schema=GenerateImageInput)
async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    n: int = 1,
    model: str | None = None,
    quality: str = "standard",
    user_id: str | None = None,
) -> str:
    """Generate images from a text description using AI.

    Use this tool when the user asks you to:
    - Create, generate, or make an image
    - Visualize or illustrate a concept
    - Design graphics, logos, or artwork
    - Create visual content for any purpose

    The tool generates high-quality images based on your prompt.
    Be specific and detailed in your prompt for best results.

    Generated images are automatically saved to storage for persistent access.

    Supports multiple providers (configured server-side).
    Do NOT specify a model unless the user explicitly requests one.

    Args:
        prompt: Detailed description of the desired image
        size: Image dimensions (default 1024x1024)
        n: Number of images to generate (1-4)
        model: Leave as None to use the server default. Only set if user requests a specific model.
        quality: Quality setting (standard or hd)
        user_id: User ID for storage (auto-injected by system)

    Returns:
        JSON string with image URLs and base64-encoded data
    """
    logger.info(
        "generate_image_tool_invoked",
        prompt=prompt[:100],
        size=size,
        n=n,
        model=model,
        quality=quality,
        user_id=user_id,
    )

    try:
        results = await image_generation_service.generate_image(
            prompt=prompt,
            size=size,
            n=n,
            model=model,
            quality=quality,
        )

        if not results:
            return json.dumps({
                "success": False,
                "error": "No images generated",
                "prompt": prompt,
            })

        # Save images to storage and format output
        images = []
        for i, result in enumerate(results):
            image_data = {
                "base64_data": result.base64_data,
                "index": i,
            }

            # Save to storage if user_id is available
            if user_id:
                try:
                    # Decode base64 to bytes
                    image_bytes = base64.b64decode(result.base64_data)

                    # Save to storage
                    storage_result = await file_storage_service.save_generated_image(
                        image_data=image_bytes,
                        user_id=user_id,
                        content_type="image/png",
                        metadata={
                            "prompt": prompt,
                            "model": model or "default",
                            "size": size,
                            "index": i,
                        },
                    )

                    # Add storage information to output
                    image_data["url"] = storage_result["url"]
                    image_data["storage_key"] = storage_result["storage_key"]

                    logger.info(
                        "image_saved_to_storage",
                        storage_key=storage_result["storage_key"],
                        user_id=user_id,
                    )
                except Exception as e:
                    logger.warning(
                        "image_storage_failed",
                        error=str(e),
                        index=i,
                    )
                    # Continue without storage - base64 data is still available
            else:
                logger.warning(
                    "image_not_saved_no_user_id",
                    index=i,
                )

            images.append(image_data)

        logger.info(
            "generate_image_completed",
            prompt=prompt[:50],
            image_count=len(images),
        )

        return json.dumps({
            "success": True,
            "images": images,
            "prompt": prompt,
            "count": len(images),
        })

    except Exception as e:
        logger.error(
            "generate_image_failed",
            prompt=prompt[:50],
            error=str(e),
        )
        return json.dumps({
            "success": False,
            "error": str(e),
            "prompt": prompt,
        })
