"""Image Generation Skill for creating AI-generated images."""

import base64
from typing import Any

from app.agents.skills.skill_base import SkillContext, SkillMetadata, SkillParameter, ToolSkill
from app.ai.image import image_generation_service
from app.core.logging import get_logger
from app.services.file_storage import file_storage_service

logger = get_logger(__name__)


class ImageGenerationSkill(ToolSkill):
    """Generates AI images from text descriptions."""

    metadata = SkillMetadata(
        id="image_generation",
        name="Image Generation",
        version="1.0.0",
        description=(
            "Generates high-quality images from text descriptions "
            "using AI (Gemini/Imagen or DALL-E)"
        ),
        category="creative",
        parameters=[
            SkillParameter(
                name="prompt",
                type="string",
                description=(
                    "Detailed description of the image to generate. "
                    "Be specific about style, composition, colors, and subject."
                ),
                required=True,
            ),
            SkillParameter(
                name="size",
                type="string",
                description=(
                    "Image size: 1024x1024 (square), 1792x1024 (landscape), "
                    "1024x1792 (portrait), or smaller sizes"
                ),
                required=False,
                default="1024x1024",
            ),
            SkillParameter(
                name="n",
                type="number",
                description="Number of images to generate (1-4)",
                required=False,
                default=1,
            ),
            SkillParameter(
                name="model",
                type="string",
                description="Leave empty to use server default. Only specify if user requests a specific model.",
                required=False,
                default=None,
            ),
            SkillParameter(
                name="quality",
                type="string",
                description="Image quality: standard or hd (provider support varies)",
                required=False,
                default="standard",
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "images": {
                    "type": "array",
                    "description": "Generated images with storage URLs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to access the stored image",
                            },
                            "storage_key": {
                                "type": "string",
                                "description": "Storage key for the image",
                            },
                            "base64_data": {
                                "type": "string",
                                "description": "Base64 encoded image data",
                            },
                            "index": {"type": "number"},
                        },
                    },
                },
                "prompt": {"type": "string", "description": "The prompt used"},
                "count": {"type": "number", "description": "Number of images generated"},
            },
        },
        required_tools=[],
        risk_level="medium",
        side_effect_level="low",
        data_sensitivity="internal",
        network_scope="external",
        idempotency_hint=False,
        max_iterations=1,
        tags=["image", "creative", "generation", "ai-art"],
    )

    async def execute(self, params: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        """Generate images from prompt."""
        prompt = params["prompt"]
        size = params.get("size", "1024x1024")
        n = int(params.get("n", 1))
        model = params.get("model")
        quality = params.get("quality", "standard")
        user_id = context.user_id

        logger.info(
            "image_generation_skill_generating",
            prompt=prompt[:100],
            size=size,
            n=n,
            model=model,
            user_id=user_id,
        )

        # Generate images using the image generation service
        results = await image_generation_service.generate_image(
            prompt=prompt,
            size=size,
            n=n,
            model=model,
            quality=quality,
        )

        if not results:
            raise ValueError("No images generated")

        # Save images to storage and format output
        images = []
        for i, result in enumerate(results):
            image_data: dict[str, Any] = {
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
            "image_generation_skill_completed",
            prompt=prompt[:50],
            image_count=len(images),
        )

        return {
            "images": images,
            "prompt": prompt,
            "count": len(images),
        }
