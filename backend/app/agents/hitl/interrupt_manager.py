"""Interrupt Manager for Human-in-the-Loop (HITL) workflows.

Manages the lifecycle of interrupts using Redis pub/sub:
1. Agent creates interrupt and waits for response
2. Interrupt is stored in Redis and streamed to frontend
3. Frontend shows dialog and user responds
4. Response is published to Redis channel
5. Agent receives response and continues

Uses Redis for:
- Storage: Pending interrupts with TTL for reconnection recovery
- Pub/Sub: Real-time response delivery to waiting agents
"""

import asyncio
import json
import uuid
from typing import Any

import redis.asyncio as redis

from app.agents import events
from app.agents.events import InterruptType
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Redis key prefixes
INTERRUPT_KEY_PREFIX = "hitl:interrupt:"
INTERRUPT_CHANNEL_PREFIX = "hitl:response:"


class InterruptManager:
    """Manages interrupt lifecycle using Redis pub/sub."""

    def __init__(self, redis_url: str | None = None):
        """Initialize the interrupt manager.

        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        self._redis_url = redis_url or settings.redis_url
        self._redis: redis.Redis | None = None

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def generate_interrupt_id(self) -> str:
        """Generate a unique interrupt ID."""
        return f"int_{uuid.uuid4().hex[:12]}"

    async def create_interrupt(
        self,
        thread_id: str,
        interrupt_id: str,
        interrupt_data: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Store an interrupt in Redis for the given thread.

        Args:
            thread_id: Thread/conversation ID
            interrupt_id: Unique interrupt identifier
            interrupt_data: Interrupt event data
            ttl_seconds: Time-to-live in seconds (default from interrupt timeout)
        """
        r = await self._get_redis()
        key = f"{INTERRUPT_KEY_PREFIX}{thread_id}:{interrupt_id}"

        # Use interrupt timeout if not specified
        if ttl_seconds is None:
            ttl_seconds = interrupt_data.get("timeout_seconds", 120) + 30  # Add buffer

        await r.setex(key, ttl_seconds, json.dumps(interrupt_data))
        logger.info(
            "interrupt_created",
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            interrupt_type=interrupt_data.get("interrupt_type"),
        )

    async def wait_for_response(
        self,
        thread_id: str,
        interrupt_id: str,
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        """Wait for user response to an interrupt via Redis pub/sub.

        Args:
            thread_id: Thread/conversation ID
            interrupt_id: Interrupt ID to wait for
            timeout_seconds: Maximum wait time

        Returns:
            Response dict with action and optional value

        Raises:
            TimeoutError: If no response within timeout
        """
        r = await self._get_redis()
        channel = f"{INTERRUPT_CHANNEL_PREFIX}{thread_id}:{interrupt_id}"

        logger.info(
            "wait_for_response_subscribing",
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            channel=channel,
        )

        pubsub = r.pubsub()
        await pubsub.subscribe(channel)

        logger.info(
            "wait_for_response_subscribed",
            channel=channel,
            timeout_seconds=timeout_seconds,
        )

        try:
            # Wait for response with timeout using asyncio.wait_for on the listen generator
            async def wait_for_message():
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        return json.loads(message["data"])
                return None

            try:
                response = await asyncio.wait_for(
                    wait_for_message(),
                    timeout=timeout_seconds,
                )
                if response:
                    logger.info(
                        "interrupt_response_received",
                        thread_id=thread_id,
                        interrupt_id=interrupt_id,
                        action=response.get("action"),
                    )
                    return response
            except asyncio.TimeoutError:
                pass

            # Timeout - return default action
            logger.warning(
                "interrupt_timeout",
                thread_id=thread_id,
                interrupt_id=interrupt_id,
                timeout_seconds=timeout_seconds,
            )
            raise TimeoutError(f"No response to interrupt {interrupt_id} within {timeout_seconds}s")

        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def submit_response(
        self,
        thread_id: str,
        interrupt_id: str,
        action: str,
        value: str | None = None,
    ) -> bool:
        """Submit user response to an interrupt.

        Args:
            thread_id: Thread/conversation ID
            interrupt_id: Interrupt ID being responded to
            action: User action (approve, deny, skip, select, input)
            value: Selected value or input text

        Returns:
            True if response was published successfully
        """
        r = await self._get_redis()
        channel = f"{INTERRUPT_CHANNEL_PREFIX}{thread_id}:{interrupt_id}"

        response = {
            "action": action,
            "value": value,
            "interrupt_id": interrupt_id,
        }

        # Publish response to waiting agent
        subscribers = await r.publish(channel, json.dumps(response))

        # Clean up stored interrupt
        key = f"{INTERRUPT_KEY_PREFIX}{thread_id}:{interrupt_id}"
        await r.delete(key)

        logger.info(
            "interrupt_response_submitted",
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            action=action,
            subscribers=subscribers,
        )

        return subscribers > 0

    async def get_pending_interrupt(
        self,
        thread_id: str,
    ) -> dict[str, Any] | None:
        """Get any pending interrupt for a thread (for reconnection recovery).

        Args:
            thread_id: Thread/conversation ID

        Returns:
            Pending interrupt data or None if no pending interrupt
        """
        r = await self._get_redis()
        pattern = f"{INTERRUPT_KEY_PREFIX}{thread_id}:*"

        # Find all pending interrupts for this thread
        keys = []
        async for key in r.scan_iter(match=pattern):
            keys.append(key)

        if not keys:
            return None

        # Return the most recent interrupt (highest timestamp)
        latest_interrupt = None
        latest_timestamp = 0

        for key in keys:
            data = await r.get(key)
            if data:
                interrupt = json.loads(data)
                timestamp = interrupt.get("timestamp", 0)
                if timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_interrupt = interrupt

        return latest_interrupt

    async def cancel_interrupt(
        self,
        thread_id: str,
        interrupt_id: str,
    ) -> bool:
        """Cancel a pending interrupt.

        Args:
            thread_id: Thread/conversation ID
            interrupt_id: Interrupt ID to cancel

        Returns:
            True if interrupt was cancelled
        """
        r = await self._get_redis()
        key = f"{INTERRUPT_KEY_PREFIX}{thread_id}:{interrupt_id}"

        deleted = await r.delete(key)
        if deleted:
            logger.info(
                "interrupt_cancelled",
                thread_id=thread_id,
                interrupt_id=interrupt_id,
            )
        return deleted > 0


# Global singleton instance
_interrupt_manager: InterruptManager | None = None


def get_interrupt_manager() -> InterruptManager:
    """Get the global interrupt manager instance."""
    global _interrupt_manager
    if _interrupt_manager is None:
        _interrupt_manager = InterruptManager()
    return _interrupt_manager


def create_approval_interrupt(
    tool_name: str,
    args: dict[str, Any],
    interrupt_id: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Create an approval interrupt for a high-risk tool.

    Args:
        tool_name: Name of the tool requiring approval
        args: Tool arguments
        interrupt_id: Optional custom interrupt ID
        timeout_seconds: Optional custom timeout

    Returns:
        Interrupt event dict
    """
    from app.agents.hitl.tool_risk import get_tool_approval_message

    if interrupt_id is None:
        interrupt_id = get_interrupt_manager().generate_interrupt_id()

    title, message = get_tool_approval_message(tool_name, args)

    return events.interrupt(
        interrupt_id=interrupt_id,
        interrupt_type=InterruptType.APPROVAL,
        title=title,
        message=message,
        tool_info={
            "name": tool_name,
            "args": args,
        },
        default_action="deny",
        timeout_seconds=timeout_seconds or settings.hitl_approval_timeout,
    )


def create_decision_interrupt(
    title: str,
    message: str,
    options: list[dict[str, str]],
    interrupt_id: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Create a decision interrupt for multiple choice selection.

    Args:
        title: Dialog title
        message: Question/prompt
        options: List of options [{"label": "...", "value": "...", "description": "..."}]
        interrupt_id: Optional custom interrupt ID
        timeout_seconds: Optional custom timeout

    Returns:
        Interrupt event dict
    """
    if interrupt_id is None:
        interrupt_id = get_interrupt_manager().generate_interrupt_id()

    return events.interrupt(
        interrupt_id=interrupt_id,
        interrupt_type=InterruptType.DECISION,
        title=title,
        message=message,
        options=options,
        default_action="skip",
        timeout_seconds=timeout_seconds or settings.hitl_decision_timeout,
    )


def create_input_interrupt(
    title: str,
    message: str,
    interrupt_id: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Create an input interrupt for free-form text collection.

    Args:
        title: Dialog title
        message: Prompt for input
        interrupt_id: Optional custom interrupt ID
        timeout_seconds: Optional custom timeout

    Returns:
        Interrupt event dict
    """
    if interrupt_id is None:
        interrupt_id = get_interrupt_manager().generate_interrupt_id()

    return events.interrupt(
        interrupt_id=interrupt_id,
        interrupt_type=InterruptType.INPUT,
        title=title,
        message=message,
        default_action="skip",
        timeout_seconds=timeout_seconds or settings.hitl_decision_timeout,
    )


def create_confirm_interrupt(
    title: str,
    message: str,
    interrupt_id: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Create a confirm interrupt for yes/no confirmation.

    Args:
        title: Dialog title
        message: Confirmation prompt
        interrupt_id: Optional custom interrupt ID
        timeout_seconds: Optional custom timeout

    Returns:
        Interrupt event dict
    """
    if interrupt_id is None:
        interrupt_id = get_interrupt_manager().generate_interrupt_id()

    return events.interrupt(
        interrupt_id=interrupt_id,
        interrupt_type=InterruptType.CONFIRM,
        title=title,
        message=message,
        default_action="deny",
        timeout_seconds=timeout_seconds or settings.hitl_decision_timeout,
    )
