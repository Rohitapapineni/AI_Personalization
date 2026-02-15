import httpx
import hmac
import hashlib
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WebhookDispatcher:
    def __init__(self, db):
        self.db = db

    def _sign_payload(self, payload: dict, secret: str) -> str:
        """Generate HMAC-SHA256 signature so receivers can verify authenticity."""
        body = json.dumps(payload, separators=(',', ':')).encode()
        return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    async def fire(self, event: str, data: dict):
        """Find all registered webhooks for this event and dispatch them."""
        webhooks = self.db.get_webhooks_for_event(event)
        if not webhooks:
            return

        payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        tasks = [self._send(wh, payload) for wh in webhooks]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send(self, webhook: dict, payload: dict, retries: int = 3):
        """Send payload with retry logic and optional signature."""
        headers = {"Content-Type": "application/json"}
        
        if webhook.get("secret"):
            sig = self._sign_payload(payload, webhook["secret"])
            headers["X-Webhook-Signature"] = f"sha256={sig}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(retries):
                try:
                    response = await client.post(
                        webhook["url"],
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    logger.info(f"Webhook delivered to {webhook['url']} [{event}]")
                    return
                except Exception as e:
                    wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Attempt {attempt+1} failed for {webhook['url']}: {e}. Retrying in {wait}s")
                    if attempt < retries - 1:
                        await asyncio.sleep(wait)
            
            logger.error(f"All retries exhausted for webhook {webhook['url']}")
