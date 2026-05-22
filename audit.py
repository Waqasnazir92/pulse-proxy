import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TIMEOUT = float(os.getenv("SUPABASE_TIMEOUT", "10"))

logger = logging.getLogger(__name__)


class SupabaseAuditClient:
    def __init__(self, url: Optional[str], key: Optional[str]) -> None:
        self.url = url.rstrip("/") if url else None
        self.key = key

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    async def insert_audit_log(
        self,
        *,
        agent_id: Optional[str],
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        bdi_score: Optional[float],
    ) -> bool:
        if not self.configured:
            logger.warning("Skipping audit log insert because Supabase is not configured")
            return False

        row = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "request_payload": request_payload,
            "response_payload": response_payload,
            "bdi_score": bdi_score,
        }
        headers = {
            "apikey": self.key or "",
            "authorization": f"Bearer {self.key}",
            "content-type": "application/json",
            "prefer": "return=minimal",
        }

        try:
            async with httpx.AsyncClient(timeout=SUPABASE_TIMEOUT) as client:
                response = await client.post(
                    f"{self.url}/rest/v1/audit_logs",
                    headers=headers,
                    json=row,
                )
                response.raise_for_status()
            return True
        except httpx.HTTPError:
            logger.exception("Failed to insert audit log")
            return False


supabase_audit_client = SupabaseAuditClient(SUPABASE_URL, SUPABASE_KEY)
