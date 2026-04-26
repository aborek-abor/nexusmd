"""
NexusMD — Job Queue Manager
Async job queue with Redis pub/sub for WebSocket streaming.
Falls back to in-memory queue if Redis is unavailable.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger("nexusmd.queue")

# Try to import Redis; fall back gracefully
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed — using in-memory queue only")


class Job:
    def __init__(self, job_id: str, description: str):
        self.job_id = job_id
        self.description = description
        self.status = "queued"
        self.progress = 0
        self.message = "Queued"
        self.result = None
        self.logs: List[dict] = []
        self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._redis: Optional[object] = None
        self._running = False

    async def start(self):
        self._running = True
        if REDIS_AVAILABLE:
            try:
                self._redis = aioredis.from_url(
                    "redis://localhost:6379",
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                await self._redis.ping()
                logger.info("Redis connected")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}) — in-memory mode")
                self._redis = None

    async def stop(self):
        self._running = False
        if self._redis:
            await self._redis.aclose()

    async def redis_ok(self) -> bool:
        if not self._redis:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    def create_job(self, description: str) -> str:
        job_id = str(uuid.uuid4())[:8].upper()
        self._jobs[job_id] = Job(job_id, description)
        logger.info(f"Job created: {job_id} — {description}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[dict]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: -j.created_at)]

    async def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.status = "cancelled"
        job.updated_at = time.time()
        await self._publish(job_id, {"type": "status", "status": "cancelled", "message": "Job cancelled"})
        return True

    async def update(self, job_id: str, status: str, progress: int, message: str, result=None):
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = status
        job.progress = progress
        job.message = message
        job.updated_at = time.time()
        if result is not None:
            job.result = result

        msg = {
            "type": "progress",
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
        }
        if result is not None:
            msg["result"] = result

        job.logs.append(msg)
        await self._publish(job_id, msg)

    async def log(self, job_id: str, line: str, level: str = "info"):
        """Stream a log line to WebSocket subscribers."""
        msg = {"type": "log", "job_id": job_id, "level": level, "line": line, "ts": time.time()}
        job = self._jobs.get(job_id)
        if job:
            job.logs.append(msg)
        await self._publish(job_id, msg)

    async def _publish(self, job_id: str, msg: dict):
        """Publish to both in-memory subscribers and Redis."""
        # In-memory subscribers (WebSocket connections on this process)
        q = self._subscribers.get(job_id)
        if q:
            await q.put(msg)

        # Redis pub/sub (for multi-process / multi-worker setups)
        if self._redis:
            try:
                await self._redis.publish(f"nexusmd:job:{job_id}", json.dumps(msg))
            except Exception as e:
                logger.debug(f"Redis publish failed: {e}")

    async def subscribe(self, job_id: str) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields job messages.
        Yields historical logs first, then live updates.
        """
        # First yield any historical logs
        job = self._jobs.get(job_id)
        if job:
            for msg in job.logs:
                yield msg
            if job.status in ("done", "failed", "cancelled"):
                return

        # Subscribe for live updates
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id] = q
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=60.0)
                    yield msg
                    if msg.get("status") in ("done", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield {"type": "ping", "job_id": job_id}
        finally:
            self._subscribers.pop(job_id, None)


# Singleton
job_manager = JobManager()
