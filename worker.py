from __future__ import annotations

import asyncio
import logging
import signal
import sys

from config.settings import AppSettings
from core.ingest_store import (
    IngestJobCancelledError,
    list_pending_ingest_jobs,
    requeue_stale_running_ingest_jobs,
)
from core.ingestion import process_ingest_job
from core.job_queue import record_worker_heartbeat, wait_for_ingest_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("rag-worker")
STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


async def run_worker() -> int:
    settings = AppSettings.load()
    errors = [error for error in settings.validation_errors() if not error.startswith("OPENAI_")]
    if errors:
        LOGGER.error("Worker configuration invalid: %s", errors)
        return 1
    LOGGER.info("Worker started")
    last_recovery_at = 0.0
    while not STOP:
        try:
            await asyncio.to_thread(record_worker_heartbeat, settings)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Worker heartbeat failed: %s", exc)
        now = asyncio.get_running_loop().time()
        if now - last_recovery_at >= settings.worker_recovery_interval_seconds:
            try:
                recovered = await requeue_stale_running_ingest_jobs(
                    settings,
                    settings.ingest_running_stale_seconds,
                )
                if recovered:
                    LOGGER.warning("Requeued %s stale ingest job(s)", recovered)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Ingest recovery scan failed: %s", exc)
            last_recovery_at = now
        try:
            job_id = await asyncio.to_thread(wait_for_ingest_job, settings, 5)
        except Exception as exc:  # noqa: BLE001
            if STOP:
                break
            LOGGER.warning("Queue polling failed: %s", exc)
            await asyncio.sleep(1)
            continue
        if not job_id:
            # Redis is only a wake-up signal. The database remains the durable source
            # of truth, so jobs created while Redis was unavailable are still picked up.
            try:
                pending = await list_pending_ingest_jobs(settings, limit=1)
                job_id = pending[0].id if pending else None
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Pending ingest job recovery failed: %s", exc)
                continue
            if not job_id:
                continue
        LOGGER.info("Processing ingest job %s", job_id)
        try:
            await process_ingest_job(settings, job_id)
            LOGGER.info("Completed ingest job %s", job_id)
        except IngestJobCancelledError:
            LOGGER.info("Cancelled ingest job %s", job_id)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed ingest job %s", job_id)
    LOGGER.info("Worker stopped")
    return 0


def main() -> int:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    return asyncio.run(run_worker())


if __name__ == "__main__":
    sys.exit(main())
