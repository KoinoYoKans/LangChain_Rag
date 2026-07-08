from __future__ import annotations

import asyncio
import logging
import signal
import sys

from config.settings import AppSettings
from core.ingest_store import list_recoverable_ingest_jobs
from core.ingestion import process_ingest_job
from core.job_queue import wait_for_ingest_job

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
    while not STOP:
        try:
            job_id = await asyncio.to_thread(wait_for_ingest_job, settings, 5)
        except TimeoutError:
            recoverable = await list_recoverable_ingest_jobs(settings, limit=10)
            if recoverable:
                job_id = recoverable[0].id
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            if STOP:
                break
            LOGGER.warning("Queue polling failed: %s", exc)
            await asyncio.sleep(1)
            continue
        if not job_id:
            continue
        LOGGER.info("Processing ingest job %s", job_id)
        try:
            await process_ingest_job(settings, job_id)
            LOGGER.info("Completed ingest job %s", job_id)
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
