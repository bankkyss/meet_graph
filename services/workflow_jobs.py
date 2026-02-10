import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4


class JobNotFoundError(Exception):
    pass


class JobNotReadyError(Exception):
    def __init__(self, status: str):
        super().__init__(f"Job not ready: {status}")
        self.status = status


class JobFailedError(Exception):
    pass


class JobResultMissingError(Exception):
    pass


class WorkflowJobService:
    def __init__(self, logger: Any, output_dir: str = "output", retention_seconds: int = 86400) -> None:
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = retention_seconds

        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _iso_now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    @staticmethod
    def _file_ts() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M")

    async def _prune_jobs(self) -> None:
        if self.retention_seconds <= 0:
            return

        now = datetime.utcnow().timestamp()
        async with self._lock:
            stale_ids = []
            for job_id, item in self._jobs.items():
                status = str(item.get("status", ""))
                if status not in ("succeeded", "failed"):
                    continue
                done_ts = float(item.get("done_ts", 0.0) or 0.0)
                if done_ts <= 0:
                    continue
                if now - done_ts > self.retention_seconds:
                    stale_ids.append(job_id)

            for job_id in stale_ids:
                old = self._jobs.pop(job_id, None)
                if not old:
                    continue
                result_path = old.get("result_path")
                if isinstance(result_path, str) and result_path:
                    try:
                        p = Path(result_path)
                        if p.exists():
                            p.unlink()
                    except Exception:
                        self.logger.warning("Cannot delete old result file for job %s", job_id)

    async def _update_job(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.update(fields)
            record["updated_at"] = self._iso_now()

    async def _get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return None
            return dict(item)

    @staticmethod
    def _to_public(job: Dict[str, Any]) -> Dict[str, Any]:
        status = str(job.get("status", "unknown"))
        job_id = str(job.get("job_id", ""))
        payload: Dict[str, Any] = {
            "job_id": job_id,
            "workflow_tag": job.get("workflow_tag"),
            "status": status,
            "status_url": f"/jobs/{job_id}",
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
        }
        if status == "failed":
            payload["error"] = job.get("error")
        if status == "succeeded":
            payload["result_url"] = f"/jobs/{job_id}/result"
            payload["official_rewritten_count"] = job.get("official_rewritten_count")
            payload["filename"] = job.get("result_filename")
        return payload

    async def create_and_start(
        self,
        *,
        workflow: Any,
        workflow_tag: str,
        filename_prefix: str,
        init_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        await self._prune_jobs()

        job_id = uuid4().hex
        record = {
            "job_id": job_id,
            "workflow_tag": workflow_tag,
            "filename_prefix": filename_prefix,
            "status": "queued",
            "created_at": self._iso_now(),
            "updated_at": self._iso_now(),
            "done_ts": 0.0,
            "error": None,
            "result_path": None,
            "result_filename": None,
            "official_rewritten_count": None,
        }
        async with self._lock:
            self._jobs[job_id] = record

        self.logger.info("Queued job %s (%s)", job_id, workflow_tag)
        asyncio.create_task(
            self._run_job(
                job_id=job_id,
                workflow=workflow,
                workflow_tag=workflow_tag,
                filename_prefix=filename_prefix,
                init_state=init_state,
            )
        )
        return self._to_public(record)

    async def _run_job(
        self,
        *,
        job_id: str,
        workflow: Any,
        workflow_tag: str,
        filename_prefix: str,
        init_state: Dict[str, Any],
    ) -> None:
        await self._update_job(job_id, status="running", error=None)
        self.logger.info("Run workflow: %s (job=%s)", workflow_tag, job_id)
        try:
            out = await workflow.ainvoke(init_state)
            final_html = out["final_html"]
            official_count = out.get("official_rewritten_count")
            filename = f"{filename_prefix}_{self._file_ts()}_{job_id[:8]}.html"
            out_path = self.output_dir / filename
            out_path.write_text(final_html, encoding="utf-8")
            await self._update_job(
                job_id,
                status="succeeded",
                done_ts=datetime.utcnow().timestamp(),
                result_path=str(out_path),
                result_filename=filename,
                official_rewritten_count=official_count,
                error=None,
            )
            self.logger.info("Job succeeded: %s (%s)", job_id, out_path)
        except Exception as e:
            self.logger.exception("Workflow failed: %s (job=%s)", workflow_tag, job_id)
            await self._update_job(
                job_id,
                status="failed",
                done_ts=datetime.utcnow().timestamp(),
                error=f"เกิดข้อผิดพลาด: {str(e)}",
            )

    async def get_public(self, job_id: str) -> Dict[str, Any]:
        await self._prune_jobs()
        job = await self._get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        return self._to_public(job)

    async def get_result_meta(self, job_id: str) -> Tuple[Path, str]:
        await self._prune_jobs()
        job = await self._get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)

        status = str(job.get("status", ""))
        if status == "failed":
            raise JobFailedError(str(job.get("error") or "Job failed"))
        if status != "succeeded":
            raise JobNotReadyError(status)

        result_path_raw = job.get("result_path")
        if not isinstance(result_path_raw, str) or not result_path_raw:
            raise JobResultMissingError("Missing result path")

        result_path = Path(result_path_raw)
        if not result_path.exists():
            raise JobResultMissingError("Result file not found")

        filename = str(job.get("result_filename") or result_path.name)
        return result_path, filename
