import asyncio
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
        self.topic_time_padding_seconds = self._env_float("TOPIC_TIME_PADDING_SECONDS", 120.0)
        self.topic_time_cluster_gap_seconds = self._env_float("TOPIC_TIME_CLUSTER_GAP_SECONDS", 300.0)
        self.topic_time_enforce_non_overlap = self._env_flag("TOPIC_TIME_ENFORCE_NON_OVERLAP", False)
        self.topic_time_mode_default = self._resolve_topic_time_mode(
            os.getenv("TOPIC_TIME_MODE"),
            default="semantic",
        )

        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _iso_now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    @staticmethod
    def _file_ts() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M")

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        val = str(raw).strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        if val in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _normalize_title(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _resolve_topic_time_mode(raw: Any, *, default: str = "semantic") -> str:
        value = str(raw or "").strip().lower()
        if not value:
            return default
        mapping = {
            "semantic": "semantic",
            "cluster": "semantic",
            "overlap": "semantic",
            "default": "semantic",
            "chronological": "chronological",
            "agenda_order": "chronological",
            "agenda": "chronological",
            "non_overlap": "chronological",
            "strict": "chronological",
            "legacy": "legacy",
            "minmax": "legacy",
            "legacy_minmax": "legacy",
        }
        return mapping.get(value, default)

    def _apply_chronological_non_overlap(
        self,
        ranges: List[Dict[str, Any]],
        *,
        meeting_end: float,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        last_end = 0.0
        for item in ranges:
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            st = self._safe_float(item.get("starttime"))
            ed = self._safe_float(item.get("endtime"))
            if st is None and ed is None:
                continue
            if st is None:
                st = ed
            if ed is None:
                ed = st
            if st is None or ed is None:
                continue
            if ed < st:
                st, ed = ed, st
            duration = max(0.0, ed - st)
            if st < last_end:
                st = last_end
                ed = st + duration
            if meeting_end > 0:
                if st > meeting_end:
                    st = meeting_end
                if ed > meeting_end:
                    ed = meeting_end
            if ed < st:
                ed = st
            out.append(
                {
                    "name": name,
                    "starttime": round(st, 3),
                    "endtime": round(ed, 3),
                }
            )
            last_end = ed
        return out

    @staticmethod
    def _pick_best_cluster(
        intervals: List[Tuple[float, float]],
        *,
        cluster_gap_seconds: float,
    ) -> Optional[Tuple[float, float, int]]:
        if not intervals:
            return None
        ordered = sorted(intervals, key=lambda iv: iv[0])
        clusters: List[Tuple[float, float, int]] = []

        cur_start, cur_end = ordered[0]
        cur_count = 1
        for st, ed in ordered[1:]:
            if st - cur_end <= cluster_gap_seconds:
                cur_end = max(cur_end, ed)
                cur_count += 1
                continue
            clusters.append((cur_start, cur_end, cur_count))
            cur_start, cur_end, cur_count = st, ed, 1
        clusters.append((cur_start, cur_end, cur_count))

        # Prioritize cluster support count, then prefer tighter windows.
        return max(
            clusters,
            key=lambda c: (c[2], -(c[1] - c[0]), -c[1]),
        )

    def _extract_topic_time_ranges(
        self,
        *,
        workflow_out: Dict[str, Any],
        transcript_json: Dict[str, Any],
        mode: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        resolved_mode = self._resolve_topic_time_mode(
            mode,
            default=self.topic_time_mode_default,
        )
        parsed = workflow_out.get("parsed_agenda")
        kg = workflow_out.get("kg")
        if not isinstance(parsed, dict) or not isinstance(kg, dict):
            return [], resolved_mode

        agendas = parsed.get("agendas")
        nodes = kg.get("nodes")
        edges = kg.get("edges")
        segments = transcript_json.get("segments") if isinstance(transcript_json, dict) else None
        if not isinstance(agendas, list) or not isinstance(nodes, dict) or not isinstance(edges, list) or not isinstance(segments, list):
            return [], resolved_mode
        meeting_end = 0.0
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            st = self._safe_float(seg.get("start"))
            ed = self._safe_float(seg.get("end"))
            if st is not None:
                meeting_end = max(meeting_end, st)
            if ed is not None:
                meeting_end = max(meeting_end, ed)

        agenda_ids_by_norm: Dict[str, List[str]] = {}
        for node_id, payload in nodes.items():
            if not isinstance(payload, dict):
                continue
            if str(payload.get("type", "") or "") != "agenda":
                continue
            title_norm = self._normalize_title(payload.get("title"))
            if not title_norm:
                continue
            agenda_ids_by_norm.setdefault(title_norm, []).append(str(node_id))

        out: List[Dict[str, Any]] = []
        for ag in agendas:
            if not isinstance(ag, dict):
                continue
            topic_name = str(ag.get("title", "") or "").strip()
            if not topic_name:
                continue

            agenda_id = None
            key = self._normalize_title(topic_name)
            choices = agenda_ids_by_norm.get(key) or []
            if choices:
                agenda_id = choices.pop(0)
            if not agenda_id:
                continue

            source_segment_ids: set[int] = set()
            for edge in edges:
                if not (isinstance(edge, (list, tuple)) and len(edge) == 3):
                    continue
                src, rel, dst = edge
                if str(src) != agenda_id:
                    continue
                rel_text = str(rel or "")
                if rel_text not in {"has_topic", "has_action", "has_decision"}:
                    continue
                target = nodes.get(str(dst))
                if not isinstance(target, dict):
                    continue
                raw_segments = target.get("source_segments")
                if not isinstance(raw_segments, list):
                    continue
                for sid in raw_segments:
                    try:
                        sid_int = int(sid)
                    except Exception:
                        continue
                    source_segment_ids.add(sid_int)

            intervals: List[Tuple[float, float]] = []
            starts: List[float] = []
            ends: List[float] = []
            for sid in sorted(source_segment_ids):
                if sid < 0 or sid >= len(segments):
                    continue
                seg = segments[sid]
                if not isinstance(seg, dict):
                    continue
                st = self._safe_float(seg.get("start"))
                ed = self._safe_float(seg.get("end"))
                if st is None and ed is None:
                    continue
                if st is None:
                    st = ed
                if ed is None:
                    ed = st
                if st is None or ed is None:
                    continue
                starts.append(st)
                ends.append(ed)
                padded_start = max(0.0, st - self.topic_time_padding_seconds)
                padded_end = max(padded_start, ed + self.topic_time_padding_seconds)
                if meeting_end > 0:
                    padded_start = min(padded_start, meeting_end)
                    padded_end = min(padded_end, meeting_end)
                intervals.append((padded_start, padded_end))

            if resolved_mode == "legacy":
                if not starts or not ends:
                    continue
                starttime = min(starts)
                endtime = max(ends)
            else:
                best_cluster = self._pick_best_cluster(
                    intervals,
                    cluster_gap_seconds=self.topic_time_cluster_gap_seconds,
                )
                if not best_cluster:
                    continue
                starttime, endtime, _count = best_cluster

            out.append(
                {
                    "name": topic_name,
                    "starttime": round(starttime, 3),
                    "endtime": round(endtime, 3),
                }
            )

        if resolved_mode == "chronological":
            out = self._apply_chronological_non_overlap(out, meeting_end=meeting_end)

        if self.topic_time_enforce_non_overlap and resolved_mode != "chronological" and len(out) > 1:
            ordered_indices = sorted(
                range(len(out)),
                key=lambda i: self._safe_float(out[i].get("starttime")) or 0.0,
            )
            for pos in range(1, len(ordered_indices)):
                prev = out[ordered_indices[pos - 1]]
                curr = out[ordered_indices[pos]]
                prev_end = self._safe_float(prev.get("endtime"))
                curr_start = self._safe_float(curr.get("starttime"))
                if prev_end is None or curr_start is None:
                    continue
                if curr_start >= prev_end:
                    continue
                boundary = round((curr_start + prev_end) / 2.0, 3)
                prev["endtime"] = boundary
                curr["starttime"] = boundary
                curr_end = self._safe_float(curr.get("endtime"))
                if curr_end is not None and curr_end < boundary:
                    curr["endtime"] = boundary
                prev_start = self._safe_float(prev.get("starttime"))
                if prev_start is not None and boundary < prev_start:
                    prev["starttime"] = boundary
        return out, resolved_mode

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
            payload["topic_time_ranges"] = job.get("topic_time_ranges") or []
            payload["topic_time_mode"] = job.get("topic_time_mode")
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
            "topic_time_ranges": None,
            "topic_time_mode": None,
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
            requested_mode = None
            if isinstance(init_state, dict):
                requested_mode = init_state.get("topic_time_mode")
            topic_time_ranges, topic_time_mode = self._extract_topic_time_ranges(
                workflow_out=out if isinstance(out, dict) else {},
                transcript_json=init_state.get("transcript_json") if isinstance(init_state, dict) else {},
                mode=requested_mode,
            )
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
                topic_time_ranges=topic_time_ranges,
                topic_time_mode=topic_time_mode,
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
