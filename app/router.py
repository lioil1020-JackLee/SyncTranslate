from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import AudioRouteConfig


@dataclass(slots=True)
class RouteIssue:
    level: str
    field: str
    message: str


@dataclass(slots=True)
class RouteCheckResult:
    issues: list[RouteIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "ERROR" for issue in self.issues)

    @property
    def summary(self) -> str:
        if not self.issues:
            return "OK"
        if self.ok:
            return "Warning"
        return "Error"


def check_routes(
    routes: AudioRouteConfig,
    input_device_names: set[str],
    output_device_names: set[str],
    mode: str = "bidirectional",
) -> RouteCheckResult:
    result = RouteCheckResult()

    need_remote_in = mode in ("remote_only", "bidirectional")
    need_local_mic_in = mode in ("local_only", "bidirectional")

    if need_local_mic_in and not routes.local_mic_in.strip():
        result.issues.append(RouteIssue("ERROR", "local_mic_in", "local_mic_in 不能空白"))
    if need_remote_in and not routes.remote_in.strip():
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 不能空白"))

    if not routes.local_tts_out.strip():
        result.issues.append(RouteIssue("WARNING", "local_tts_out", "local_tts_out 尚未設定"))
    if not routes.meeting_tts_out.strip():
        result.issues.append(RouteIssue("WARNING", "meeting_tts_out", "meeting_tts_out 尚未設定"))

    if routes.remote_in and routes.remote_in == routes.meeting_tts_out:
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 不能等於 meeting_tts_out"))

    if routes.remote_in and routes.remote_in == routes.local_tts_out:
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 不能等於 local_tts_out"))

    if routes.meeting_tts_out and routes.meeting_tts_out not in output_device_names:
        result.issues.append(RouteIssue("ERROR", "meeting_tts_out", "meeting_tts_out 必須是輸出裝置"))

    if routes.remote_in and routes.remote_in not in input_device_names:
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 必須是輸入裝置"))

    if routes.local_mic_in and routes.local_mic_in not in input_device_names:
        result.issues.append(RouteIssue("ERROR", "local_mic_in", "local_mic_in 必須是輸入裝置"))

    if routes.local_tts_out and routes.local_tts_out not in output_device_names:
        result.issues.append(RouteIssue("ERROR", "local_tts_out", "local_tts_out 必須是輸出裝置"))

    return result