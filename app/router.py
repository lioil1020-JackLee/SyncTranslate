from __future__ import annotations

from dataclasses import dataclass, field

from app.audio_device_selection import canonical_device_name
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
    remote_in_name = canonical_device_name(routes.remote_in)
    local_mic_in_name = canonical_device_name(routes.local_mic_in)
    local_tts_out_name = canonical_device_name(routes.local_tts_out)
    meeting_tts_out_name = canonical_device_name(routes.meeting_tts_out)

    need_remote_in = mode in ("remote_only", "bidirectional")
    need_local_mic_in = mode in ("local_only", "bidirectional")

    if need_local_mic_in and not routes.local_mic_in.strip():
        result.issues.append(RouteIssue("ERROR", "local_mic_in", "local_mic_in 必須設定"))
    if need_remote_in and not routes.remote_in.strip():
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 必須設定"))

    if not routes.local_tts_out.strip():
        result.issues.append(RouteIssue("WARNING", "local_tts_out", "local_tts_out 尚未設定"))
    if not routes.meeting_tts_out.strip():
        result.issues.append(RouteIssue("WARNING", "meeting_tts_out", "meeting_tts_out 尚未設定"))

    if remote_in_name and remote_in_name == meeting_tts_out_name:
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 不能等於 meeting_tts_out"))

    if remote_in_name and remote_in_name == local_tts_out_name:
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 不能等於 local_tts_out"))

    if meeting_tts_out_name and meeting_tts_out_name not in output_device_names:
        result.issues.append(RouteIssue("ERROR", "meeting_tts_out", "meeting_tts_out 必須是輸出裝置"))

    if remote_in_name and remote_in_name not in input_device_names:
        result.issues.append(RouteIssue("ERROR", "remote_in", "remote_in 必須是輸入裝置"))

    if local_mic_in_name and local_mic_in_name not in input_device_names:
        result.issues.append(RouteIssue("ERROR", "local_mic_in", "local_mic_in 必須是輸入裝置"))

    if local_tts_out_name and local_tts_out_name not in output_device_names:
        result.issues.append(RouteIssue("ERROR", "local_tts_out", "local_tts_out 必須是輸出裝置"))

    return result
