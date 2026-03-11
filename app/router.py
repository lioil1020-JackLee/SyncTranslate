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
    meeting_in_name = canonical_device_name(routes.meeting_in)
    microphone_in_name = canonical_device_name(routes.microphone_in)
    speaker_out_name = canonical_device_name(routes.speaker_out)
    meeting_out_name = canonical_device_name(routes.meeting_out)

    need_meeting_in = mode in ("meeting_to_local", "bidirectional")
    need_microphone_in = mode in ("local_to_meeting", "bidirectional")

    if need_microphone_in and not routes.microphone_in.strip():
        result.issues.append(RouteIssue("ERROR", "microphone_in", "microphone_in 必須設定"))
    if need_meeting_in and not routes.meeting_in.strip():
        result.issues.append(RouteIssue("ERROR", "meeting_in", "meeting_in 必須設定"))

    if not routes.speaker_out.strip():
        result.issues.append(RouteIssue("WARNING", "speaker_out", "speaker_out 尚未設定"))
    if not routes.meeting_out.strip():
        result.issues.append(RouteIssue("WARNING", "meeting_out", "meeting_out 尚未設定"))

    if meeting_in_name and meeting_in_name == meeting_out_name:
        result.issues.append(RouteIssue("ERROR", "meeting_in", "meeting_in 不能等於 meeting_out"))

    if meeting_in_name and meeting_in_name == speaker_out_name:
        result.issues.append(RouteIssue("ERROR", "meeting_in", "meeting_in 不能等於 speaker_out"))

    if meeting_out_name and meeting_out_name not in output_device_names:
        result.issues.append(RouteIssue("ERROR", "meeting_out", "meeting_out 必須是輸出裝置"))

    if meeting_in_name and meeting_in_name not in input_device_names:
        result.issues.append(RouteIssue("ERROR", "meeting_in", "meeting_in 必須是輸入裝置"))

    if microphone_in_name and microphone_in_name not in input_device_names:
        result.issues.append(RouteIssue("ERROR", "microphone_in", "microphone_in 必須是輸入裝置"))

    if speaker_out_name and speaker_out_name not in output_device_names:
        result.issues.append(RouteIssue("ERROR", "speaker_out", "speaker_out 必須是輸出裝置"))

    return result
