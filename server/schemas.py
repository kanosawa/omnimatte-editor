from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


ModelStateLiteral = Literal["loading", "ready", "failed"]


class _CamelModel(BaseModel):
    """JSON では camelCase、Python 属性は snake_case の応答モデル基底。"""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_state: ModelStateLiteral
    session_active: bool


class VideoMeta(_CamelModel):
    width: int
    height: int
    fps: float
    num_frames: int
    duration_sec: float


class StartSessionResponse(_CamelModel):
    video_meta: VideoMeta


class SegmentRequest(BaseModel):
    frame_idx: int = Field(ge=0)
    bbox: list[float] = Field(min_length=4, max_length=4)

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: list[float]) -> list[float]:
        x1, y1, x2, y2 = v
        if not (x2 > x1 and y2 > y1):
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
        return v
