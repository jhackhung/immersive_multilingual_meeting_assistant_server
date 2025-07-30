from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TranslateRequest(_message.Message):
    __slots__ = ("text_to_translate", "source_language", "target_language")
    TEXT_TO_TRANSLATE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    TARGET_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    text_to_translate: str
    source_language: str
    target_language: str
    def __init__(self, text_to_translate: _Optional[str] = ..., source_language: _Optional[str] = ..., target_language: _Optional[str] = ...) -> None: ...

class TranslateResponse(_message.Message):
    __slots__ = ("translated_text",)
    TRANSLATED_TEXT_FIELD_NUMBER: _ClassVar[int]
    translated_text: str
    def __init__(self, translated_text: _Optional[str] = ...) -> None: ...

class Wav2LipRequest(_message.Message):
    __slots__ = ("audio_data", "image_data")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    image_data: bytes
    def __init__(self, audio_data: _Optional[bytes] = ..., image_data: _Optional[bytes] = ...) -> None: ...

class Wav2LipResponse(_message.Message):
    __slots__ = ("video_data",)
    VIDEO_DATA_FIELD_NUMBER: _ClassVar[int]
    video_data: bytes
    def __init__(self, video_data: _Optional[bytes] = ...) -> None: ...

class TtsRequest(_message.Message):
    __slots__ = ("text_to_speak", "reference_audio", "language")
    TEXT_TO_SPEAK_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_AUDIO_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    text_to_speak: str
    reference_audio: bytes
    language: str
    def __init__(self, text_to_speak: _Optional[str] = ..., reference_audio: _Optional[bytes] = ..., language: _Optional[str] = ...) -> None: ...

class TtsResponse(_message.Message):
    __slots__ = ("generated_audio",)
    GENERATED_AUDIO_FIELD_NUMBER: _ClassVar[int]
    generated_audio: bytes
    def __init__(self, generated_audio: _Optional[bytes] = ...) -> None: ...

class SpeakerAnnoteRequest(_message.Message):
    __slots__ = ("audio_data",)
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    def __init__(self, audio_data: _Optional[bytes] = ...) -> None: ...

class DiarizationSegment(_message.Message):
    __slots__ = ("speaker", "start_time", "end_time")
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    speaker: str
    start_time: float
    end_time: float
    def __init__(self, speaker: _Optional[str] = ..., start_time: _Optional[float] = ..., end_time: _Optional[float] = ...) -> None: ...

class SpeakerTimeline(_message.Message):
    __slots__ = ("speaker", "segments")
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    speaker: str
    segments: _containers.RepeatedCompositeFieldContainer[DiarizationSegment]
    def __init__(self, speaker: _Optional[str] = ..., segments: _Optional[_Iterable[_Union[DiarizationSegment, _Mapping]]] = ...) -> None: ...

class SpeakerAnnoteResponse(_message.Message):
    __slots__ = ("all_segments", "speaker_timelines")
    ALL_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_TIMELINES_FIELD_NUMBER: _ClassVar[int]
    all_segments: _containers.RepeatedCompositeFieldContainer[DiarizationSegment]
    speaker_timelines: _containers.RepeatedCompositeFieldContainer[SpeakerTimeline]
    def __init__(self, all_segments: _Optional[_Iterable[_Union[DiarizationSegment, _Mapping]]] = ..., speaker_timelines: _Optional[_Iterable[_Union[SpeakerTimeline, _Mapping]]] = ...) -> None: ...
