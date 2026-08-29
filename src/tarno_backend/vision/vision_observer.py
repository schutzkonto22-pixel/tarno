"""Vision-Observer für die autonome Trigger-Engine (Block 7, Phasen 67-68).

Same ProactiveObserver interface as TimeCalendarObserver/SystemObserver/
UserBehaviorObserver (tarno/core/proactive_engine.py) - a fourth observer,
no special-casing needed in ProactiveEngine itself. Permanently polls the
camera at low cost (motion gate only, Phase 62); only on detected motion
does it collect a short burst of candidate frames (Phase 63) and send them
to a vision-capable Mistral model for frame-selection + relevance scoring
(Phase 65/66). The returned score feeds the exact same TRACES threshold gate
as every other observer (Phase 68).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from tarno_backend.core.proactive_engine import ProactiveDraft
from tarno_backend.vision.camera_capture import CameraCapture
from tarno_backend.vision.motion_gate import MotionGate
from tarno_backend.vision.vision_provider import MistralVisionProvider

log = logging.getLogger(__name__)


class VisionObserver:
    """Phase 67: the fourth ProactiveEngine observer, backed by a webcam."""

    name = "vision"

    def __init__(
        self,
        camera: CameraCapture,
        vision_provider: MistralVisionProvider,
        motion_threshold: float = 25.0,
        candidate_frames: int = 5,
        downscale_max_edge: int = 640,
        jpeg_quality: int = 80,
        warmup_frames: int = 15,
        warmup_delay: float = 0.03,
    ) -> None:
        self._camera = camera
        self._vision_provider = vision_provider
        self._motion_gate = MotionGate(threshold=motion_threshold)
        self._candidate_frames = max(1, candidate_frames)
        self._downscale_max_edge = downscale_max_edge
        self._jpeg_quality = jpeg_quality
        self._warmup_frames = max(0, warmup_frames)
        self._warmup_delay = warmup_delay
        self._camera_started = False
        self._enabled = True

    @property
    def is_capturing(self) -> bool:
        """True only while the camera hardware is actually open - reflects
        genuine device activity 1:1 (Phase 69: "Hardware-LED-Respekt"), since
        the physical capture LED lights up exactly when this is True."""
        return self._camera_started and self._camera.is_active

    def set_enabled(self, enabled: bool) -> None:
        """Phase 69: recording opt-in/opt-out. Both directions take effect
        immediately rather than waiting for the next ProactiveEngine tick
        (which can be up to tick_seconds away, e.g. 10s) - a caller checking
        is_capturing right after calling this (e.g. the gRPC status
        broadcast) would otherwise see stale state. Disabling releases the
        camera hardware outright (not just "ignore its output"), so the
        physical capture LED goes off with it."""
        self._enabled = enabled
        if enabled and not self._camera_started:
            self._camera_started = self._camera.start()
        elif not enabled and self._camera_started:
            self._camera.stop()
            self._camera_started = False
            self._motion_gate.reset()

    def check(self) -> ProactiveDraft | None:
        if not self._enabled:
            return None

        if not self._vision_provider.available:
            return None  # no API key configured - nothing useful this observer can do

        if not self._camera_started:
            if not self._camera.start():
                return None  # no camera available - observer effectively no-ops
            self._camera_started = True

        frame = self._camera.read_frame()
        if frame is None:
            return None

        if not self._motion_gate.detect(frame):
            return None

        # Motion detected (Phase 62) - collect a short burst of candidate
        # frames (Phase 63) rather than judging on the single triggering frame.
        candidates = [frame]
        for _ in range(self._candidate_frames - 1):
            next_frame = self._camera.read_frame()
            if next_frame is not None:
                candidates.append(next_frame)

        assessment = self._vision_provider.assess_frames(
            candidates, max_edge=self._downscale_max_edge, jpeg_quality=self._jpeg_quality,
        )
        if assessment is None or not assessment.relevant:
            return None

        return ProactiveDraft(
            source=self.name,
            message=f"Ich habe etwas bemerkt: {assessment.reason}",
            score=assessment.score,
        )

    def capture_and_describe(self) -> tuple[str, bytes] | None:
        """On-demand single-shot description for direct user queries ("was
        siehst du gerade?") - distinct from check()'s passive, motion-gated
        background loop. If the camera wasn't already running, it is opened
        just long enough to grab one frame and closed again immediately
        afterwards (Phase 69 privacy contract: no lingering capture beyond
        what was explicitly requested). Returns None if unavailable, no
        camera, or the vision call itself fails.

        Returns a tuple of (description, jpeg_bytes) so the exact frame
        analysed by the model can be shown in the UI as an attachment.

        Found via live testing: a snapshot taken immediately after a cold
        camera open came back dark/washed-out even in a well-lit room (the
        continuously-running Windows camera preview looked fine at the same
        moment) - auto-exposure/auto-white-balance need a few frames to
        settle after opening. The check()/background-loop path never hits
        this because it keeps the camera open continuously once started, so
        the warm-up only needs to run here, for the cold-open on-demand path.

        Additional hardening: reject frozen/identical frames (screen capture
        or blocked camera) and log device label + frame metadata.
        """
        if not self._enabled or not self._vision_provider.available:
            return None

        opened_here = False
        if not self._camera_started:
            if not self._camera.start():
                log.warning("Kamera konnte nicht gestartet werden")
                return None
            self._camera_started = True
            opened_here = True

        try:
            if opened_here:
                for _ in range(self._warmup_frames):
                    self._camera.read_frame()
                    if self._warmup_delay:
                        time.sleep(self._warmup_delay)

            frame = self._read_living_frame(max_attempts=15)
            if frame is None:
                log.warning(
                    "Kamera %s liefert kein frisches Bild - vermutlich blockiert "
                    "oder eingefroren.",
                    getattr(self._camera, "device_label", "unbekannt"),
                )
                return None

            log.info(
                "Frisches Kamerabild erfasst: %s, Groesse=%dx%d, Zeitstempel=%.3f",
                getattr(self._camera, "device_label", "unbekannt"),
                frame.shape[1], frame.shape[0], time.time(),
            )
            result = self._vision_provider.describe_frame(
                frame,
                max_edge=self._downscale_max_edge,
                jpeg_quality=self._jpeg_quality,
            )
            if result is None:
                return None
            description, jpeg_bytes = result
            log.info(
                "Vision-Beschreibung erhalten: %d Bytes JPEG, %d Zeichen Beschreibung",
                len(jpeg_bytes), len(description),
            )
            return result
        finally:
            if opened_here:
                self._camera.stop()
                self._camera_started = False

    def _read_living_frame(self, max_attempts: int = 15) -> np.ndarray | None:
        """Read multiple frames and return the first one that is not
        identical to the previous frame. This protects against screen-
        capture sources and cameras that keep returning the same cached
        frame. Degenerate/black checks are already done by CameraCapture
        validation during start(); the warm-up phase above handles the
        auto-exposure settling."""
        prev_frame: np.ndarray | None = None
        for _ in range(max_attempts):
            frame = self._camera.read_frame()
            if frame is None:
                continue
            if prev_frame is not None and CameraCapture._is_frozen(prev_frame, frame):
                prev_frame = frame
                continue
            return frame
        return None

    def close(self) -> None:
        """Release the camera. Called at engine shutdown (Phase 69: the
        camera must not stay open/active any longer than the engine runs)."""
        if self._camera_started:
            self._camera.stop()
            self._camera_started = False
