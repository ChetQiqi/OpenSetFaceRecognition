"""Face Recognition System - Handles face detection, recognition, and management."""

def __getattr__(name):
    """Lazy loading of modules to avoid import errors due to missing dependencies."""
    if name == "FeatureDB":
        from .core.feature_db import FeatureDB
        return FeatureDB
    elif name == "build_runtime":
        from .core.operations import build_runtime
        return build_runtime
    elif name == "recognize_faces":
        from .core.operations import recognize_faces
        return recognize_faces
    elif name == "FaceDetector":
        from .core.detector import FaceDetector
        return FaceDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FeatureDB", "build_runtime", "recognize_faces", "FaceDetector"]
