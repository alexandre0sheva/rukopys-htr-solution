SOURCE_DATASET = "UkrainianCatholicUniversity/rukopys"
DEFAULT_HF_NAMESPACE = "your-hf-username-or-org"
DEFAULT_CURATED_DATASET = f"{DEFAULT_HF_NAMESPACE}/rukopys-curated-mvp"
DEFAULT_DETECTOR_MODEL = "yolo11n.pt"
DEFAULT_VLM_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

DEFAULT_DETECTOR_CONFIDENCE = 0.25
DEFAULT_DETECTOR_IOU = 0.5
DEFAULT_REGION_MAX_NEW_TOKENS = 192
DEFAULT_PAGE_MAX_NEW_TOKENS = 2048
DEFAULT_INFERENCE_MAX_PIXELS = 262144
READING_ORDER_ROW_BAND = 40

REGION_TYPES = [
    "handwritten",
    "printed",
    "formula",
    "table",
    "annotation",
    "image",
    "graph",
]

TRANSCRIBED_TYPES = {"handwritten", "printed", "formula", "table", "annotation"}
EMPTY_TEXT_TYPES = {"image", "graph"}

QUALITY_WEIGHTS = {
    "annotator": 1.0,
    "volunteer": 0.75,
    "auto": 0.35,
}
