SOURCE_DATASET = "UkrainianCatholicUniversity/rukopys"
DEFAULT_DETECTOR_MODEL = "yolo11n.pt"
DEFAULT_VLM_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
COLAB_T4_VLM_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

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
