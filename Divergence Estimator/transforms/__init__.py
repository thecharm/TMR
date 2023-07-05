from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
}



