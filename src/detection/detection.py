from typing import Any, List
from inference_sdk import InferenceHTTPClient
import numpy as np
from numpy._typing import NDArray
from src.common import utils


# TODO: do not hardcode api key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="fxmOxhDyezdXl0VWZrCi"
)


def sort_predictions_by_x(data):
    """
    Sorts the predictions in a dictionary by the 'x' value in ascending order.

    Parameters:
        data (dict): The input dictionary containing predictions.

    Returns:
        list: The sorted list of predictions.
    """
    if "predictions" not in data:
        raise ValueError("The input dictionary must contain a 'predictions' key.")

    # Sort the predictions by the 'x' value and extract only the 'class' strings
    sorted_classes = [
        p["class"].lower() for p in sorted(data["predictions"], key=lambda p: p["x"])
    ]
    return sorted_classes


@utils.run_if_enabled
def detect(image: NDArray[Any]) -> List[str]:
    result = CLIENT.infer(image, model_id="maplestory/9")
    classes = sort_predictions_by_x(result)
    return classes


# NOTE: used to test, captures monitor and runs detection on it
if __name__ == "__main__":
    from src.common import config
    import mss

    config.enabled = True
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    with mss.mss() as sct:
        frame = np.array(sct.grab(monitor))
        result = CLIENT.infer(frame, model_id="maplestory/9")
        print(sort_predictions_by_x(result))
