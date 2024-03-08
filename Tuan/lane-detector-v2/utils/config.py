import yaml


class Config:
    def __init__(self, config_path: str) -> None:
        super().__init__()

        self.config = self.load_config(config_path)
        self.lane_fitting = self.config["lane_fitting"]
        self.lane_detector = self.config["lane_detector"]
        self.video_path = self.config["video_path"]

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)
