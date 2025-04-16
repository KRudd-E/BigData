from controller import PipelineController
from config import config


if __name__ == "__main__":
    print("Starting pipeline via controller.py")
    config = config
    controller = PipelineController(config)
    controller.run()