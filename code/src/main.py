from controller_loop import PipelineController_Loop
from config import config


if __name__ == "__main__":
    print("Starting pipeline via controller.py")
    config = config
    controller = PipelineController_Loop(config)
    controller.run()