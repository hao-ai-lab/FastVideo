from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)
import os


def main():
    print(os.environ["RANK"])
    print(os.environ["WORLD_SIZE"])
    print(os.environ["LOCAL_RANK"])
    print(os.environ["MASTER_ADDR"])
    print(os.environ["MASTER_PORT"])
    print(os.environ["RANK"])
    print(os.environ["WORLD_SIZE"])
    print(os.environ["LOCAL_RANK"])
    print(os.environ["MASTER_ADDR"])


if __name__ == "__main__":
    main()
