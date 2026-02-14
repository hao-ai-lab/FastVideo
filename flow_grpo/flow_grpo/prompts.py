from importlib import resources
import os
import functools
import random
# import inflect

# IE = inflect.engine()
IE=None
ASSETS_PATH = resources.files("flow_grpo.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `flow_grpo/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or flow_grpo.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")

def general_ocr():
    return from_file("general_ocr_train.txt")

def simple_ocr_animals():
    animals = _load_lines("simple_ocr_animals.txt")
    # random_number = random.randint(100, 999)
    # random_number = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    num=random.randint(1, 9)
    random_number = ''.join([str(6) for _ in range(num)])
    return f'A {random.choice(animals)} holding a sign that says "{random_number}"', {}

def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata
