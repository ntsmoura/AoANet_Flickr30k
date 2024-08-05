import argparse
import json
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_json",
    type=str,
    default="data/coco.json",
    help="path to the json file containing additional info and vocab",
)

parser.add_argument(
    "--image_id",
    type=int,
    default="0",
    help="Return information about the dataset image id",
)

parser.add_argument("--function", type=str, default="find_image", help="The function you wanna use...")

args = parser.parse_args()


def get_image_info(input_json, image_id):
    with open(input_json, "rb") as file:
        json_file = json.load(file)

    images = json_file["images"]
    try:
        image = images[image_id]
        filename = f"Filename: {image['filename']}"
        raw_sentences = "\n".join([sentence["raw"] for sentence in image["sentences"]])
        sentences = f"Sentences:\n{raw_sentences}"
        split = f"Split: {image['split']}"
        sep = (
            "\n---------------------------------------------------------------------\n"
        )
        print(filename, sentences, split, sep=sep)
    except IndexError:
        print("Image not found")


def select_random_test_split(input_json):
    with open(input_json, "rb") as file:
        json_file = json.load(file)

    test_images = []
    images = json_file["images"]
    for image in images:
        if image["split"] == "test":
            test_images.append(image)

    selected_images = random.sample(test_images, 30)
    for image in selected_images:
        print(f"Id:{image['imgid']}; Filename:{image['filename']};", end="\n")


function = args.function
match function:
    case "find_image":
        get_image_info(args.input_json, args.image_id)
    case "select_random_test":
        select_random_test_split(args.input_json)
    case _:
        raise NotImplementedError
