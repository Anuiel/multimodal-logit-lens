import click
from PIL import Image
from matplotlib import pyplot as plt

from logit_features import LogitsData, load_logits_json


@click.command()
@click.option("--logits-path", default="logit.json", help="Path to the logits json file.")
@click.option("--output-path", default=".", help="Path to save the output images.")
def main(logits_path: str, output_path: str):
    logits_data = load_logits_json(logits_path)

    samples = LogitsData(logits_data.entries[:6])
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    for ax, sample in zip(axs.ravel(), samples.entries):
        image_path = sample.image_path
        image = Image.open(image_path)
        image = image.resize((512, 512), resample=Image.BILINEAR)
        question = sample.question
        answer = sample.answer
        output_number = sample.output_number

        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"Question: {question}\nAnswer: {answer}\nModel output: {output_number}")

    plt.tight_layout()
    plt.savefig(f"{output_path}/example_showcase.png")
    plt.show()


if __name__ == "__main__":
    main()
