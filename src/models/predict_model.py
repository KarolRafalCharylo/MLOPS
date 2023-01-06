import logging
import os
import sys

import click
import torch
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeModel

sys.path.insert(1, os.path.join(sys.path[0], ".."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("data_filepath", type=click.Path())
def main(model_filepath, data_filepath):
    print("Evaluating until hitting the ceiling")
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_filepath))

    test_data = torch.load(data_filepath)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    acc = []
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            acc.append(accuracy.item() * 100)
        print(f"Final Accuracy: {sum(acc)/len(acc)}%")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
