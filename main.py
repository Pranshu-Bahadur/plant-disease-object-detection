import torch
import argparse
from experiment import Experiment
from torch.utils.data import DataLoader as Loader
from torchvision import transforms as transforms
import PIL
from util import ImageFilelist


def _model_config(args):
    config = {
        "model_name": args.model_name,
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "optimizer_name": args.optimizer,
        "criterion_name": args.loss,
        "scheduler_name": args.scheduler,
        "checkpoint": args.checkpoint if args.checkpoint else "",
        "num_classes": int(args.num_classes),
        "curr_epoch": int(args.curr_epoch) if args.curr_epoch else 0,
        "resolution": int(args.resolution),
        "epochs": int(args.epochs) if args.epochs else 0,
        "train": True if args.train else False,
        "pretrained": True if args.pretrained else False,
        "save_interval": int(args.save_interval),
        "library": args.library,
        "save_directory": args.save_directory,
        "list": args.list
    }
    return config



def _tbx11k_output(classifier, dataset: Loader):
    classifier.model.eval()
    with torch.no_grad():
        with open('outputs/{}-submission.txt'.format(classifier.name), "w+") as output:
            result = ""
            for _, data in enumerate(dataset):
                x = data[0]
                #if classifier.cuda:
                x = x.cuda()
                outputs = classifier.model(x)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                for _, prob in enumerate(probs.detach().cpu().numpy()):
                    print(' '.join(map(str, prob)))
                    result += ' '.join(map(str, prob))
                    result += "\n"
            output.write(result)
            output.close()

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="Pick a model name")
    parser.add_argument("--dataset_directory", "-d", help="Set dataset directory path")
    parser.add_argument("--resolution", "-r", help="Set image resolution")
    parser.add_argument("--batch_size", "-b", help="Set batch size")
    parser.add_argument("--learning_rate", "-l", help="set initial learning rate")
    parser.add_argument("--checkpoint", "-c", help="Specify path for model to be loaded")
    parser.add_argument("--num_classes", "-n", help="set num classes")
    parser.add_argument("--curr_epoch", "-e", help="Set number of epochs already trained")
    parser.add_argument("--epochs", "-f", help="Train for these many more epochs")
    parser.add_argument("--optimizer", help="Choose an optimizer")
    parser.add_argument("--scheduler", help="Choose a scheduler")
    parser.add_argument("--loss", help="Choose a loss criterion")
    parser.add_argument("--train", help="Set this model to train mode", action="store_true")
    parser.add_argument("--library")
    parser.add_argument("--list")
    parser.add_argument("--save_directory", "-s")
    parser.add_argument("--save_interval")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--opMode", action="store_true")
    parser.add_argument("--list")

    args = parser.parse_args()
    config = _model_config(args)
    experiment = Experiment(config)
    if args.opMode:
        transformations = [
            transforms.Resize([int(args.resolution), int(args.resolution)], PIL.Image.ANTIALIAS),
            transforms.ToTensor(),
        ]
        transformations = transforms.Compose(transformations)
        dataset = ImageFilelist(root=args.dataset_directory, flist=args.list, transform=transformations)#TBX11K(txt_path=args.list, img_dir=args.dataset, transform=transformations)
        testLoader = Loader(dataset, batch_size=experiment.classifier.bs, num_workers=4, shuffle=False)
        _tbx11k_output(experiment.classifier, testLoader)
    elif args.train:
        experiment._run(args.dataset_directory, config)
    else:
        dataset = experiment._preprocessing(args.dataset_directory, config["resolution"], False)
        loader = Loader(dataset, experiment.classifier.bs, shuffle=False, num_workers=4)
        experiment.classifier._test(loader)

