import argparse

parser = argparse.ArgumentParser()


def stringToBoolean(s):
    return s.lower() in ("true", "1")


parser.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Running mode of program. Either test or train")

parser.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

parser.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

parser.add_argument("--num_epoch", type=int,
                       default=100,
                       help="Number of epochs to train")

parser.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

parser.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

parser.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

parser.add_argument("--resume", type=stringToBoolean,
                       default=True,
                       help="Whether to resume training from existing checkpoint")

parser.add_argument("--model_type", type=str,
                       default="svm",
                       choices=["cross_entropy", "svm"],
                       help="Type of data loss to be used")

parser.add_argument("--normalize", type=stringToBoolean,
                       default=True,
                       help="Whether to normalize with mean/std or not")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()