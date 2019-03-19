
from config import get_config, print_usage


def train(config):

    print("in train config")

def test(config):
    print("in test config")

def main(config):
    """The main function."""

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


# Parse configuration
config, unparsed = get_config()
# print message of usage if something is wrong
if len(unparsed) > 0:
    print_usage()
    exit(1)

main(config)
