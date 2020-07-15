import argparse
import yaml
from utils import *

parser = argparse.ArgumentParser(description='test functions')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    default='configs/function_test.yaml')
args = parser.parse_args()

with open(args.filename, 'r') as file:
    try :
        config = yaml.load(file, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        print(e)


test_function = {
    'genDatasetCelebA' : genDatasetCelebA_test
}

if __name__ == "__main__":
    for func_name in test_function:
        test_function[func_name](**config[func_name])

    