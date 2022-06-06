import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pch_size')
    parser.add_argument('--radius')
    parser.add_argument('--noise_type')
    
    return parser.parse_args()