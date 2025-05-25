import argparse
from Bill import Bill
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def parse_args():
    parser = argparse.ArgumentParser(description="I'm a snake")
    parser.add_argument("-play", action="store_true",
                        help="Play the game ignores other options")
    parser.add_argument("-logtrain", action="store_true",
                        help="train from logfile may ignore other options")
    parser.add_argument("-exhibit", action="store_true",
                        help="Shows model in autorun may ignore other options")
    parser.add_argument("-save", type=str,
                        default="models/latest.keras", help="Save model route")
    parser.add_argument("-load", type=str,
                        default="models/latest.keras", help="Load model route")
    parser.add_argument("-logfile", type=str,
                        default="logs/snakelog.jsonl", help="Logfile route")
    parser.add_argument("-sessions", type=int,
                        help="number of training sessions", default=10)
    parser.add_argument("-debug", action="store_true",
                        help="Stop at each play")

    return parser.parse_args()


def main():

    args = parse_args()

    Bill(args).run()


if __name__ == "__main__":
    main()
