import argparse

from deepfish.bot import FishingBot


def parse_args() -> argparse.Namespace:
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-k",
        "--fishing_skill_key",
        help="Key, assigned to fishing skill",
        type=str,
        required=True,
    )
    arguments_parser.add_argument(
        "--suppress_logging",
        help="Pass this argument to suppress logs.",
        action='store_true'
    )
    args = arguments_parser.parse_args()
    return args


def main():
    args = parse_args()
    bot_instance = FishingBot(
        fishing_skill_key=args.fishing_skill_key,
        need_logging=not args.suppress_logging
    )
    bot_instance.run()


if __name__ == "__main__":
    main()
