#!/usr/bin/env python3
import logging
from app.runner import create_pipeline


def main():
    logging.basicConfig(level=logging.INFO)
    pipeline = create_pipeline()
    pipeline.run_forever()


if __name__ == '__main__':
    main()
