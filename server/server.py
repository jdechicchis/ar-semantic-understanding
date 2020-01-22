"""
Edge server.
"""

import argparse
import json
from flask import Flask

APP = Flask(__name__)

@APP.route('/')
def hello_world():
    """
    Just return hellow world to test the server.
    """
    return "Hello, World!"

@APP.route('/mask')
def mask():
    """
    Return the segmentation mask information.
    """
    locations_file = open("locations.json", "r")
    return json.load(locations_file)

def main():
    """
    Parse arguments and start server.
    """
    parser = argparse.ArgumentParser(description="Run edge server.")
    parser.add_argument("--public", action="store_true", help="Make server publically visible.")
    args = parser.parse_args()

    if args.public:
        print("\n########## RUNNING SERVER PUBLICLY!!!! ##########\n")
        APP.run(host="0.0.0.0", port=5005)
    else:
        APP.run()

if __name__ == '__main__':
    main()
