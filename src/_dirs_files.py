

import os

def make_dirs(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)

    except OSError:
        print("Error: failed to make %s", dir)


