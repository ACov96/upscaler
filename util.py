import sys

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
