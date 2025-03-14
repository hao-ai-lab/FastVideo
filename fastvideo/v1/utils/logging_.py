import os
import pdb
import sys


def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)


# ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
