import nussl
import sys

def peak_wrapper(array, n):
    nussl.find_peak_indices(array, n)


if __name__ == '__main__':
    args = sys.argv[1:]
    array = args[0]
    n = args[1]