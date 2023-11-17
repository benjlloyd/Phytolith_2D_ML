import os
import glob

field_map = {
    'subfamily': 0, 'tribe': 1, 'genus': 2, 'species': 3
}


def parse_filename(fn):
    bn = os.path.basename(fn)
    toks = [t.strip().lower() for t in bn.split('_')]

    rpidx = fn.rfind(')')
    lpidx = fn.rfind('(')

    if rpidx == -1 or lpidx == -1:
        idx = 0
    else:
        idx = int(fn[lpidx+1:rpidx])

    return tuple(toks[:4]) + (idx,)


if __name__ == '__main__':
    fns = glob.glob('../../data/new_dataset/original/*.tif')
    for fn in fns:
        bn = os.path.basename(fn)
        print(parse_filename(bn))
