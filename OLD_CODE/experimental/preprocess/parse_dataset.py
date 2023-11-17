import os
import parse
import re
from collections import defaultdict
import pprint


def get_subfamily(dirname):
    r = parse.parse('{}- Subfamily', dirname)
    if r is not None:
        return r[0].strip().lower()
    else:
        raise RuntimeError('unable to get subfamily name from %r' % dirname)


def get_tribe(dirname):
    r = parse.parse('{}- Tribe', dirname)
    if r is not None:
        return r[0].strip().lower()
    else:
        return dirname.strip().lower()


def get_genus(filename):
    s = filename.replace('Copy of ', '')
    idx_first_nonletter = re.search('[^a-zA-Z]', s).start()
    return s[:idx_first_nonletter].lower()


def get_image_stacks(path):
    res = {}
    for d in os.listdir(path):
        dpath = os.path.join(path, d)
        if os.path.isdir(dpath) and 'stack' in d.lower():
            res['dir'] = d
            files_per_genus = defaultdict(list)
            files = os.listdir(dpath)
            for fn in files:
                if fn.endswith('.tif'):
                    genus = get_genus(fn)
                    files_per_genus[genus].append(fn)
            res['files'] = dict(files_per_genus)
    return res


def build_tree(dirname):
    tree = {}
    for subfamily_dir in os.listdir(dirname):
        subfamily_path = os.path.join(dirname, subfamily_dir)
        if os.path.isdir(subfamily_path):
            subfamily = get_subfamily(subfamily_dir)
            tree[subfamily] = {'dir': subfamily_dir}
            tribes = {}
            for tribe_dir in os.listdir(subfamily_path):
                tribe_path = os.path.join(subfamily_path, tribe_dir)
                if os.path.isdir(tribe_path):
                    tribe = get_tribe(tribe_dir)
                    tribes[tribe] = {'dir': tribe_dir}

                    genus = {'image_stack': get_image_stacks(tribe_path)}
                    tribes[tribe]['genus'] = genus

            tree[subfamily]['tribes'] = tribes

    return tree


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(build_tree('../../data/official_data'))
