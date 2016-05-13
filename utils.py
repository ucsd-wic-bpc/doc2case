import sys
import os

def print_list(l, sep='\n\n'):
    print(sep.join(map(str, l)))

def check_dir(directory):
    if os.path.isfile(directory):
        print('Directory already exists. Abort.')
        sys.exit(1)
    elif os.path.isdir(directory):
        msg = 'Directory {} already exsits. Overwrite? [y/n]: '.format(directory)
        if not input(msg).startswith('y'):
            sys.exit(0)

def write_to_files(path, name, text):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    with open(path, 'w') as fout:
        fout.write(text)

def align(text, offset=0, mark='-'):
    left, right = zip(*[(l[:offset], l[offset:]) for l in text.split('\n')])
    lines = [(l, l.find(mark)) for l in right]
    _, imk = max(lines, key=lambda x: x[1])
    return '\n'.join(h + (l[:i] + ' ' * (imk - i) + l[i:] if i != -1 else l) for h, (l, i) in zip(left, lines))
