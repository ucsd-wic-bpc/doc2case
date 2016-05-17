import sys
import os
import shutil
import re

def print_list(l, sep='\n\n'):
    print(sep.join(map(str, l)))

def check_dir(directory):
    if os.path.isfile(directory):
        print('Directory already exists. Abort.')
        sys.exit(1)
    elif os.path.isdir(directory):
        msg = 'Directory {} already exsits. Overwrite? [y/n]: '.format(directory)
        if input(msg).startswith('y'):
            shutil.rmtree(directory)
        else:
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

def to_readable(text):
    # Leave at most 2 consecutive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Leave only one line break after colon
    text = re.sub(r'(?<=:)(\s*\n){2,}', '\n', text)
    # No spaces at the beginning of a line
    text = re.sub(r'(?<=\n)\s+', '', text, re.MULTILINE)
    # No spaces at both ends of the text
    text = text.strip('\n\t ')
    return text

TWO_SPCS = ' ' * 2
def indent(text, spaces=TWO_SPCS):
    return '\n'.join(spaces + l for l in text.split('\n'))

def toCamelCase(s):
    return ''.join(w.capitalize() if i != 0 else w for i, w in enumerate(s.lower().split()))
