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

def write_to_files(self, path, name, text):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    with open(path, 'w') as fout:
        fout.write(text)