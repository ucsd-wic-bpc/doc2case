#! /usr/bin/env python3
import argparse
import os
import datetime

from doc2case import DocParser, TypeParser
from utils import print_list, check_dir

OUT_DIR = 'templates'


def toCamelCase(s):
    return ''.join(w.capitalize() if i != 0 else w for i, w in enumerate(s.lower().split()))


class TemplateGenerator:
    BPC_BANNER = (
        '# ---------------------- DO NOT EDIT BELOW THIS LINE ------------------------- #\n'
        '# _       __ ____ ______       ____   ____   ______\n'
        '#| |     / //  _// ____/      / __ ) / __ \ / ____/\n'
        '#| | /| / / / / / /   ______ / __  |/ /_/ // /\n'
        '#| |/ |/ /_/ / / /___/_____// /_/ // ____// /___\n'
        '#|__/|__//___/ \____/      /_____//_/     \____/\n'
        '#\n'
        '#{quarter} {year}, University of California, San Diego.\n'
    )
    SP = 'Spring'
    FA = 'Fall'
    WI = 'Winter'
    QUARTER = {
        4: SP,
        5: SP,
        6: SP,
        9: FA,
        10: FA,
        11: FA,
        12: FA,
        1: WI,
        2: WI,
        3: WI,
    }

    def __init__(self, examples, suffix):
        self.examples = examples
        self.suffix = suffix

    def generate(self):
        for (pnum, pname), _, params, ret in self.examples:
            params = self._get_valid_types(params)
            [ret] = self._get_valid_types(ret)
            self.write(pnum, self._generate(pname, params, ret))

    def write(self, pnum, text):
        name = 'Problem{}.{}'.format(pnum, self.suffix)
        path = os.path.join(OUT_DIR, name)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(path, 'w') as fout:
            fout.write(text)

    @classmethod
    def _generate(cls, pname, params, ret):
        '''
        :param str, [(str, str, str)], (str, str):
            problem_name, [(param_type, param_name, param_desc)], (return_name, return_desc)
        '''
        raise NotImplementedError

    @classmethod
    def make_bpc_banner(cls):
        today = datetime.datetime.today()
        return cls.BPC_BANNER.format(quarter=cls.QUARTER[today.month], year=today.year)


class PythonGenerator(TemplateGenerator):
    HEADER = (
        'import json\n'
        'import sys\n'
    )

    SIGNATURE = (
        '"""\n'
        'Parameters: {params}\n'
        'Return: {ret}\n'
        '"""\n'
    )

    METHOD = (
        'def {name}({params}):\n'
        '    # TODO\n'
        '    return {ret}\n'
    )

    MAIN = (
        'def main():\n'
        '    argumentData = json.loads(raw_input())\n'
        '    print(str({name}(*argumentData)))\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    sys.exit(main())\n'
    )

    PADDING = len('Parameters: ')

    def __init__(self, examples):
        super().__init__(examples, 'py')

    @classmethod
    def _get_valid_types(cls, type_tuples):
        new_tuples = []
        for stype, name, desc in type_tuples:
            new_tuples.append((stype, name, desc.capitalize()))
        return new_tuples

    @classmethod
    def _join_params(cls, format, params, offset):
        return '\n'.join(cls.PADDING * ' ' + format(*p) for p in params)[offset:]

    @classmethod
    def _generate(cls, pname, params, ret):
        new_params = []
        for stype, name, desc in params:
            stype = TypeParser.get_valid_stype(stype)
            new_params.append((name, stype, desc))
        ret_stype, _, ret_desc = ret
        ret_stype = TypeParser.get_valid_stype(ret_stype)
        method_name = toCamelCase(pname)
        return ''.join([
            cls.HEADER,
            '\n',
            cls.SIGNATURE.format(params=cls._join_params('{}: {} - {}'.format, new_params, cls.PADDING),
                                 ret=cls._join_params('{} - {}'.format, [(ret_stype, ret_desc)], len('Return: '))),
            cls.METHOD.format(name=method_name, params=', '.join(
                n for (_, n, _) in params), ret=repr(TypeParser.get_type(ret[0])())),
            '\n',
            cls.make_bpc_banner(),
            '\n',
            cls.MAIN.format(name=method_name)
        ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='the file to transcribe')
    args = parser.parse_args()
    return args


def main():
    # Get arguments
    args = parse_args()
    filein = args.filename

    # Parse the document
    with open(filein) as fin:
        text = fin.read()
    examples = DocParser(text).findall()

    # Check output directory
    check_dir(OUT_DIR)

    # Generate templates
    PythonGenerator(examples).generate()


if __name__ == '__main__':
    main()
