#! /usr/bin/env python3
import argparse
import os
import re
import datetime

from doc2case import DocParser, TypeParser
from utils import print_list, check_dir, write_to_files, align

OUT_DIR = 'templates'

TWO_SPCS = ' ' * 2


def indent(text, spaces=TWO_SPCS):
    return '\n'.join(spaces + l for l in text.split('\n'))


def toCamelCase(s):
    return ''.join(w.capitalize() if i != 0 else w for i, w in enumerate(s.lower().split()))


class TemplateGenerator:
    BPC_BANNER = ''

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
    del SP, FA, WI

    SIGNATURE = ''

    HEADER = ''

    SIGN_PADDING = ''
    PARAM_TEMPLATE = ''

    LIB = ''

    SUFFIX = 'txt'

    def __init__(self, examples):
        self.examples = examples

    def generate(self):
        def validate(type_tuples):
            new_tuples = []
            for stype, name, desc in type_tuples:
                new_tuples.append((stype, name, desc.capitalize()))
            return new_tuples

        for (pnum, pname), _, params, ret in self.examples:
            pname = toCamelCase(pname)
            params = validate(params)
            [ret] = validate(ret)
            code = '\n'.join(b for b in self._generate(
                pnum, pname, params, ret) if not b.isspace())
            filename = 'Problem{}.{}'.format(pnum, self.SUFFIX)
            write_to_files(OUT_DIR, filename, code)

    @classmethod
    def _generate(cls, pnum, pname, params, ret):
        '''
        :param str, [(str, str, str)], (str, str):
            problem_name, [(param_type, param_name, param_desc)], (return_name, return_desc)
        :return [str]:
        '''
        return [
            cls._gen_header(),
            cls._gen_sign(params, ret) + cls._gen_method(pname, params, ret),
            cls._gen_banner(),
            cls._gen_lib(),
            cls._gen_main(pname, params, ret)
        ]

    @classmethod
    def _gen_method(cls, pname, params, ret):
        raise NotImplementedError

    @classmethod
    def _gen_main(cls, pname, params, ret):
        raise NotImplementedError

    @classmethod
    def _gen_header(cls):
        return cls.HEADER

    @classmethod
    def _gen_sign(cls, params, ret):
        def join(format, params, offset, sep='\n'):
            return sep.join(cls.SIGN_PADDING + format(*p) for p in params)[offset:]

        ret_stype, _, ret_desc = ret
        ret = [(ret_stype, ret_desc)]
        ret_offset = len(cls.SIGN_PADDING) - len('Parameters') + len('Return')
        return align(cls.SIGNATURE.format(
            params=join(cls.PARAM_TEMPLATE.format, params, len(cls.SIGN_PADDING)),
            ret=join('{0} - {1}'.format, ret, ret_offset)))

    @classmethod
    def _gen_lib(cls):
        return cls.LIB

    @classmethod
    def _gen_banner(cls):
        today = datetime.datetime.today()
        return cls.BPC_BANNER.format(quarter=cls.QUARTER[today.month], year=today.year)

    @classmethod
    def _is_header(cls, line):
        raise NotImplementedError


class PythonGenerator(TemplateGenerator):
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

    SIGN_PADDING = len('Parameters: ') * ' '
    PARAM_TEMPLATE = '{1}: {0} - {2}'

    SUFFIX = 'py'

    @classmethod
    def _gen_sign(cls, params, ret):
        new_params = []
        for stype, name, desc in params:
            stype = TypeParser.get_python_stype(stype)
            new_params.append((stype, name, desc))
        ret_stype, _, ret_desc = ret
        ret_stype = TypeParser.get_python_stype(ret_stype)
        return super()._gen_sign(new_params, (ret_stype, _, ret_desc))

    @classmethod
    def _gen_method(cls, pname, params, ret):
        return cls.METHOD.format(
            name=pname,
            params=', '.join(n for (_, n, _) in params),
            ret=repr(TypeParser.get_type(ret[0])()))

    @classmethod
    def _gen_main(cls, pname, params, ret):
        return cls.MAIN.format(name=pname)


class CppGenerator(TemplateGenerator):
    BPC_BANNER = (
        '/*  ---------------------- DO NOT EDIT BELOW THIS LINE --------------------\n'
        '   _       __ ____ ______       ____   ____   ______\n'
        '  | |     / //  _// ____/      / __ ) / __ \ / ____/\n'
        '  | | /| / / / / / /   ______ / __  |/ /_/ // /\n'
        '  | |/ |/ /_/ / / /___/_____// /_/ // ____// /___\n'
        '  |__/|__//___/ \____/      /_____//_/     \____/\n'
        '\n'
        '  {quarter} {year}, University of California, San Diego.\n'
        '*/\n'
    )

    HEADER = (
        '#include <iostream>\n'
    )

    SIGNATURE = (
        '/* Parameters: {params}\n'
        ' * Return: {ret}\n'
        ' */\n'
    )

    METHOD = (
        '{ret_type} {name}({params}) {{\n'
        '{content}'
        '{ret_val}'
        '}}\n'
    )

    RET_EXPR = 'return {};\n'

    MAIN = METHOD.format(ret_type='int', name='main', params='',
                         ret_val=indent(RET_EXPR.format(0)),
                         content='{args_expr}\n{print_expr}\n')

    SIGN_PADDING = ' * ' + len('Parameters: ') * ' '
    PARAM_TEMPLATE = '{0} {1} - {2}'

    LIB = ''
    LIB_PATHS = ['jsonFASTParse/C++/json_fast_parse.hpp']
    PAT_HEADER_CMT = re.compile(r'/\*[^*]*\*+(?:[^*/][^*]*\*+)*/')

    DEFAULT_VAL = {
        TypeParser.INT: '0',
        TypeParser.STR: '""',
        TypeParser.CHR: "'0'",
        TypeParser.FLT: '0.0f',
        TypeParser.DBL: '0.0',
        TypeParser.BOOL: 'false',
    }

    SUFFIX = 'cpp'

    def __init__(self, examples):
        super() .__init__(examples)
        self._load_lib()

    @classmethod
    def _load_lib(cls):
        lib_text = []
        header = set([cls.HEADER.rstrip('\n\t ')])
        cmts = ''
        for path in cls.LIB_PATHS:
            with open(path) as fin:
                text = fin.read()

            if not cmts:
                m = cls.PAT_HEADER_CMT.search(text)
                if m:
                    cmts = m.group(0)

            lines = cls.PAT_HEADER_CMT.sub('', text).strip('\n\t ').split('\n')
            for i, line in enumerate(lines):
                if cls._is_header(line) or line.isspace():
                    header.add(line)
                else:
                    break
            lib_text.extend(lines[i:])
            lib_text.append('')

        cls.LIB = cmts + '\n' + '\n'.join(lib_text).strip('\n\t ') + '\n'
        cls.HEADER = '\n'.join(sorted(header)).strip('\n\t ') + '\n'

        @classmethod
        def _t(cls):
            pass
        cls._load_lib = _t

    @classmethod
    def _is_header(cls, line):
        return line.startswith('# include') or line.startswith('using namespace')

    @classmethod
    def _gen_method(cls, pname, params, ret):
        ret_type, _, _ = ret
        stype, depth = TypeParser.parse_type(ret_type)
        return cls.METHOD.format(
            ret_type=ret_type,
            name=pname,
            params=', '.join('{0} {1}'.format(*p) for p in params),
            ret_val=indent(cls.RET_EXPR.format('null' if depth > 0 else cls.DEFAULT_VAL[stype])),
            content=indent('// TODO;\n'))

    @classmethod
    def _gen_main(cls, pname, params, ret):
        return cls.MAIN.format(
            args_expr=indent(cls._gen_args_expr(pname, params, ret)),
            print_expr=indent(cls._gen_print_expr(pname, params, ret)))

    @classmethod
    def _gen_args_expr(cls, pname, params, ret):
        'string inputContents;\n'
        'getline(cin, inputContents);\n'
        'JSONList* argumentsList =\n'
        '  (JSONList*) JSONParser::getObjectFromString(inputContents);\n'
        pass
        # def new_array(lv, last_list_name, arr_name, top_lv_index=None):
        #     def apfmt(fmt, **kwargs):
        #         exprs.append(TWO_SPCS * lv + fmt.format(**kwargs))

        #     if lv == depth:
        #         return apfmt('{arr_name} = {last_list_name}.getItem({iname}).{cast_func}();',
        #                      arr_name=arr_name, last_list_name=last_list_name,
        #                      iname=top_lv_index if lv == 0 else 'i{}'.format(lv - 1),
        #                      cast_func=cls.LIB_CAST[stype])

        #     list_name = lname_fmt.format(lv)
        #     apfmt('JSONList {list_name} = (JSONList) {last_list_name}.getItem({iname});',
        #           list_name=list_name, last_list_name=last_list_name,
        #           iname=top_lv_index if lv == 0 else 'i{}'.format(lv - 1))
        #     apfmt('{arr_name} = new {str_type}[{list_name}.getEntryCount()]{sub_arr};',
        #           arr_name=arr_name, str_type=str_type, list_name=list_name,
        #           sub_arr='[]' * (depth - lv - 1))
        #     apfmt('for (int i{lv} = 0; i{lv} < {list_name}.getEntryCount(); i{lv}++) {{',
        #           lv=lv, list_name=list_name)
        #     new_array(lv + 1, list_name, '{}[i{}]'.format(arr_name, lv))
        #     apfmt('}}')

        # exprs = []
        # for i, (str_type, name, _) in enumerate(params):
        #     stype, depth = TypeParser.parse_type(str_type)
        #     exprs.append('{} {};'.format(str_type, name))
        #     str_type = str_type[:len(str_type) - depth * 2]  # get pure type
        #     lname_fmt = name + 'List{}'
        #     new_array(0, 'argumentList', name, i)
        # return '\n'.join(exprs)

    @classmethod
    def _gen_print_expr(cls, pname, params, ret):
        'cout << ({print_expr}) << endl;\n'
        pass
        # print_expr = '{}({})'.format(pname, ', '.join(n for (_, n, _) in params))
        # ret_type, _, _ = ret
        # stype, depth = TypeParser.parse_type(ret_type)
        # if depth > 1:
        #     cast_func = 'Arrays.deepToString'
        # elif depth == 1:
        #     cast_func = 'Arrays.toString'
        # elif stype == 'String':
        #     return print_expr
        # elif stype == 'boolean':
        #     return '{} ? "True" : "False"'.format(print_expr)
        # else:
        #     cast_func = 'String.valueOf'
        # return '{}({})'.format(cast_func, print_expr)


class JavaGenerator(CppGenerator):
    HEADER = (
        'import java.util.Arrays;\n'
        'import java.util.Scanner;\n'
    )

    METHOD = 'public static ' + CppGenerator.METHOD

    MAIN = METHOD.format(ret_type='void', name='main', params='String[] args',
                         ret_val='', content='{args_expr}\n{print_expr}\n')

    CLASS = (
        'public class Problem{pnum} {{\n'
        '\n'
        '{body}\n'
        '\n'
        '{foot}\n'
        '}}\n\n'
    )

    LIB_PATHS = [
        'jsonFASTParse/Java/JSONList.java',
        'jsonFASTParse/Java/JSONObject.java',
        'jsonFASTParse/Java/JSONParser.java'
    ]

    LIB_CAST = {
        TypeParser.INT: 'castToInt',
        TypeParser.FLT: 'castToDouble',
        TypeParser.DBL: 'castToDouble',
        TypeParser.CHR: 'castToChar',
        TypeParser.STR: 'getData'
        # TypeParser.BOOL no such thing
    }

    SUFFIX = 'java'

    @classmethod
    def _is_header(cls, line):
        return line.startswith('import')

    @classmethod
    def _generate(cls, pnum, pname, params, ret):
        return [
            cls._gen_header(),
            cls._gen_class(pnum,
                           cls._gen_sign(params, ret) + cls._gen_method(pname, params, ret),
                           cls._gen_banner() + cls._gen_main(pname, params, ret)),
            cls._gen_lib(),
        ]

    @classmethod
    def _gen_class(cls, pnum, body, foot):
        body, foot = map(indent, (body, foot))
        return cls.CLASS.format(pnum=pnum, body=body, foot=foot)

    @classmethod
    def _gen_args_expr(cls, pname, params, ret):
        def new_array(lv, last_list_name, arr_name, top_lv_index=None):
            def apfmt(fmt, **kwargs):
                exprs.append(TWO_SPCS * lv + fmt.format(**kwargs))

            if lv == depth:
                return apfmt('{arr_name} = {last_list_name}.getItem({iname}).{cast_func}();',
                             arr_name=arr_name, last_list_name=last_list_name,
                             iname=top_lv_index if lv == 0 else 'i{}'.format(lv - 1),
                             cast_func=cls.LIB_CAST[stype])

            list_name = lname_fmt.format(lv)
            apfmt('JSONList {list_name} = (JSONList) {last_list_name}.getItem({iname});',
                  list_name=list_name, last_list_name=last_list_name,
                  iname=top_lv_index if lv == 0 else 'i{}'.format(lv - 1))
            apfmt('{arr_name} = new {str_type}[{list_name}.getEntryCount()]{sub_arr};',
                  arr_name=arr_name, str_type=str_type, list_name=list_name,
                  sub_arr='[]' * (depth - lv - 1))
            apfmt('for (int i{lv} = 0; i{lv} < {list_name}.getEntryCount(); i{lv}++) {{',
                  lv=lv, list_name=list_name)
            new_array(lv + 1, list_name, '{}[i{}]'.format(arr_name, lv))
            apfmt('}}')

        exprs = [
            'String input = new Scanner(System.in).nextLine();',
            'JSONList argumentList = (JSONList) JSONParser.getObjectFromString(input);'
        ]
        for i, (str_type, name, _) in enumerate(params):
            stype, depth = TypeParser.parse_type(str_type)
            exprs.append('{} {};'.format(str_type, name))
            str_type = str_type[:len(str_type) - depth * 2]  # get pure type
            lname_fmt = name + 'List{}'
            new_array(0, 'argumentList', name, i)
        return '\n'.join(exprs)

    @classmethod
    def _gen_print_expr(cls, pname, params, ret):
        print_expr = '{}({})'.format(pname, ', '.join(n for (_, n, _) in params))

        cast_fmt, cast_func = '{}({})', 'String.valueOf'
        ret_type, _, _ = ret
        stype, depth = TypeParser.parse_type(ret_type)
        if depth > 1:
            cast_func = 'Arrays.deepToString'
        elif depth == 1:
            cast_func = 'Arrays.toString'
        elif stype == 'String':
            cast_fmt = '{}{}'
            cast_func = ''
        elif stype == 'boolean':
            cast_fmt = '{1} ? "True" : "False"'
        print_expr = cast_fmt.format(cast_func, print_expr)

        return 'System.out.println({});\n'.format(print_expr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='the file to transcribe')
    args = parser.parse_args()
    return args


def main():
    # Get arguments
    args = parse_args()

    # Parse the document
    with open(args.filename) as fin:
        text = fin.read()
    examples = DocParser(text).findall()

    # Check output directory
    check_dir(OUT_DIR)

    # Generate templates
    def gen(gters):
        for gter in gters:
            gter(examples).generate()
    gen([PythonGenerator, JavaGenerator])
    # gen([PythonGenerator, CppGenerator, JavaGenerator])


if __name__ == '__main__':
    main()
