#! /usr/bin/env python3

"""
doc2case
~~~~~~~~
Transcribe document-style I/O from the problem sets into the case format.

Usage:
    ./doc2case.py [-h] [-m directory] [-t type] filename

    directory:
        Path to the /dev/cases folder. If specified, this script will read the
        files, which contain IO cases, from the directory, and join those
        cases into yours.

    type:
        Type of your IO. It could be any string, but usually it is either
        'corner', 'general', or 'sample'.

    filename:
        The .txt file of the problem sets written by you. This script will
        look for the IO between the captions "Examples" and "Required Method
        Signature", and transcribe them into JSON files of the case format.

        So the best way to write IO is to directly replace the examples in the
        draft with yours. Make sure to download the draft on Google Docs as
        plain text so that this script can read it.

Example:
    $ ./doc2case.py -m ~/Dropbox/WiC-BPC-Fa15/Solutions/dev/cases -t corner ./Fa15ASkyFullofUnicorns.txt

    A folder called "cases" will be created in the current working directory.
    A file will be created for each of your IO in ./Fa15ASkyFullofUnicorns.txt
    and the corresponding IO file in ~/Dropbox/WiC-BPC-Fa15/Solutions/dev/cases
    joind. The files are named from "problem1_corner.json" to
    "problem15_corner.json".

IO Format:
    Reasonable format similar to those of the examples.

Requirement:
    python 3.0 (probably, but 3.5 will do anyway)

Note:
    Manual validating the output is required.

Author:
    Simon Zhang


TODO: another int argument specifying the number of case to join with


"""

import re
import json
import os
from collections import OrderedDict, Iterable
from itertools import chain
import argparse


OUT_DIR = 'cases/'

MAX_INT = 2 ** 31 - 1


# http://stackoverflow.com/questions/10097477/python-json-array-newlines
INDENT = 3
SPACE = " "
NEWLINE = "\n"

is_narray = False


def to_json(o, level=0, is_in_array=False):
    global is_narray
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o.replace('"', '\\"') + '"'
    elif isinstance(o, list):
        elements = [to_json(e, level + 1, True) for e in o]
        if is_in_array:
            is_narray = True
        if not is_in_array and is_narray:
            elements = map(lambda x: '\n' + SPACE * INDENT * (level + 1) + x, elements)
            ret += "[" + ','.join(elements) + '\n' + SPACE * INDENT * level + "]"
            is_narray = False
        else:
            ret += "[" + ', '.join(elements) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        # ret += '%.7g' % o
        ret += str(o)
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


def memo(f):
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            data = cache[args] = f(*args)
            return data
    cache = {}
    return _f


def extend_example(target, data, name):
    """
    :param dict target: dict that will be written to file
    :param [object] data: list of elements comments to add to target
    :param str name: tag name of each element from data in target
    """
    try:
        cases = target['cases']
    except KeyError:
        cases = target['cases'] = OrderedDict()

    offset = max(cases)
    for i, d in enumerate(data):
        ordinal = str(offset + i)
        try:
            case = cases[ordinal]
        except KeyError:
            case = cases[ordinal] = OrderedDict()
        case[name] = d


def export_examples(dicts, io_type, suffix):
    name = '{dir}problem{{num}}{type}'.format(dir=OUT_DIR, type=io_type) + suffix
    for d in dicts:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(name.format(num=d.pop(PBLM_NUM)), 'w') as fout:
            fout.write(to_json(d))

PBLM_NUM = 'PNUM'


def insert_dict(target, tag):
    def get_value(d, k):
        try:
            return d[k]
        except KeyError:
            v = d[k] = OrderedDict()
            return v

    cases = get_value(target, 'cases')
    len_cases = len(cases)
    while True:
        offset, value = yield len_cases
        case = get_value(cases, str(offset))
        case[tag] = value


def get_joining_filenames(join_path, cnt_examples, io_type):
    filenames = [f for f in os.listdir(join_path) if f.endswith('.json')]
    if len(filenames) != cnt_examples:
        if io_type in PRESET_IO_TYPES:
            filenames = [f for f in filenames if io_type in f]

    if len(filenames) == cnt_examples:
        try:
            filenames.sort(key=lambda x: int(
                           re.search(r'problem(\d+)_\w+.json', x).group(1)))
            return filenames
        except AttributeError:  # Some filenames are not of the format 'problemXX.json'
            pass

    raise ValueError('Found {} examples from the problem set, but there are {} files from the cases folder to join with.'
                     .format(cnt_examples, len(filenames)))

PRESET_IO_TYPES = frozenset(['corner', 'sample', 'general'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--join', metavar='path', default=None,
                        help='path to the /dev/cases folder')
    parser.add_argument('-t', '--type', metavar='type', default='', help='type of the IO')
    parser.add_argument('-d', '--dest', metavar='type', default=SEP,
                        choices=(SEP, TGTH), help='save IO seperately or together')
    parser.add_argument('filename', help='the file to transcribe')
    args = parser.parse_args()
    return args

SEP = 'seperate'
TGTH = 'together'


class TypeParser:
    """Parse tokens with the help of types from method signature
    :param [str] types:
    """

    GENERIC = 'a'
    LIST = '[]'
    INT = 'int'
    STR = 'str'
    CHR = 'char'
    FLT = 'float'
    DBL = 'double'
    BOOL = 'boolean'

    TYPES = [
        (frozenset(['int', 'Integer', 'Int', 'long', 'Long']), INT),  # long long!?
        (frozenset(['str', 'String', 'string']), STR),
        (frozenset(['char', 'Character']), CHR),
        (frozenset(['float', 'Float']), FLT),
        (frozenset(['double', 'Double']), DBL),
        (frozenset(['bool', 'boolean', 'Boolean']), BOOL)
    ]

    def _java_str(s):
        if s is None:
            return None
        if isinstance(s, list):
            return ', '.join(map(str, s))
        return str(s)

    def _bool(s):
        if isinstance(s, bool):
            return s
        elif isinstance(s, str):
            s = s.lower()
            if s == 'true':
                return True
            elif s == 'false':
                return False
        raise ValueError(repr(s) + ' is not a valid boolean')

    def _char(c):
        try:
            c = str(c)
        except TypeError:
            pass
        else:
            if len(c) == 1:
                return str(c)
        raise ValueError('invalid char ' + repr(c))

    def _to_num(f):
        def _tn(s):
            try:
                return f(s)
            except Exception:
                pass
            ss = s.split()
            if ss != [s]:
                return _tn(ss[0])
            raise ValueError('invalid number ' + repr(s))

        return _tn

    FUNCS = {
        INT: _to_num(int),
        STR: _java_str,
        CHR: _char,
        FLT: _to_num(float),
        DBL: _to_num(float),
        BOOL: _bool
    }
    del _java_str, _bool, _char, _to_num

    def __init__(self, strs_type=None):
        if strs_type:
            self.types = []
            for t in strs_type:
                type_ = self._parse_type(t)
                if type_:
                    self.types.append(type_)
        else:
            self.types = None
        # print(strs_type)
        # print(self.types)

    def parse(self, text):
        """
        :return object:
        """
        # TODO could contain multiple arguments
        # TODO continue to split string if necessary
        stack = []
        curr = []
        local_depth = 0
        stypes = []
        for token in Tokenizer(text).line():
            if isinstance(token, Tokenizer.OpeningBracket):
                stack.append(curr)
                curr = []
                local_depth = max(len(stack), local_depth)
            elif isinstance(token, Tokenizer.ClosingBracket):
                if not stack:
                    raise SyntaxError('brackets do not pair')
                last_curr = stack.pop()
                last_curr.append(curr)
                curr = last_curr
                if len(stack) == 0:
                    stypes.append(self.GENERIC + self.LIST * local_depth)
                    local_depth = 0
            elif isinstance(token, Tokenizer.Token):
                pass
            else:
                curr.append(token)
                if len(stack) == 0:
                    stypes.append(type(token).__name__)
        if stack:
            raise SyntaxError('missing closing brackets')

        # print(curr)
        data = curr
        if self.types:
            try:
                data = self.cast(data)
            except TypeError as e:
                e.args = ('{}: type ( {} ) is required, but type ( {} ) is given in {}'.format(
                    e.args[0], ' * '.join(t + self.LIST * d for t, d in self.types), ' * '.join(stypes), repr(text)),)
                raise
        return data

    def _assemble(self, data):
        def get_depth(data):
            if isinstance(data, list):
                return (max(get_depth(d) for d in data) if data else 0) + 1
            return 0

        # Check data depth
        expected_depth = max(d for t, d in self.types)
        max_depth = get_depth(data) - 1

        while max_depth > expected_depth and len(data) == 1 and isinstance(data[0], list):
            [data] = data  # do not need the outmost list wrapper
            max_depth -= 1
        # assert expected_depth = self.types[0][1] > 0
        while max_depth < expected_depth and len(self.types) == 1:
            data = [data]  # need the outmost list wrapper
            max_depth += 1

        if max_depth == expected_depth:
            return data

        if max_depth < expected_depth:
            # Assemble data
            types = list(self.types)
            ll, rl = [], []
            while len(data) > 1 and len(types) > 1:
                (_, dl), (_, dr) = types[0], types[-1]
                if dl == 0:
                    if isinstance(data[0], list):
                        raise TypeError('type mismatch')
                    ll.append(data[0])
                    types.pop(0)
                    data.pop(0)
                elif dr == 0:
                    if isinstance(data[-1], list):
                        raise TypeError('type mismatch')
                    rl.append(data[-1])
                    types.pop()
                    data.pop()
                else:
                    # TODO time to split multiple arrays again by comma
                    raise NotImplementedError('cannot parse multiple arrays')
            ll.append(data)
            ll.extend(reversed(rl))
            data = ll

            # print('data assembled', data)
            return data

        raise TypeError('lists in the data have dimension {}, but the method signature specifies {}'.format(
            max_depth, expected_depth))

    def cast(self, data):
        def c(value, type_, depth=0):
            if depth == 0:
                return type_(value)
            if not isinstance(value, Iterable):
                raise TypeError('value and method signature do not match')
            return [c(val, type_, depth - 1) for val in value]

        def cast_iterable(data, types):
            return [c(d, self.FUNCS[t], depth) for d, (t, depth) in zip(data, types)]

        data = self._assemble(data)

        # Check data length
        len_type = len(self.types)
        delta = len(data) - len_type

        # print(self.types)
        # print(data)
        # print(self.types, data)
        if delta > 0:
            if len_type == 1 and self.types[0][0] == self.STR:
                # Parse the whole data as a string
                return c(data, self.FUNCS[self.STR])
            else:
                while set(map(type, data)) == set([list]) and delta > 0:
                    # Decrease dimension of arrays by joining
                    data = chain(*data)

        elif delta < 0:
            # Split each Tokenizer.str since they could be aggregation of other types
            # Assume all str either have quotes or contain no spaces, and the data is
            # of correct type
            data = list(chain.from_iterable(d.split()
                                            for d in data if isinstance(d, str) and not isinstance(d, Tokenizer.str)))
        if len(data) >= len_type:
            # If length is the same, it should be fine. Otherwise discard excess elements
            return cast_iterable(data, self.types)

        raise TypeError(('more' if delta > 0 else 'less') +
                        ' arguments than the method signature has specified')

    @classmethod
    def _parse_type(cls, stype):
        depth = 0
        while stype.endswith(cls.LIST):
            stype = stype[:-2]
            depth += 1

        type_ = None
        for types, t in cls.TYPES:
            if stype in types:
                type_ = t
        if type_ is None:
            return None

        return (type_, depth)


class Tokenizer:
    """Parse a JSON-like formatted, one-line string"""
    DBL_QUOTES = ('"', '“', '”')
    SGL_QUOTES = ("'", '‘', '’')
    END = '\0'
    COMMA = ','
    SPACE = ' '
    O_BRKT = '['
    C_BRKT = ']'
    PARENTHESIS = '('
    ARROW = '⇒'
    ARROW_BODIES = ('-', '=')
    ARROW_HEAD = '>'

    MIN_SPACE_WIDTH = 2  # minimum consequtive spaces required to be a seperation

    class State:
        START = 'start'
        NORMAL = 'normal'
        STR = 'string'
        CHR = 'character'
        CMT = 'comment'
        ARR = 'arrow'
        PARR = 'post arrow'
        SPACE = 'spaces'

    class Token:

        def __repr__(self):
            return '<{}.{}>'.format(Tokenizer.__name__, type(self).__name__)

    class OpeningBracket(Token):
        pass

    class ClosingBracket(Token):
        pass

    class _RangeToken(Token):

        def __init__(self, istart, iend):
            self.range = (istart, iend)

        def __repr__(self):
            t = type(self)
            return '<{}.{} range={}>'.format(Tokenizer.__name__, t.__name__, repr(self.range))

        def __eq__(self, other):
            return type(self) == type(other) and self.range == other.range

    class Comment(_RangeToken):

        def __init__(self, istart, iend, text):
            super().__init__(istart, iend)
            self.text = text

    class Arrow(_RangeToken):

        def __init__(self, outer_start, outer_end, inner_start=None, inner_end=None):
            super().__init__(outer_start, outer_end)
            self.inner_range = self.range if inner_start is None or inner_end is None else (
                inner_start, inner_end)

    class Space(_RangeToken):
        pass

    class char(str):
        pass

    class str(str):
        pass

    def __init__(self, text):
        self.text = text
        self.i = None

    def lines(self):
        """Return parsed tokens in lines.
        :return [[Token]]:
        """
        return [self._tokens(line) for line in self.text.split('\n')]

    def line(self):
        """Return parsed tokens in one list.
        :return [Token]:
        """
        return self._tokens(self.text.replace('\n', ','))

    def _tokens(self, text):
        try:
            return self._run(text)
        except SyntaxError as e:
            e.args = ('{} at index {} in {}'.format(e.args[0], self.i, repr(self.text)),)
            raise

    def _run(self, text):
        # Assume the first argument start with no spaces
        State = self.State
        text = list(text)
        text.append(self.END)
        tokens = []
        buf = []

        state = State.NORMAL
        i = 0
        while i < len(text):
            self.i = i
            c = text[i]
            if state is State.NORMAL:
                if c in self.DBL_QUOTES:
                    self._rstrip(buf)
                    if buf:
                        raise SyntaxError('expected seperator between identifier and string')
                    state = State.STR
                elif c in self.SGL_QUOTES:
                    self._rstrip(buf)
                    if buf:
                        # Assume it's apostrophe
                        buf.append(c)
                    else:
                        state = State.CHR
                elif c == self.ARROW:
                    buf.append(c)
                    state = State.PARR
                elif c in self.ARROW_BODIES:
                    state = State.ARR
                    i -= 1
                elif c == self.COMMA:
                    self._tokenize(tokens, buf)
                elif c == self.END:
                    self._tokenize(tokens, buf)
                elif c == self.O_BRKT:
                    self._tokenize_brkt(tokens, buf, c)
                elif c == self.C_BRKT:
                    self._tokenize(tokens, buf)
                    self._tokenize_brkt(tokens, buf, c)
                elif c in self.PARENTHESIS:
                    self._tokenize(tokens, buf)
                    state = State.CMT
                    i -= 1
                elif c == self.SPACE:
                    state = State.SPACE
                    i -= 1
                else:
                    buf.append(c)
            elif state is State.STR:
                if c in self.DBL_QUOTES:
                    self._tokenize_str(tokens, buf)
                    state = State.NORMAL
                elif c == self.END:
                    raise SyntaxError('missing terminating " character')
                else:
                    buf.append(c)
            elif state is State.CHR:
                if c in self.SGL_QUOTES:
                    self._tokenize_char(tokens, buf)
                    state = State.NORMAL
                elif c == self.END:
                    raise SyntaxError('missing terminating \' character')
                else:
                    buf.append(c)
            elif state is State.ARR:
                if c in self.ARROW_BODIES:
                    if buf and buf[-1] == self.ARROW_HEAD:
                        state = State.NORMAL
                    buf.append(c)
                elif c == self.ARROW_HEAD:
                    buf.append(c)
                else:
                    if buf and buf[-1] != self.ARROW_HEAD:
                        state = State.NORMAL
                    else:
                        state = State.PARR
                    i -= 1
            elif state is State.PARR:
                if c == self.SPACE:
                    buf.append(c)
                elif c == self.END:
                    raise SyntaxError('no identifier on the right-hand side of arrow')
                else:
                    self._tokenize_arrow(tokens, buf, i)
                    state = State.NORMAL
                    i -= 1
            elif state is State.SPACE:
                if c == self.SPACE:
                    buf.append(c)
                elif c == self.ARROW:
                    buf.append(c)
                    state = State.PARR
                elif c in self.ARROW_BODIES:
                    state = State.ARR
                    i -= 1
                elif c == self.END:
                    self._tokenize(tokens, buf)
                else:
                    self._tokenize_space(tokens, buf, i)
                    state = State.NORMAL
                    i -= 1
            elif state is State.CMT:
                if c == self.END:
                    self._tokenize_cmt(tokens, buf, i)
                else:
                    buf.append(c)
            else:
                raise RuntimeError('corrupted state')
            i += 1
        i -= 1  # for self.END
        # TODO replace with self.END
        tokens.append(self.Space(i, i))

        return tokens

    @classmethod
    def _rstrip(cls, buf):
        count = 0
        while buf:
            if buf[-1] == cls.SPACE:
                count += 1
            else:
                break
            buf.pop()
        return count

    @classmethod
    def _tokenize_arrow(cls, tokens, buf, outer_end):
        count = 0
        count += cls._rstrip(buf)
        if buf and buf[-1] == cls.ARROW:
            count += 1
            inner_start = outer_end - count
            inner_end = inner_start + 1
            buf.pop()
        else:
            inner_end = outer_end - count
            while buf and buf[-1] == cls.ARROW_HEAD:
                count += 1
                buf.pop()
            while buf and buf[-1] in cls.ARROW_BODIES:
                count += 1
                buf.pop()
            inner_start = outer_end - count
        count += cls._rstrip(buf)

        cls._tokenize(tokens, buf)
        tokens.append(cls.Arrow(outer_end - count, outer_end, inner_start, inner_end))

    @classmethod
    def _tokenize_cmt(cls, tokens, buf, iend):
        istart = iend - cls._rstrip(buf)
        if not buf:
            return

        text = ''.join(buf)
        buf.clear()
        tokens.append(cls.Comment(istart - len(text), iend, text))

    @classmethod
    def _tokenize_brkt(cls, tokens, buf, c):
        cls._rstrip(buf)
        if buf:
            raise SyntaxError('expected seperator between identifier and list')
        if c == cls.O_BRKT:
            b = cls.OpeningBracket()
        elif c == cls.C_BRKT:
            b = cls.ClosingBracket()
        else:
            return
        tokens.append(b)

    @classmethod
    def _tokenize_str(cls, tokens, buf):
        s = cls.str(''.join(buf))
        buf.clear()
        tokens.append(s)

    @classmethod
    def _tokenize_char(cls, tokens, buf):
        if len(buf) == 0:
            raise SyntaxError('empty character constant')
        elif len(buf) > 1:
            raise SyntaxError('multi-character character constant')

        c = cls.char(''.join(buf))
        buf.clear()
        tokens.append(c)

    @classmethod
    def _tokenize_space(cls, tokens, buf, iend):
        count = cls._rstrip(buf)
        if count >= cls.MIN_SPACE_WIDTH:
            cls._tokenize(tokens, buf)
            tokens.append(cls.Space(iend - count, iend))
        else:
            buf.extend([cls.SPACE] * count)

    @classmethod
    def _tokenize(cls, tokens, buf):
        cls._rstrip(buf)
        if not buf:
            return

        text = ''.join(buf).lstrip()
        buf.clear()

        if text in ('true', 'True'):
            data = True
        elif text in ('false', 'False'):
            data = False
        elif text in ('null', 'Null', 'None'):
            data = None
        else:
            try:
                data = int(text)
            except Exception:
                try:
                    data = float(text)
                except Exception:
                    data = text
        tokens.append(data)


class ExampleParser:
    PAT_IO_HEADERS = re.compile(
        r'\s*(?:Sample )?Input:((?:.|\n)+?)(?:Sample )?Output:((?:.|\n)+?)(?=(?:Sample )?Input:)', re.IGNORECASE)

    RATIO_WM = 1.8  # delta width / delta mid, for calculating weighted distance in find_margins_one_way

    def __init__(self, text, ts_param=None, ts_ret=None):
        self.text = text
        self.strs_ln = text.split('\n')
        self.params = TypeParser(ts_param)
        self.rets = TypeParser(ts_ret)

    def parse(self):
        """Parse cases, within each of which there is a pair of input and output.
        :return (object, object, [str]): input, output, comments
        """
        # print()
        # for data in self._parse_cases():
        #     print('cases:', data)
        cases = self.PAT_IO_HEADERS.findall(self.text + 'Input:')
        try:
            # TODO how about cmts?
            cases = [(case, []) for case in cases] if cases else self._parse_cases()
            return [(self.params.parse(i), self.rets.parse(o), cmts) for (i, o), cmts in cases]
        except (ValueError, SyntaxError) as e:
            # e.args = ('{} in {}'.format(e.args[0], repr(self.text)), )
            # print(e)
            raise type(e)('{} in {}'.format(e.args[0], repr(self.text))) from e

    def _parse_cases(self):
        """
        :return [((str, str), [str])]:
        """
        # TODO cannot parse the first line if it is argument names. Maybe get it
        # from parse signature? It does not appear in recent quarters though
        lines = Tokenizer(self.text).lines()
        lines.append([Tokenizer.Space(0, 0)])
        cases = []

        irear = 0
        ilast_arrow = None
        cmts = []
        for ifront, line in enumerate(lines):
            has_arrow = False
            for token in line:
                if isinstance(token, Tokenizer.Arrow):
                    if has_arrow:
                        raise ValueError('cannot have multiple arrows in one case')
                    has_arrow = True
                elif isinstance(token, Tokenizer.Comment):
                    cmts.append(token.text)

            if has_arrow:
                if ilast_arrow is not None:
                    if ifront - ilast_arrow == 1:
                        # 1. Two arrows in two consequtive lines means there is a case until the
                        # first line
                        case = self._make_case(lines, irear, ifront, ilast_arrow)
                        cases.append((case, cmts))
                        cmts = []
                        irear = ifront
                    else:
                        # TODO how to avoid?
                        # 4. Lines with more than one arrow in them cannot be parsed
                        raise ValueError('don\'t know how to split these cases')
                ilast_arrow = ifront

            elif len(line) == 1:  # for trailing Tokenizer.Space
                # elif line == []:
                if irear == ifront:
                    pass  # consecutive empty lines
                elif ilast_arrow is not None:
                    # 2. An empty line means there is a case until last line
                    case = self._make_case(lines, irear, ifront, ilast_arrow)
                    cases.append((case, cmts))
                    cmts = []
                    ilast_arrow = None
                else:
                    # 3. Lines without an arrow in them cannot be a case. Assume comments.
                    if not cases:
                        raise ValueError('cannot have comments before cases')
                    cmts.append('\n'.join(self.strs_ln[irear:ifront]))
                    cases[-1][1].extend(cmts)
                    cmts = []
                irear = ifront + 1

        return cases

    def _make_case(self, lines, irear, ifront, iarrow):
        """
        :param [Token] lines:
        :return (str, str):
        """
        lines, strs_ln = lines[irear:ifront], self.strs_ln[irear:ifront]
        iarrow -= irear

        # print('tokens before:', iarrow, lines)
        for token in lines[iarrow]:
            if isinstance(token, Tokenizer.Arrow):
                break  # assert lines[iarrow].count(token) == 1

        broken_lns = []
        margins = self.find_margins(lines, iarrow, token.range)
        for str_ln, (_, rmargin), line in zip(strs_ln, margins, lines):
            char_ln = list(str_ln)
            # Remove non-data characters. Remove after splitting to preserving
            # the structure of text
            # Remove comments and arrow
            for l, r in [tk.range for tk in line if isinstance(tk, Tokenizer._RangeToken)]:
                char_ln[l:r] = ' ' * (r - l)

            broken_lns.append(map(''.join, (char_ln[:rmargin], char_ln[rmargin:])))

        return tuple(map('\n'.join, zip(*broken_lns)))

    @classmethod
    def find_margins(cls, lines, iarrow, arrow_range):
        upper = reversed(cls.find_margins_one_way(
            lines[iarrow - 1::-1], *arrow_range)) if iarrow > 0 else []
        lower = cls.find_margins_one_way(
            lines[iarrow + 1:], *arrow_range) if iarrow < len(lines) - 1 else []
        # print('lines', lines)
        # print('upper, lower', upper, lower)
        return chain(upper, [arrow_range], lower)

    @classmethod
    def find_margins_one_way(cls, lines, ileft, iright):
        """
        :param [[Token]] lines:
        :return [(int, int)]: left and right margin of the gap in each line started from ileft and iright

        Example::
            In text version: (this method require tokens though)
                                          0        ileft(11)  iright(23)
                                          v          v           v
                pointer to (lines - 1) -> [[x, x, x],     ==>    [[x, x, x],
                                 lines ->  [x, x, x, x],             [x, x, x],
                                           [x, x, x, x],            [x, x, x],
                                           [x, x, x]], x, x               [x, x, x]]

            Return:
                [(14, 27), (14, 26), (17, 32)]

        dist = delta(value) + delta()
        """
        # print(lines)
        # TODO Gotta use machine learning one day
        arrow_width = iright - ileft
        last_spaces = [(ileft, iright, (iright + ileft) / 2, 0, None)]
        for line in lines:
            curr_spaces = []

            for token in line:
                if not isinstance(token, Tokenizer.Space):
                    continue

                inext_left, inext_right = token.range
                next_width, inext_center = inext_right - inext_left, (inext_right + inext_left) / 2
                min_dist = MAX_INT
                min_ancestor = None

                for ancestor in last_spaces:
                    _, iright, icenter, dist, _ = ancestor
                    dwidth = abs(next_width - arrow_width)
                    if next_width == 0:
                        dwidth *= 10
                    next_dist = dwidth * cls.RATIO_WM + \
                        abs(inext_center - icenter) + dist  # TODO proof?
                    if next_dist < min_dist:
                        min_dist = next_dist
                        min_ancestor = ancestor

                curr_spaces.append((inext_left, inext_right, inext_center, min_dist, min_ancestor))

            if not curr_spaces:
                raise ValueError('cannot find the gap')
            last_spaces = curr_spaces
            # print(curr_spaces)

        poss = []
        ancestor = min(last_spaces, key=lambda x: x[3])
        # print(ancestor)
        # print()
        while True:
            ileft, iright, _, _, ancestor = ancestor
            if ancestor is None:
                break
            poss.append((ileft, iright))
        return list(reversed(poss))


class DocParser:
    """The top-level parser."""
    PAT_BRKTS = re.compile(r'(?<=\n)(\[\w\]).*?\n')
    PAT_EXAMPLE = re.compile(
        r'Problem\s*(\d+)\s*:(?:.|\n)+?Examples?\s*:\n((?:.|\n)+?)(?:Sample File Format\s*:\n(?:.|\n)+?)?Required Method Signature\s*:\n\s*/\*\n((?:.|\n)+?)\s*\*/', re.IGNORECASE)
    PAT_TYPE_ENTRY = re.compile(r'^\s*\*?\s*(\w+(\s*(\[\])*)?)(?:\s|$)')
    # Unfortunately, not all types in method signature end with a dash
    # PAT_TYPE_ENTRY = re.compile(r'^\s*\*?\s*(\w+(?: *\[\])*)\s*(?:\w+\s+)?[–-]?.*(?:!=,)')

    TAB_WIDTH = 4

    def __init__(self, text):
        self.text = self._refine(text)

    @classmethod
    def _refine(cls, text):
        # Remove possible brackets of comment indices at the end of lines
        # TODO may replace text in test cases
        brkts = cls.PAT_BRKTS.findall(text + '\n')
        ifst = len(brkts)
        for i, bracket in enumerate(brkts):
            if bracket == '[a]':
                ifst = i
        brkts = brkts[ifst:]
        brkts = map(re.escape, brkts)
        brkts = map(r'({})$'.format, brkts)
        text = re.sub(r'|'.join(brkts), '', text)

        # Replace tabs
        text = re.sub(r'\t', ' ' * cls.TAB_WIDTH, text)

        return text

    def parse(self):
        """Extract strings of each example in a document, and pass them to ExampleParser.
        :return [object, object, [str]]: data_in, data_out, comments
        """
        # Find examples in release
        strs_example = self.PAT_EXAMPLE.findall(self.text)

        # In case there is a metaexample
        if len(strs_example) == 16:
            del strs_example[0]
        elif len(strs_example) == 0:
            strs_example = [(1, self.text, '')]

        # print(strs_example)
        examples = []
        for pnum, str_data, str_sign in strs_example:
            try:
                ts_param = self._parse_signature(str_sign, 'Parameters:')
                ts_ret = self._parse_signature(str_sign, 'Return:')
                example = ExampleParser(str_data, ts_param, ts_ret).parse()
                examples.append((pnum, example))
            except Exception as e:
                e.args = (e.args[0] + ' at problem ' + pnum, )
                print(e)
                print()
                # raise
        return examples

    @classmethod
    def _parse_signature(cls, text, start=None):
        """Parse types in a Javadoc-style method signature
        :param str start: paring types after the first occurence of start in text
        :return TypeParser:
        """
        if start:
            istart = text.find(start)
            if istart != -1:
                text = text[istart + len(start):]

        strs_type = []
        for line in text.split('\n'):
            if line.isspace():
                continue

            sgms = line.split(',')
            has_type = False
            for sgm in sgms:
                m = cls.PAT_TYPE_ENTRY.match(sgm)
                if not m:
                    break
                has_type = True
                strs_type.append(m.group(1).replace(
                    ' ', '').replace('\t', ''))  # TODO more spaces?

            if not has_type:
                break

        return strs_type if strs_type else None


def main():
    # Get arguments
    args = parse_args()
    filein, join_path, io_type, dest = args.filename, args.join, args.type, args.dest

    # Parse the document
    with open(filein) as fin:
        text = fin.read()
    examples = DocParser(text).parse()
    len_examples = len(examples)

    # # Get files to join
    # if join_path:
    #     targets = []
    #     filenames = get_joining_filenames(join_path, len(examples), io_type)
    #     for filename in filenames:
    #         with open(os.path.join(join_path, filename)) as fin:
    #             try:
    #                 target = json.load(fin, object_pairs_hook=OrderedDict)
    #             except json.JSONDecodeError:
    #                 raise ValueError('{} has invalid format'.format(filename)) from None
    #         targets.append(target)
    # else:
    #     targets = (OrderedDict() for _ in iter(int, 1))

    # Set saving destinations
    make_targets = lambda: [OrderedDict() for _ in range(len_examples)]
    if dest == SEP:
        dicts_in = make_targets()
        dicts_out = make_targets()
    elif dest == TGTH:
        dicts_in = dicts_out = make_targets()

    # Join examples with target dicts
    for (pnum, example), dict_in, dict_out in zip(examples, dicts_in, dicts_out):
        dict_in[PBLM_NUM] = dict_out[PBLM_NUM] = pnum
        gi = insert_dict(dict_in, 'input')
        go = insert_dict(dict_out, 'output')
        gc1 = insert_dict(dict_in, 'comments')
        gc2 = insert_dict(dict_out, 'comments')
        offset = max(map(next, (gi, go, gc1, gc2)))
        for i, (data_in, data_out, cmts) in enumerate(example):
            ordinal = offset + i
            if not isinstance(data_in, list):
                data_in = [data_in]  # TODO all data should be list. done in TypeParser
            gi.send((ordinal, data_in))
            assert len(data_out) == 1, 'Number of returned value should be 1'
            go.send((ordinal, data_out[0]))
            if cmts:
                gc1.send((ordinal, cmts))
                gc2.send((ordinal, cmts))

    # Check output directory
    if os.path.isdir(OUT_DIR):
        msg = 'Directory {} already exsits. Overwrite? [y/n]: '.format(OUT_DIR)
        if not input(msg).startswith('y'):
            return

    # Write target dicts to files
    io_type = '_' + io_type if io_type else ''
    for dicts, suffix in zip((dicts_in, dicts_out), ('.in', '.out')):
        export_examples(dicts, io_type, suffix)

if __name__ == '__main__':
    main()
