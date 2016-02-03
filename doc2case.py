#! /usr/bin/env python3

"""
doc2case
~~~~~~~~
Transcribe document-style I/O from the problem sets into the case format.

Usage:
    ./doc2case.py [-h] [-m directory] [-t type] filename

    directory:
        Path to the /dev/cases folder. If specified, this script will read the
        files, which contain IO cases, from the directory, and merge those
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
    merged. The files are named from "problem1_corner.json" to
    "problem15_corner.json".

IO Format:
    Reasonable format similar to those of the examples.

Requirement:
    python 3.0 (probably, but 3.5 will do anyway)

Note:
    Manual validating the output is required.

Author:
    Simon Zhang

"""

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
        for k,v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level+1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o.replace('"','\\"') + '"'
    elif isinstance(o, list):
        elements = [to_json(e, level + 1, True) for e in o]
        if is_in_array:
            is_narray = True
        if not is_in_array and is_narray:
            elements = map(lambda x: '\n' + SPACE * INDENT * (level+1) + x, elements)
            ret += "[" + ','.join(elements) + '\n' + SPACE * INDENT * level + "]"
            is_narray = False
        else:
            ret += "[" + ', '.join(elements) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


import re
import sys
import json
import os
from collections import OrderedDict
import argparse

PATTERN_ALL_QUOTES = r'[“”"]'
QUOTE = r'"'
PATTERN_ALL_SINGLE_QUOTES = r"[‘’']"
SINGLE_QUOTE = r"'"
PATTERN_ALL_ARROWS = r'([-=]+>)|⇒'
ARROW = r'⇒'
PATTERN_EXAMPLES_ALPHA = r'Examples?\s*:\n((?:.|\n)+?)Sample File Format'
PATTERN_EXAMPLES_RELEASE = r'Examples?\s*:\n((?:.|\n)+?)Required Method Signature'
PATTERN_COMMENT_BRACKETS = r'(?<=\n)(\[\w\]).*?\n'
PATTERN_COMMA_AT_EOL = r',\s+$'
PATTERN_ALL_WHITESPACES = r'^\s*$'
PATTERN_SPECIAL_VALUE = r'^\s*((true)|(false)|(null))\s*$'

FORMAT_COMPLEMENTS = ['{}'] + ['[{}]', '{{{}}}', '\"{}\"']
FORMAT_REPLACEMENS = [('', '')] + [('\'', '\"'), ('\n', ','), ('\'', '\\\''), ('\"', '\\\"')]

DIR = 'cases/'
PRESET_IO_TYPES = frozenset(['corner', 'sample', 'general'])

def guess_json(text):
    # Strip whitespaces
    text = text.rstrip(' \n\t').lstrip(' \n\t')
    for COMP in FORMAT_COMPLEMENTS:
        for REPL in FORMAT_REPLACEMENS:
            try:
                return json.loads(COMP.format(text.replace(*REPL)))
            except json.JSONDecodeError:
                pass
    raise json.JSONDecodeError

def recognize_json(text, is_output=False):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Check the case of true/false/null
    if re.search(PATTERN_SPECIAL_VALUE, text, re.IGNORECASE):
        try:
            return json.loads(text.lower())
        except json.JSONDecodeError:
            pass

    # Check if non-JSON strings like comments exist
    try:
        comment, seg_text = text.split('\n', 1)
    except ValueError:
        pass
    else:
        try:
            json_obj = json.loads(seg_text)
        except json.JSONDecodeError:
            pass
        else:
            return (comment, json_obj)

    # Try to guess a potential JSON string
    try:
        return guess_json(text)
    except json.JSONDecodeError:
        pass

    raise ValueError('Cannot recognize the {}: \'{}\''.format('output' if is_output else 'input', text))

def recognize_example(raw, merge_json=None):
    cases = []
    eol_comment = None

    while not re.search(PATTERN_ALL_WHITESPACES, raw):
        try:
            _input, raw = raw.split(ARROW, 1)
        except ValueError:
            # There is no more input and output but still text, probably it is the comment at the last line
            eol_comment, raw = raw.split('\n', 1)
            break
        output, raw = raw.split('\n', 1)
        while re.search(PATTERN_COMMA_AT_EOL, _input):
            input_rem, raw = raw.split('\n', 1)
            _input += input_rem
        cases.append((_input, output))

    # Parse as json
    if merge_json:
        json_cases = merge_json['cases']
        offset = len(json_cases)
    else:
        json_cases = OrderedDict()
        offset = 0
    for i, case in enumerate(cases):
        i += offset
        json_case = OrderedDict()
        _input, output = case
        try:
            json_in = recognize_json(_input)
            json_out = recognize_json(output, True)
        except ValueError as e:
            raise ValueError('cannot recognize the IO: \n{}'.format(raw)) from e
        if isinstance(json_in, tuple):
            comment, json_in = json_in
            if eol_comment:
                # Comments at the EOL of a case will be parsed into the next case
                last_case = json_cases[i - 1]
                last_case['comment'] = comment
                last_case.move_to_end('comment', False)
            else:
                json_case['comment'] = comment
        json_case['input'] = json_in
        json_case['output'] = json_out
        json_cases[i] = json_case
    if eol_comment:
        last_case = json_cases[i]
        last_case['comment'] = eol_comment
        last_case.move_to_end('comment', False)

    output = {'cases' : json_cases}
    return output

def get_merge_filenames(merge_dir, *, io_type=None, examples=None, filename=None):
    merge_files = os.listdir(merge_dir)
    merge_files = list(filter(lambda x: x.endswith('.json'), merge_files))
    if examples and len(merge_files) != len(examples):
        if io_type in PRESET_IO_TYPES:
            possible_files = list(filter(lambda x: io_type in x, merge_files))
            if len(possible_files) == len(examples):
                return possible_files
        raise ValueError('Found {nexp} examples from {filename}, but there are {nmerge} JSON files from {path_merge} to merge with.'
                         .format(nexp=len(examples), filename=filename, nmerge=len(merge_files), path_merge=merge_dir))
    return merge_files


parser = argparse.ArgumentParser()
parser.add_argument('-m', metavar='directory', required=False, default=None, help='path to the /dev/cases folder')
parser.add_argument('-t', metavar='type', required=False, default='', help='type of the IO')
parser.add_argument('filename', help='the IO to transcribe')
args = parser.parse_args()
filename, merge_dir, io_type = args.filename, args.m, args.t

with open(filename) as fin:
    text = fin.read()

# Validate the text
text += '\n'
text = re.sub(PATTERN_ALL_ARROWS, ARROW, text)
text = re.sub(PATTERN_ALL_QUOTES, QUOTE, text)
text = re.sub(PATTERN_ALL_SINGLE_QUOTES, SINGLE_QUOTE, text)
# Remove comment brackets at the end of lines
brackets = re.findall(PATTERN_COMMENT_BRACKETS, text)
if brackets:
    ifirst_bracket = 0
    for i, bracket in enumerate(brackets):
        if bracket == '[a]':
            ifirst_bracket = i
    brackets = brackets[ifirst_bracket:]
    brackets = map(re.escape, brackets)
    brackets = map('({})'.format, brackets)
    pattern_brackets = '|'.join(brackets)
    text = re.sub(pattern_brackets, '', text)

# Find all segments of examples
examples = re.findall(PATTERN_EXAMPLES_RELEASE, text, re.IGNORECASE)
if len(examples) == 0:
    examples = [text]
else:
    alpha_examples = re.findall(PATTERN_EXAMPLES_ALPHA, text, re.IGNORECASE)
    if alpha_examples:
        examples = alpha_examples
    # The problem set draft
    if len(examples) == 16:
        examples = examples[1:]

# Recognize JSONs
if merge_dir:
    # Load the files to merge
    merge_jsons = []
    merge_files = get_merge_filenames(merge_dir, io_type=io_type, examples=examples, filename=filename)
    merge_files = map(lambda x: os.path.join(merge_dir, x), merge_files)
    for merge_file in merge_files:
        with open(merge_file) as fin:
            try:
                mjson = json.load(fin, object_pairs_hook=OrderedDict)
            except json.JSONDecodeError as e:
                raise ValueError('{} cannot be parsed as JSON'.format(merge_file)) from e
            else:
                merge_jsons.append(mjson)
    jsons = [recognize_example(*raw_cases) for raw_cases in zip(examples, merge_jsons)]
else:
    jsons = [recognize_example(raw_cases) for raw_cases in examples]

# Output
if not os.path.isdir(DIR):
    os.mkdir(DIR)
else:
    response = input('Directory {} already exsits. Overwrite? [y/n]: '.format(DIR))
    if response[0] != 'y':
        sys.exit(1)
output_name = '{dir}problem{{num}}{id}.json'.format(dir=DIR, id=('_' + io_type if io_type else ''))
for i, json_obj in enumerate(jsons):
    with open(output_name.format(num=i+1), 'w') as fout:
        fout.write(to_json(json_obj))
