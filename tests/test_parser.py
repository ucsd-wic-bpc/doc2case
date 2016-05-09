import inspect, os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

import doc2case
from doc2case import Tokenizer as Tk, ExampleParser as EP

import unittest

class BaseTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None


class TestTokenizer(BaseTest):

    def compare(self, text, expected):
        actual = Tk(text).lines()
        print(actual)
        # actual = [[e for e in ln if not (isinstance(e, Tk.Space) and e.range[0] == e.range[1])] for ln in actual]
        # for line, e in zip(text.split('\n'), expected):
        #     new_expected.append(e)
        #     if not(e and isinstance(e[-1], Tk.Space)):
        #         length = len(line) + line.count(Tk.TAB) * (Tk.TAB_WIDTH - 1)
        #         e.append(Tk.Space(length, length))
        # self.assertEqual(actual, new_expected)
        self.assertEqual(actual, expected)

    # @unittest.skip('')
    def test_general(self):
        text = '    \n  \n  123,     tRue, 789, \'a\'     \n    1  False, 3 4\n"a","b",Null,\n"asdf",\n,\n12   '
        expected = [
            [],
            [],
            [Tk.Space(0, 2), 123, Tk.Space(6, 11), 'tRue', 789, 'a'],
            [Tk.Space(0, 4), 1, Tk.Space(5, 7), False, '3 4'],
            ['a', 'b', None],
            ['asdf'],
            [],
            [12]
        ]
        self.compare(text, expected)

    # @unittest.skip('')
    def test_arrow(self):
        text = '123=>4\n1 3 ⇒4a\n0        -> 7\n123 =>> 4\n123    -->> 4\n123 ---  4\n 123 >-    5\n'
        expected = [
            [123, Tk.Arrow(3, 5), 4, Tk.Space(6, 13)],
            ['1 3', Tk.Arrow(3, 5), '4a', Tk.Space(7, 13)],
            [0, Tk.Arrow(1, 12), 7, Tk.Space(13, 13)],
            [123, Tk.Arrow(3, 8), 4, Tk.Space(9, 13)],
            [123, Tk.Arrow(3, 12), 4, Tk.Space(13, 13)],
            ['123 ---', Tk.Space(7, 9), 4, Tk.Space(10, 13)],
            ['123 >-', Tk.Space(7, 11), 5, Tk.Space(12, 13)],
            [Tk.Space(0, 13)]
        ]
        self.compare(text, expected)

    # @unittest.skip('')
    def test_brackets(self):
        text = '[4\t\t], 2\n'
        expected = [
            [Tk.OpeningBracket(), 4, Tk.ClosingBracket(), 2],
        ]
        # self.compare(text, expected)


class TestTypeParser(BaseTest):

    def compare(self, strs_type, data, expected):
        self.assertEqual(doc2case.TypeParser(strs_type).parse(','.join(map(str, data)).replace("'", '"')), expected)

    # @unittest.skip('')
    def test_general(self):
        strs_type = ('int[][]', 'String', 'String')
        data = [
            [['1', '0', 2, 3], [], [2], ['3']],
            False,
            None
        ]
        expected = [
            [[1, 0, 2, 3], [], [2], [3]],
            'False',
            None
        ]
        self.compare(strs_type, data, expected)
        self.compare(strs_type, [data], expected)


class TestExampleParser(BaseTest):

    # @unittest.skip('')
    def test_find_margin(self):
        def compare(expected, text, ileft, iright, iarrow=None):
            if iarrow is None:
                actual = EP.find_margins_one_way(Tk(text).lines(), ileft, iright)
            else:
                actual = EP.find_margins(Tk(text).lines(), iarrow, (ileft, iright))
            self.assertEqual(list(actual), expected)
        
        text = ("[[2, 3, 4],          [[9, 1, 2],\n"
                " [1, 5, 3],           [0, 5, 3],\n"
                " [9, 0, 1]],3,L        => [1, 3, 4]]")
        expected = [(11, 21), (11, 22), (15, 26)]
        compare(expected, text, 15, 26, 2)

        #       "[[x, x, x],  ==> [[x, x, x],\n"
        text = (" [x, x, x, x],       [x, x, x],\n"
                " [x, x, x, x],      [x, x, x],\n"
                " [x, x, x]], x, x, x, x       [x, x, x]]")
        expected = [(14, 21), (14, 20), (23, 30)]
        compare(expected, text, 11, 15)

              # "[[],    [],   =>   [[],     []\n"
        text = (" [],    [],         [],     [x, x]\n"
                " [x, x],    [x, x],         [],     [x]\n"
                " [],    []]         [x, x],     [x, x, x]]")
        expected = [(11, 20), (19, 28), (11, 20)]
        compare(expected, text, 11, 19)

    @classmethod
    def clean(cls, l):
        if isinstance(l, str):
            return l.replace('\n', '').replace(' ', '')
        elif not isinstance(l, (list, tuple)):
            return l
        return [cls.clean(e) for e in l]

    # @unittest.skip('')
    def test_make_case(self):
        def compare(expected, text, iarrow):
            lines = Tk(text).lines()
            actual = EP(text)._make_case(lines, 0, len(lines), iarrow)
            self.assertEqual(self.clean(actual), self.clean(expected))

        text = ("[[2, 3, 4],          [[9, 1, 2],\n"
                " [1, 5, 3],           [0, 5, 3],\n"
                " [9, 0, 1]],3,L        => [1, 3, 4]]")
        expected = [
            '[[2, 3, 4], [1, 5, 3], [9, 0, 1]],3,L=>', # _make_case does not remove arrows or comments
            '[[9, 1, 2], [0, 5, 3], [1, 3, 4]]'
        ]
        compare(expected, text, 2)


    # @unittest.skip('')
    def test_parse_cases(self):
        def compare(expected, text):
            actual = EP(text)._parse_cases()
            self.assertEqual(*map(self.clean, (actual, expected)))

        text = ('90.7,  16, ‘B’         => true\n'
                '97.6,  22, ‘F’         => false\n'
                '140.6, 37, ‘A’         => true')
        expected = [
            (('90.7,  16, ‘B’', 'true'), []),
            (('97.6,  22, ‘F’', 'false'), []),
            (('140.6, 37, ‘A’', 'true'), [])
        ]
        compare(expected, text)
        

    # @unittest.skip('')
    def test_general(self):
        def compare(text, expected):
            names = ('input', 'output', 'comments')
            [actual] = EP(text).parse()
            for a, e, name in zip(actual, expected, names):
                self.assertEqual(a, e, 'incorrect {}'.format(name))

        text = ("[[2, 3, 4],          [[9, 1, 2],\n"
                " [1, 5, 3],           [0, 5, 3],\n"
                " [5, 7, 6],           [9, 8, 7],\n"
                " [9, 0, 1]],3,L        => [1, 3, 4]]")
        expected = [
            [[[2, 3, 4],
              [1, 5, 3],
              [5, 7, 6],
              [9, 0, 1]], 3, 'L'],

            [[[9, 1, 2],
              [0, 5, 3],
              [9, 8, 7],
              [1, 3, 4]]],

            []
        ]
        compare(text, expected)

if __name__ == '__main__':
    unittest.main()

