import unittest

from gp import __version__, greet
from gp.__main__ import build_parser


class TestGreet(unittest.TestCase):
    def test_default(self) -> None:
        self.assertEqual(greet(), "Hello, world!")

    def test_custom_name(self) -> None:
        self.assertEqual(greet("Ada"), "Hello, Ada!")


class TestCli(unittest.TestCase):
    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.name, "world")

    def test_version_is_set(self) -> None:
        self.assertTrue(__version__)


if __name__ == "__main__":
    unittest.main()
