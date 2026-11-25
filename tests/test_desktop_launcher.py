import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import desktop_launcher as launcher


class TestLauncherHelpers(unittest.TestCase):
    def test_build_streamlit_argv_headless(self):
        argv = launcher.build_streamlit_argv("8502", headless=True)
        self.assertEqual(argv[0], "streamlit")
        self.assertEqual(argv[1], "run")
        self.assertTrue(argv[2].endswith("AkShareDemo.py"))
        self.assertIn("--server.address", argv)
        self.assertEqual(argv[argv.index("--server.address") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--server.headless") + 1], "true")
        self.assertEqual(argv[argv.index("--server.port") + 1], "8502")

    def test_build_streamlit_argv_browser_mode(self):
        argv = launcher.build_streamlit_argv("8503", headless=False)
        self.assertEqual(argv[argv.index("--server.headless") + 1], "false")

    def test_build_child_command_source_mode(self):
        command = launcher.build_child_command("8502")
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[1].endswith("desktop_launcher.py"))

    def test_build_child_command_frozen_mode(self):
        with patch.object(sys, "frozen", True, create=True):
            command = launcher.build_child_command("8502")
        self.assertEqual(command, [sys.executable])

    def test_find_free_port_skips_used_ports(self):
        with patch.object(launcher, "_port_is_free", side_effect=[False, True]) as mocked:
            port = launcher.find_free_port(8501)
        self.assertEqual(port, "8502")
        self.assertEqual(mocked.call_count, 2)

    def test_wait_for_server_success(self):
        response = MagicMock()
        response.__enter__.return_value.status = 200
        with patch.object(launcher.urllib.request, "urlopen", return_value=response) as mocked:
            self.assertTrue(launcher.wait_for_server("http://127.0.0.1:8502"))
        mocked.assert_called_once()

    def test_wait_for_server_timeout(self):
        with patch.object(launcher.urllib.request, "urlopen", side_effect=OSError):
            self.assertFalse(launcher.wait_for_server("http://127.0.0.1:8502", timeout=0.1))

    def test_main_dispatches_server_mode(self):
        env = {
            launcher.PORT_ENV: "8510",
            launcher.SERVER_MODE_ENV: "1",
        }
        with patch.dict(os.environ, env), patch.object(launcher, "run_server", return_value=0) as mocked:
            result = launcher.main()
        self.assertEqual(result, 0)
        mocked.assert_called_once_with("8510")


if __name__ == "__main__":
    unittest.main()
