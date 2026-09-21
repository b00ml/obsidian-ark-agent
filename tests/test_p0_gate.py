import unittest

from scripts.p0_gate import _configure_stdout, _tail, build_commands, run_command


class P0GateTests(unittest.TestCase):
    def test_command_matrix_covers_required_surfaces(self):
        names = {name for name, _command, _cwd in build_commands()}
        self.assertEqual(
            names,
            {
                "agentlab", "agentlab_compileall", "bili_summarizer",
                "inbox_collector", "mcp", "ark_unit", "ark_build",
                "git_diff_check", "doctor", "contract_schemas",
            },
        )

    def test_skip_is_explicit_and_successful(self):
        result = run_command("demo", ("missing-command-for-gate-test",), skip=True)
        self.assertTrue(result.skipped)
        self.assertEqual(result.returncode, 0)
        self.assertIn("skipped", result.output_tail)

    def test_tail_is_bounded(self):
        output = "\n".join(f"line-{i}" for i in range(1000))
        tail = _tail(output, limit=100)
        self.assertLessEqual(len(tail), 100)
        self.assertIn("line-999", tail)

    def test_stdout_configuration_is_optional(self):
        _configure_stdout()


if __name__ == "__main__":
    unittest.main()
