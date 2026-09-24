import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


human = load("run_humaneval")
gsm = load("run_gsm8k")
tool = load("test_tool_calling_regression")
guard = load("gpu_idle_guard")


def response(content=None, finish="tool_calls", calls=None):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": content, "tool_calls": calls}}]}


def call(name="view_file", arguments='{"path":"test.py"}', call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class ExtractionTests(unittest.TestCase):
    def test_fenced_body_preserves_first_line_indentation(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=newline):
                source = human.extract_code("def candidate():\n", f"```python{newline}    return 42{newline}```")
                self.assertIn("    return 42", source)
                compile(source, "completion", "exec")

    def test_full_function_not_duplicated(self):
        source = human.extract_code("def candidate():", "```python\ndef candidate():\n    return 42\n```")
        self.assertEqual(source.count("def candidate"), 1)

    def test_number_formats(self):
        cases = {"Judy makes $7,425 in one week": "7425", "net -1,234.50": "-1234.5",
                 "total +1,234,567": "1234567", "The answer is: $7,425.": "7425",
                 "#### 7,425": "7425", "result .5": "0.5", "result -42": "-42",
                 "Choose from 1, 2, 3": "3", "no number": ""}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(gsm.extract_prediction(text), expected)


class SandboxTests(unittest.TestCase):
    def run_fake(self, start_result=0, cleanup_error=False, mounts=None):
        commands = []

        def execute(cmd, **kwargs):
            commands.append((cmd, kwargs))
            if cmd[1] == "start":
                if isinstance(start_result, Exception):
                    raise start_result
                return subprocess.CompletedProcess(cmd, start_result)
            if cmd[1] == "inspect":
                if ".Mounts" in cmd[2]:
                    return subprocess.CompletedProcess(cmd, 0, json.dumps(mounts or []))
                return subprocess.CompletedProcess(cmd, 0, json.dumps({"Status": "exited", "ExitCode": start_result, "Error": ""}))
            if cmd[1] == "rm" and cleanup_error:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0, "")

        with patch.object(human.subprocess, "run", side_effect=execute):
            result = human.run_test("def f(): return 42", "def check(f): assert f() == 42", "f")
        return result, commands

    def test_restricted_execution_and_cleanup(self):
        passed, commands = self.run_fake()
        self.assertTrue(passed)
        create = commands[0][0]
        for flag in ("--runtime=runsc", "--network=none", "--read-only", "--cap-drop=ALL",
                     "--user=65534:65534", "--security-opt=no-new-privileges=true",
                     "--pids-limit=64", "--memory=256m", "--log-driver=none"):
            self.assertIn(flag, create)
        self.assertNotIn("--volume", create)
        self.assertNotIn("--env", create)
        start = next(kwargs for cmd, kwargs in commands if cmd[1] == "start")
        self.assertEqual(start["stdout"], subprocess.DEVNULL)
        self.assertEqual(commands[-1][0][1:4], ["rm", "--force", "--volumes"])

    def test_timeout_removes_container(self):
        passed, commands = self.run_fake(subprocess.TimeoutExpired("docker", 5))
        self.assertFalse(passed)
        self.assertEqual(commands[-1][0][1], "rm")

    def test_failed_candidate_is_not_infrastructure_error(self):
        self.assertFalse(self.run_fake(1)[0])

    def test_cleanup_failure_aborts(self):
        with self.assertRaises(human.SandboxError):
            self.run_fake(cleanup_error=True)

    def test_image_declared_host_volume_rejected_before_execution(self):
        commands = []

        def execute(cmd, **kwargs):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, json.dumps([{"Type": "volume", "Destination": "/data"}]))

        with patch.object(human.subprocess, "run", side_effect=execute):
            with self.assertRaisesRegex(human.SandboxError, "host-backed mounts"):
                human.run_test("", "", "f")
        self.assertFalse(any(cmd[1] == "start" for cmd in commands))
        self.assertEqual(commands[-1][1], "rm")

    def test_missing_runtime_never_falls_back(self):
        with patch.object(human.subprocess, "run", side_effect=FileNotFoundError("docker")) as run:
            with self.assertRaises(human.SandboxError):
                human.run_test("", "", "f")
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][0], "docker")

    def test_sandbox_error_is_not_scored(self):
        problem = {"task_id": "test", "prompt": "", "test": "", "entry_point": "f"}
        with patch.object(human, "call_model", return_value=""), patch.object(human, "run_test", side_effect=human.SandboxError("unavailable")):
            with self.assertRaises(human.SandboxError):
                human.eval_one(problem, "unused", "unused")


class ToolValidationTests(unittest.TestCase):
    def analyze(self, raw):
        return tool.analyze_turn(2, raw, None, False, False)

    def test_valid_calls_and_final_answer(self):
        for raw in (response(calls=[call()]), response(calls=[call("grep_search", '{"query":"abc"}')]),
                    response("The requested file is missing.", "stop")):
            self.assertTrue(self.analyze(raw)["passed"])

    def test_invalid_tools(self):
        bad = [call("unknown"), call(arguments="{"), call(arguments="{}"),
               call(arguments='{"path":42}'), call(arguments='{"path":"x","limit":true}'),
               call(arguments='{"path":"x","limit":"2"}'), call(arguments="null"),
               call(arguments='{"path":"x","limit":NaN}'), call(call_id=""),
               {"id": "x", "type": "function"}, "invalid"]
        for candidate in bad:
            with self.subTest(candidate=candidate):
                self.assertFalse(self.analyze(response(calls=[candidate]))["passed"])
        self.assertFalse(self.analyze(response(calls=[call(), call()]))["passed"])
        self.assertFalse(self.analyze(response(calls="not a list"))["passed"])

    def test_unusable_second_turn(self):
        for raw in (response(None, "length"), response("", "stop"), response("  ", "stop"),
                    response(None, "stop"), response("partial", "length"), response(calls=[call()], finish="length"),
                    response("answer", "content_filter"), response("Work State", "stop"),
                    response("answer", "tool_calls")):
            with self.subTest(raw=raw):
                self.assertFalse(self.analyze(raw)["passed"])

    def test_invalid_first_turn_does_not_receive_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(tool, "post_json", return_value=(response(calls=[call(arguments="{}")]), 0.1)) as post, patch("sys.stdout", new_callable=io.StringIO):
            result = tool.run_trial(1, "http://unused", "model", 0, 100, Path(tmp))
        self.assertEqual(post.call_count, 1)
        self.assertFalse(result[0]["passed"])

    def test_simulated_results_use_validated_arguments(self):
        requests = []
        replies = iter([response(calls=[call(arguments='{"path":"actual.py"}')]), response("File missing.", "stop")])

        def post(url, body):
            requests.append(copy.deepcopy(body))
            return next(replies), 0.1

        with tempfile.TemporaryDirectory() as tmp, patch.object(tool, "post_json", side_effect=post), patch("sys.stdout", new_callable=io.StringIO):
            result = tool.run_trial(1, "http://unused", "model", 0, 100, Path(tmp))
        self.assertTrue(all(r["passed"] for r in result))
        self.assertEqual(requests[1]["messages"][-1]["content"], "Error: File not found: actual.py")


class OccupancyTests(unittest.TestCase):
    def test_launcher_checks_all_nodes_before_mutating_any_node(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash unavailable")

        def shell_path(path):
            path = Path(path).resolve()
            return f"/{path.drive[0].lower()}{path.as_posix()[2:]}" if os.name == "nt" else str(path)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            log = directory / "ssh.log"
            stub = directory / "ssh"
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$GUARD_LOG"\ncase "$*" in\n  *node1*) exit 9 ;;\n  *) exit 0 ;;\nesac\n', encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            env = dict(os.environ, NODE0="node0", NODE1="node1", HOME=shell_path(directory),
                       GUARD_LOG=shell_path(log))
            result = subprocess.run(
                [bash, "-c", 'export PATH="$1:$PATH"; exec bash "$2" 2', "guard-test",
                 shell_path(directory), shell_path(ROOT / "scripts" / "sglang-flashnext-boot.sh")],
                env=env, capture_output=True, text=True, timeout=15,
            )
            self.assertNotEqual(result.returncode, 0)
            commands = log.read_text()
            self.assertIn("node0 python3", commands)
            self.assertIn("node1 python3", commands)
            self.assertNotIn("drop_caches", commands)
            self.assertNotIn("swapoff", commands)
            self.assertNotIn("docker run", commands)

    def test_gpu_containers_without_active_cuda_context(self):
        hosts = [{"DeviceRequests": [{"Driver": "nvidia", "Capabilities": [["gpu"]]}]},
                 {"Runtime": "nvidia"}, {"Privileged": True},
                 {"Devices": [{"PathOnHost": "/dev/nvidia0"}]},
                 {"Devices": [{"PathOnHost": "nvidia.com/gpu=all"}]}]
        for host in hosts:
            with self.subTest(host=host), patch.object(guard, "command", side_effect=["", "id1", json.dumps([{"Name": "/glm53-rank0", "HostConfig": host}])]):
                with self.assertRaisesRegex(RuntimeError, "glm53-rank0"):
                    guard.assert_idle(8100)

    def test_native_gpu_process_blocks(self):
        with patch.object(guard, "command", return_value="1234, python3\n"):
            with self.assertRaisesRegex(RuntimeError, "GPU process"):
                guard.assert_idle(8100)

    def test_unknown_query_status_blocks(self):
        with patch.object(guard, "command", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                guard.assert_idle(8100)

    def test_cpu_container_allowed(self):
        with patch.object(guard, "command", side_effect=["", "id1", json.dumps([{"Name": "/cpu", "HostConfig": {}, "Config": {}}]), ""]):
            guard.assert_idle(8100)

    def test_busy_port_blocks(self):
        with patch.object(guard, "command", side_effect=["", "", "LISTEN 0 128 *:8100 *:*"]):
            with self.assertRaisesRegex(RuntimeError, "Port"):
                guard.assert_idle(8100)


@unittest.skipUnless(os.environ.get("HUMANEVAL_SANDBOX_TESTS") == "1", "requires provisioned Docker/runsc")
class RealSandboxTests(unittest.TestCase):
    def test_isolation(self):
        source = '''
import os, socket
def candidate():
    assert "HUMANEVAL_HOST_SECRET" not in os.environ
    try:
        open("/host-write-probe", "w")
    except OSError:
        pass
    else:
        raise AssertionError("writable root filesystem")
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=0.5)
    except OSError:
        pass
    else:
        raise AssertionError("network access")
'''
        with patch.dict(os.environ, {"HUMANEVAL_HOST_SECRET": "not-forwarded"}):
            self.assertTrue(human.run_test(source, "def check(fn): fn()", "candidate", timeout=30))

    def test_timeout_kills_descendants(self):
        names = []
        real_run = subprocess.run

        def record(cmd, **kwargs):
            if cmd[:2] == ["docker", "create"]:
                names.append(cmd[cmd.index("--name") + 1])
            return real_run(cmd, **kwargs)

        with patch.object(human.subprocess, "run", side_effect=record):
            self.assertFalse(human.run_test("import os\nos.fork()\nwhile True: pass", "", "unused", timeout=10))
        result = real_run(["docker", "ps", "-aq", "--filter", f"name={names[0]}"], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
