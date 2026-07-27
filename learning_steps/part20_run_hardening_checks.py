from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def main() -> None:
    checks = []

    code, out = run_cmd([sys.executable, "learning_steps/part17_api_lite.py", "--self-test"])
    checks.append(
        {
            "name": "api_self_test",
            "return_code": code,
            "passed": code == 0,
            "output_snippet": out[:600],
        }
    )

    code, out = run_cmd([sys.executable, "learning_steps/part20_tests_api_lite.py"])
    checks.append(
        {
            "name": "api_unit_tests",
            "return_code": code,
            "passed": code == 0,
            "output_snippet": out[:1200],
        }
    )

    passed_all = all(item["passed"] for item in checks)
    report = {"passed_all": passed_all, "checks": checks}

    out_path = Path("learning_steps") / "outputs" / "part20_hardening_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
