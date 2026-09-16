"""Stream a training command while owning its lifetime (standard library only)."""

import subprocess
from pathlib import Path


def run_logged(command, log_path, emit=None, stop_timeout=10):
    """Append combined output; stop and reap this child if the cell is interrupted.

    This stops the current epoch, not saves it. Resume from the last completed
    checkpoint. Runtime deletion or a killed notebook kernel cannot run cleanup.
    The trainer currently uses num_workers=0 and creates no child workers.
    """
    if emit is None:
        emit = lambda line: print(line, end="", flush=True)
    with Path(log_path).open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                emit(line)
            returncode = proc.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        finally:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=stop_timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            finally:
                proc.stdout.close()
