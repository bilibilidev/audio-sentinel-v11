import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path('board-validation-20261020/scripts').resolve()))
from board_io import Nsh

out = pathlib.Path('board-validation-20261027')
out.mkdir(exist_ok=True)
shell = Nsh(out / 'initial-probe-shell.log')
rows = []
try:
    for command in ['uname -a', 'uptime', 'resetcause', 'ps', 'free', 'ls -l /dev/bootloader',
                    'ls /data', 'ls /dev', 'audiodetect', 'ifconfig']:
        text, rc = shell.run(command)
        rows.append(dict(command=command, exit_code=rc, output=text))
        print(text, flush=True)
    (out / 'initial-probe.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
finally:
    shell.close()
