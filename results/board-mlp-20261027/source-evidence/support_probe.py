import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path('board-validation-20261020/scripts').resolve()))
from board_io import Nsh
out=pathlib.Path('board-validation-20261027')
s=Nsh(out/'support-probe-shell.log')
rows=[]
try:
    for cmd in ['df','help','ls -l /data/aed_live.wav','cat /proc/61/status','cat /proc/61/stack']:
        text,rc=s.run(cmd)
        rows.append(dict(command=cmd,exit_code=rc,output=text))
        print(text,flush=True)
    (out/'support-probe.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
finally:s.close()
