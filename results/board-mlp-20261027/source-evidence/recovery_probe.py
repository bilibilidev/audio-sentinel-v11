import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path('board-validation-20261020/scripts').resolve()))
from board_io import Nsh
root=pathlib.Path('board-validation-20261027')
s=Nsh(root/'transfer-recovery-shell.log');rows=[]
try:
    for cmd in ['uname -a','uptime','resetcause','ps','free','df','ls /data/m27','dmesg']:
        text,rc=s.run(cmd)
        rows.append(dict(command=cmd,exit_code=rc,output=text))
        print(text[-9000:] if cmd=='dmesg' else text,flush=True)
    (root/'transfer-recovery.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
finally:s.close()
