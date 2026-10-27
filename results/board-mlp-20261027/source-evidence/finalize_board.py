"""Release only this run's verified RAM scratch files; retain NAND evidence."""
import hashlib,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path('board-validation-20261020/scripts').resolve()))
from board_io import Nsh
root=pathlib.Path('board-validation-20261027')
assert json.loads((root/'completed.json').read_text())['status']=='CORE_EXPERIMENTS_COMPLETE_WITH_NAND_WRITE_FAILURE'
assert (root/'ram-owned-test-directory.txt').read_text().strip()=='/tmp/m27'
for rel,digest in json.loads((root/'input-sha256.json').read_text()).items():
    assert hashlib.sha256((root/'ram-roundtrip'/rel).read_bytes()).hexdigest()==digest
capture=json.loads((root/'capture.json').read_text())
assert capture['passed'] and len(capture['records'])==5
for row in capture['records']:
    assert re.fullmatch(r'/tmp/q[0-4]\.wav',row['board_file'])
    assert hashlib.sha256((root/f'microphone-retry-{row["index"]}.wav').read_bytes()).hexdigest()==row['sha256']
s=Nsh(root/'final-inspection-shell.log');rows=[]
def run(cmd):
    text,rc=s.run(cmd);rows.append(dict(command=cmd,exit_code=rc,output=text));assert rc==0,(cmd,text)
    return text
try:
    before=run('free');ps=run('ps');assert not re.search(r'\barecord\s+-D',ps)
    run('ls /tmp/m27')
    # Exact literal path and ownership assertions above bound this cleanup.
    run('rm -r /tmp/m27')
    for row in capture['records']:run('rm '+row['board_file'])
    after=run('free')
    for cmd in ['date','uptime','resetcause','ls -l /data/aed_live.wav','cat /proc/61/stack','cat /proc/62/stack','df']:
        run(cmd)
    text,rc=s.run('dmesg');rows.append(dict(command='dmesg',exit_code=rc,output=text))
    result=dict(status='OWNED_TMPFS_INPUTS_RELEASED',nand_test_files_retained=True,old_models_and_images_unchanged=True,
                free_before_ram_cleanup=before,free_after_ram_cleanup=after,
                warning='maxused includes this run temporary RAM WAV allocations, not model-exclusive peak; underlying NAND write fault not repaired',rows=rows)
    (root/'final-inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RAM scratch released; NAND files and all host evidence retained.')
    print(after)
finally:s.close()
