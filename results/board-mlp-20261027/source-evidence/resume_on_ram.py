"""Preserve failing NAND data; continue with isolated tmpfs test inputs."""
import json,time
import board_suite as suite
from board_io import Nsh,adb
s=suite;root=s.root;stages=[]
s.shell=Nsh(root/'ram-suite-shell.log')
try:
    s.idle()
    # Read-only recheck after the storage fault before scoring any NAND data.
    adb(['pull','/data/m27/b',str(root/'roundtrip-nand-recheck')],root/'nand-recheck-pull.log',180)
    for i,r in enumerate(s.manifest['records']):assert s.sha(root/'roundtrip-nand-recheck'/f'{i:03}.wav')==r['sha256']
    nand=s.evaluate('nand-clean65',s.manifest['records']);stages.append('nand_clean65')
    s.remote='/tmp/m27'
    s.upload();stages.append('ram_input_roundtrip280')
    clean=s.evaluate('clean65',s.manifest['records']);stages.append('ram_clean65')
    s.latency();stages.append('ram_fixed3s_latency')
    for snr in [20,10,5]:s.evaluate('noise'+str(snr),s.manifest['noise'][str(snr)]);stages.append('ram_noise'+str(snr))
    s.auxiliary(clean);stages.append('auxiliary')
    s.capture();stages.append('ram_microphone_capture')
    s.soak();stages.append('short_soak200')
    s.write('completed.json',dict(status='CORE_EXPERIMENTS_COMPLETE_WITH_NAND_WRITE_FAILURE',stages=stages,
                                 model_sha256=s.model,final_free=s.run('free')[0],final_processes=s.idle(),
                                 uptime=s.run('uptime')[0],storage_issue='SPI0 transfer timeout / YAFFS write retries during initial bulk NAND input upload; subsequent uploads/captures confined to tmpfs',
                                 system_acceptance_pass=False,one_hour_stability_tested=False,power_measured=False))
    print('CORE_EXPERIMENTS_COMPLETE_WITH_NAND_WRITE_FAILURE',flush=True)
except Exception as exc:
    s.write('ram-suite-error.json',dict(error=str(exc),completed_stages=stages));raise
finally:s.shell.close()
