import csv,hashlib,html,json,pathlib,re,shutil
root=pathlib.Path(__file__).resolve().parent
read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
manifest=read('input-manifest.json');clean=read('clean65.json');nand=read('nand-clean65.json')
assert clean['total']==nand['total']==65
assert len({r['id'] for r in clean['records']})==65
for result in [clean,nand,*[read('noise'+str(s)+'.json') for s in [20,10,5]]]:
    assert result['total']==65 and sum(r['correct'] for r in result['records'])==result['correct']
    assert sum(map(sum,result['confusion_matrix']))==65
    assert sum(result['confusion_matrix'][i][i] for i in range(5))==result['correct']
for a,b in zip(clean['records'],nand['records']):
    assert a['id']==b['id'] and a['input_sha256']==b['input_sha256'] and a['prediction']==b['prediction']
for row in manifest['records']:
    assert any(r['id']==row['id'] and r['input_sha256']==row['sha256'] for r in clean['records'])
aux=read('auxiliary.json');capture=read('capture.json');soak=read('soak.json');completed=read('completed.json')
summary=dict(date='2026-10-27',target='Physical R528S3 Gemini S1 / onboard ILI9341 / OpenVela-NuttX',
             model_sha256=manifest['expected_model_sha256'],application_commit='72f4dfde5224a4e590ff66f4ce1791f5193e7b8c',
             operator_declared_image_sha256=manifest['operator_declared_image_sha256'],image_readback_performed=False,
             dataset_role=manifest['data_role'],classification=dict(correct=clean['correct'],total=65,accuracy=clean['accuracy'],
             macro_recall=clean['macro_recall'],per_class=clean['per_class'],confusion_matrix=clean['confusion_matrix']),
             linux_prediction_matches=clean['linux_prediction_matches'],max_probability_abs_delta_vs_linux=clean['max_probability_abs_delta_vs_linux'],
             nand_and_ram_prediction_matches=sum(a['prediction']==b['prediction'] for a,b in zip(clean['records'],nand['records'])),
             timing=dict(nand_full65_mean_ms=nand['board_command_seconds']['mean']*1000,nand_full65_p95_ms=nand['board_command_seconds']['p95']*1000,
                         ram_full65_mean_ms=clean['board_command_seconds']['mean']*1000,
                         ram_fixed3s=read('latency.json')['board_command_seconds'],scope=clean['timing_scope']),
             noise={str(s):{k:read('noise'+str(s)+'.json')[k] for k in ['correct','total','accuracy','macro_recall','per_class']} for s in [20,10,5]},
             keyword=dict(passed=sum(r['passed'] for r in aux['keywords']),total=len(aux['keywords']),synthetic_only=True,real_voice_accuracy_measured=False),
             input_validation=dict(passed=sum(r['passed'] for r in aux['input_validation']),total=len(aux['input_validation'])),
             int8_alias=dict(passed=sum(r['passed'] for r in aux['int8_alias']),total=len(aux['int8_alias'])),
             microphone=dict(passed=capture['passed'],captures=len(capture['records']),capture_seconds=capture['capture_seconds'],
                             inference_seconds=capture['inference_seconds'],sum_of_separate_command_seconds=capture['sum_of_separate_command_seconds'],
                             accuracy_measured=False,filesystem='tmpfs RAM',scope=capture['scope']),
             network=aux['network'],short_soak={k:soak[k] for k in ['iterations','elapsed_wall_seconds','all_passed','unique_predictions','scope']},
             cold_boot='Observed power_on(0), Oct27 build, audiodetect ui present before test; not an additional reboot cycle',
             display_touch='User confirmed Tiny MLP UI normal and STOP -> IDLE; no automated pixel/touch accuracy measurement',
             nand_write_acceptance=False,system_acceptance_pass=False,power_watts=None,energy_joules=None,
             microphone_field_accuracy_measured=False,one_hour_stability_tested=False,full_system_simulator_tested=False,
             limitations=['SPI0 NAND timeout and YAFFS retries on initial bulk write; remaining input uploads/capture moved to tmpfs',
                          '0.80 CLI full-recording alarm recall differs from argmax classification',
                          'Network host127.0.0.1 has no services; actual HTTP/MQTT send failed; board clock not synchronized',
                          'Global heap includes temporary WAV buffers/files and OS/UI; no inference-exclusive peak RAM or power meter result',
                          'Only final deployment model in firmware; prior337/390 fixed-recipe five-fold remains host evaluation, not board CV',
                          'External65 previously used for development/selection; no new blind/acoustic test labels'])
final=read('final-inspection.json')
heap=[int(x) for x in re.findall(r'\d+',final['free_after_ram_cleanup'])][:7]
memory_samples=[[int(x) for x in re.findall(r'\d+',r['text'])][:7] for r in soak['memory_observations']]
stacks={}
for r in final['rows']:
    if r['command'].endswith('/stack'):
        stacks[r['command'].split('/')[2]]={k:int(v) for k,v in re.findall(r'(StackSize|StackUsed):\s*(\d+)',r['output'])}
summary['memory']=dict(system_heap_total_bytes=heap[0],system_heap_used_after_scratch_release=heap[1],
                       system_heap_free_after_scratch_release=heap[2],lifetime_maxused_including_tmpfs_inputs=heap[3],
                       inference_exclusive_peak_measured=False,temporary_ram_inputs_released=True,
                       short_soak_idle_heap_used_min=min(x[1] for x in memory_samples),
                       short_soak_idle_heap_used_max=max(x[1] for x in memory_samples),
                       ui_thread_stacks=stacks,
                       scope='System allocator including OS/UI/ADB; lifetime high-water contains ~69MB temporary test WAVs. Does not represent model-only peak RAM. 10 idle samples over ~69s are insufficient for long-term leak acceptance.')
summary['board_date_observation']=next(r['output'] for r in final['rows'] if r['command']=='date')
original=pathlib.Path('external-real-audio-review-20261017/curation-v3-baby-20261019/manifest.json')
orig={r['id']:r for r in json.loads(original.read_text(encoding='utf-8'))['records']}
review=root/'review';audio=review/'audio';audio.mkdir(parents=True,exist_ok=True)
errors=[]
for i,row in enumerate(clean['records']):
    if row['correct']:continue
    src=root/'inputs/b'/f'{i:03}.wav'
    assert hashlib.sha256(src.read_bytes()).hexdigest()==row['input_sha256']
    dest=audio/(row['id']+'.wav');shutil.copyfile(src,dest)
    errors.append(dict(row,source_title=orig[row['id']].get('source_title',''),source_dataset=orig[row['id']]['source_dataset'],
                       source_url=orig[row['id']].get('source_url',''),audio='audio/'+dest.name))
with (root/'errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','source_title','truth','prediction','confidence','alerted','board_command_seconds','source_dataset','source_url'])
    w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in errors)
with (root/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
    names=['id','truth','prediction','correct','confidence','alerted','samples','frames','board_command_seconds','linux_prediction_match','max_probability_abs_delta_vs_linux','input_sha256']
    w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows({k:r[k] for k in names} for r in clean['records'])
names=dict(background='日常背景',cough='咳嗽',glass_break='玻璃破碎',baby_cry='婴儿哭声',dog_bark='狗叫')
cards=[]
for r in errors:
    cards.append(f'<article><h2>{html.escape(r["id"])} · {html.escape(r["source_title"])}</h2><p>真实标签：{names[r["truth"]]} → 板端预测：<b>{names[r["prediction"]]}</b>（{r["confidence"]:.1%}）</p><p>Linux C 预测一致：{r["linux_prediction_match"]}；输入 SHA256：<code>{r["input_sha256"]}</code></p><audio controls preload="none" src="{r["audio"]}"></audio></article>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tiny MLP 真板错误录音</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#172530;max-width:920px;margin:32px auto;padding:0 20px}article{background:white;border:1px solid #dbe1e6;padding:16px 22px;margin:18px 0;border-radius:12px}h2{font-size:18px}audio{width:100%}code{font-size:12px;overflow-wrap:anywhere}b{color:#b33931}</style><h1>Tiny MLP：真实开发板错误录音</h1>'''
page+=f'<p>2026-10-27；完整录音 argmax：{clean["correct"]}/65（{clean["accuracy"]:.2%}）。以下 {len(errors)} 条为错误样本；标签仍使用你已听审确认的标签。该65条是开发验证集，不是新盲测或麦克风现场准确率。</p>'
page+=''.join(cards)+'</html>'
(review/'errors.html').write_text(page,encoding='utf-8')
summary['error_recordings']=len(errors);summary['error_review']='review/errors.html'
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['classification','timing','keyword','microphone','short_soak','error_recordings']},ensure_ascii=False,indent=2))
