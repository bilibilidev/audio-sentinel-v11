"""Combine candidate spaces using INNER scores only, never outer predictions."""
import argparse,hashlib,json,math,pathlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--kind',choices=['mlp','cnn'],required=True);p.add_argument('--with-pcen',action='store_true')
p.add_argument('--with-centered',action='store_true')
p.add_argument('--stages',help='Explicit comma-separated frozen search spaces')
p.add_argument('--output-name')
a=p.parse_args();base=a.root.resolve();stages=[a.kind,a.kind+'-expanded'];read=lambda p:json.loads(p.read_text(encoding='utf-8'))
if a.stages:
    stages=a.stages.split(',');assert all(pathlib.Path(s).name==s for s in stages)
if a.with_pcen:
    assert a.kind=='cnn';stages.append('cnn-pcen')
if a.with_centered:
    assert a.kind=='cnn';stages.append('cnn-centered')
assert all((base/stage/'frozen.json').exists() for stage in stages),'Finish both candidate spaces first'
frozen={s:read(base/s/'frozen.json') for s in stages};chosen=[];finals=[];scores={mode:[] for mode in ('float','int8')}
for fold in range(1,6):
    options=[]
    for stage in stages:
        outer=next(r for r in frozen[stage]['outer'] if r['fold']==fold)
        inner=read(base/stage/'inner'/f'fold_{fold}'/(outer['config']['name']+'.json'))
        options.append(dict(stage=stage,inner_utility=inner['utility'],outer=outer))
    best=max(options,key=lambda r:r['inner_utility']);outer=best['outer']
    chosen.append(dict(model=f'fold_{fold}',stage=best['stage'],model_path=f'{best["stage"]}/models/fold_{fold}.npz',
        model_sha256=outer['model_sha256'],config=outer['config'],selected_epoch=outer['selected_epoch'],inner_utility=best['inner_utility'],
        model_selection_used_outer_score=False))
    for mode in scores:scores[mode]+=outer[mode]['predictions']
for stage in stages:
    ranking=max(frozen[stage]['inner_ranking'],key=lambda r:r['mean_inner_utility'])
    finals.append(dict(stage=stage,**ranking))
selected=max(finals,key=lambda r:r['mean_inner_utility']);stage=selected['stage'];model=base/stage/'models/final_all_esc50.npz'
chosen.append(dict(model='final_all_esc50',stage=stage,model_path=f'{stage}/models/final_all_esc50.npz',model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    config=selected['config'],selected_epoch=selected['median_epoch'],mean_inner_utility=selected['mean_inner_utility'],model_selection_used_outer_score=False))
summary={}
for mode,rows in scores.items():
    assert len(rows)==390 and len({r['filename'] for r in rows})==390
    matrix=[[0]*5 for _ in range(5)]
    for row in rows:matrix[row['truth']][row['prediction']]+=1
    correct=sum(matrix[i][i] for i in range(5))
    summary[mode]=dict(correct=correct,total=390,accuracy=correct/390,confusion_matrix=matrix,macro_recall=sum(matrix[i][i]/sum(matrix[i]) for i in range(5))/5)
output_name=a.output_name or (a.kind+'-selected');assert pathlib.Path(output_name).name==output_name
out=base/output_name;out.mkdir(exist_ok=True)
assert not (out/'frozen.json').exists()
result=dict(status='COMBINED_INNER_SELECTION_FROZEN',kind=a.kind,models=chosen,official_nested_cv=summary,
    selection='For each outer fold, compare maximum INNER utility across base and expanded-background spaces; outer labels never choose model/epoch. Final uses highest mean INNER utility.',
    stage_final_ranking=finals,threshold_correct_esc=math.ceil(.85*390),threshold_correct_external=math.ceil(.85*65),
    external_opened=False,old_default_unchanged=True,training_data_note='Expanded stages use1992 ESC recordings after excluding8 cross-fold-source extra backgrounds; base stages390. All evaluate the same390 held-out recordings.')
(out/'frozen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(a.kind,'combined nested',summary,'final',stage,selected['config']['name'])
