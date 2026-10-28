"""Generate noisy full-recording WAVs and classify them using Linux C."""
from common import ROOT,read,LABELS,NAMES,sha,title,table,pause,entry
from audio_examples import read_pcm,write_pcm,add_gaussian

def demo(ctx):
    title('亮点 2/5 · 噪声鲁棒性')
    print('五条示例录音，现场生成20/10/5 dB高斯噪声，再运行完整WAV分类。')
    examples=[(label,*ctx.example(label)) for label in LABELS]
    cases=[]
    for snr in (None,20,10,5):
        condition='干净输入' if snr is None else f'{snr} dB'
        print('\n运行：'+condition,flush=True)
        for index,(label,row,wav) in enumerate(examples):
            source=wav;actual_snr=None;seed=None
            if snr is not None:
                seed=2026102800+snr*10+index
                noisy,actual_snr=add_gaussian(read_pcm(wav),snr,seed)
                source=ctx.work_dir/'noise'/f"{row['id']}-snr{snr}.wav";write_pcm(source,noisy)
            result=ctx.classify(source)
            print(f"  {row['id']:<16} -> {NAMES[result['prediction']]}  {result['confidence']:.1%}",flush=True)
            cases.append({'id':row['id'],'truth':label,'snr_db':snr,'achieved_snr_db':actual_snr,'seed':seed,
                          'input_sha256':sha(source),**{k:v for k,v in result.items() if k!='raw'}})
        if snr!=5:pause(ctx,'按 Enter 继续下一组输入')
    print('\n本次示例的分类对照')
    rows=[]
    for label,row,_ in examples:
        values=[next(c for c in cases if c['id']==row['id'] and c['snr_db']==s) for s in (None,20,10,5)]
        rows.append([row['id'],NAMES[label]]+[NAMES[c['prediction']]+f" {c['confidence']:.0%}" for c in values])
    table(['录音','标签','干净','20 dB','10 dB','5 dB'],rows)
    totals=[]
    for snr in (None,20,10,5):
        selected=[c for c in cases if c['snr_db']==snr];correct=sum(c['truth']==c['prediction'] for c in selected)
        totals.append({'snr_db':snr,'correct':correct,'total':len(selected)})
    table(['本次输入','示例正确数'],[['干净' if r['snr_db'] is None else str(r['snr_db'])+' dB',f"{r['correct']}/{r['total']}"] for r in totals],right=(1,))
    ctx.save_result('robust',{'recordings_per_condition':len(examples),'conditions':totals,'cases':cases})
    def records():
        title('实验记录 · ESC五类五折噪声对比（2026-10-26）')
        study=read(ROOT/'results/mlp-study-20261026/summary.json')
        conditions=[('干净输入',study['fixed_recipe_cv'])]+[(str(r['snr_db'])+' dB',r) for r in study['noise']]
        table(['输入','正确/总数','准确率','宏平均召回'],[[n,f"{r['correct']}/{r['total']}",f"{r['accuracy']:.2%}",f"{r['macro_recall']:.2%}"] for n,r in conditions],right=(1,2,3))
        print('来源：results/mlp-study-20261026/summary.json')
    ctx.show_records(records)

if __name__=='__main__':raise SystemExit(entry(demo,'现场加噪并运行 Linux C 完整WAV分类'))
