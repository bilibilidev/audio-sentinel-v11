from common import ROOT,read,LABELS,NAMES,title,table,pause,entry

def demo(ctx):
    title('亮点 1/5 · 五类音频事件识别')
    print('读取五条示例录音，逐条运行 Linux C 分类。')
    results=[]
    for index,label in enumerate(LABELS,1):
        row,wav=ctx.example(label);print(f'\n[{index}/5] audiodetect {wav.name} --debug',flush=True);ctx.play(wav)
        result=ctx.classify(wav)
        print('识别：'+NAMES[result['prediction']]+f"  置信度：{result['confidence']:.1%}")
        print(f"处理时间：{result['elapsed_ms']:.2f} ms（CLI进程）")
        table(['类别','分值'],[[NAMES[k],f"{result['probabilities'][k]:.1%}"] for k in LABELS],right=(1,))
        results.append({'id':row['id'],'truth':label,**{k:v for k,v in result.items() if k!='raw'}})
        pause(ctx)
    correct=sum(r['truth']==r['prediction'] for r in results)
    print(f'本次示例：{correct}/{len(results)} 条分类正确。')
    ctx.save_result('classify',{'recordings':results,'correct':correct,'total':len(results)})
    def records():
        title('实验记录 · 五类分类对比（2026-10-26）')
        study=read(ROOT/'results/mlp-study-20261026/summary.json');rows=[('Tiny MLP',study['fixed_recipe_cv'])]
        for key,name in [('MFCC_SVM','MFCC + SVM'),('MFCC_RF','MFCC + RF')]:
            rows.append((name,next(b for b in study['baselines'] if b['name']==key)['held_out']))
        table(['模型','ESC五类五折','准确率'],[[n,f"{r['correct']}/{r['total']}",f"{r['accuracy']:.2%}"] for n,r in rows],right=(1,2))
        print('来源：results/mlp-study-20261026/summary.json')
    ctx.show_records(records)

if __name__=='__main__':raise SystemExit(entry(demo,'五条录音的 Linux C 实时分类'))
