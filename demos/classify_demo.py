from common import ROOT,read,LABELS,NAMES,title,table,pause,entry
def demo(ctx):
    title('亮点 1/5 · 五类音频事件识别')
    print('以下每类固定一个已知正确示例，用于操作演示；不是重新测量65条准确率。')
    print('执行端：Ubuntu Linux 原生 C 进程；不是完整系统模拟器或真实开发板。')
    for label in LABELS:
        row,wav=ctx.example(label);print('\n[示例] '+NAMES[label]+' | '+row['id']);ctx.play(wav)
        text,elapsed=ctx.run([wav,'--debug'],wav)
        print(f'[本次 Linux C] 进程墙钟 {elapsed:.1f} ms（含启动/读取/DSP/推理；不是真板时延）')
        pause(ctx)
    print('\n[已有实验数据] 固定配方 ESC-50 五类子集官方五折；非本次重测。')
    study=read(ROOT/'results/mlp-study-20261026/summary.json')
    rows=[('Tiny MLP',study['fixed_recipe_cv'])]
    for key,name in [('MFCC_SVM','MFCC + SVM'),('MFCC_RF','MFCC + RF')]:
        rows.append((name,next(b for b in study['baselines'] if b['name']==key)['held_out']))
    table(['模型','保留折正确/总数','准确率'],[
        [name,f'{result["correct"]}/{result["total"]}',f'{result["accuracy"]:.2%}'] for name,result in rows
    ],right=(1,2))
    print('来源：results/mlp-study-20261026/summary.json；不是ESC-50全50类成绩。')
    pause(ctx)
if __name__=='__main__':raise SystemExit(entry(demo,'五类本机 C 操作演示'))
