"""Print the published five-fold noise experiment; optionally probe one WAV."""
from common import ROOT,read,title,table,pause,entry
def demo(ctx):
    title('亮点 2/5 · 噪声鲁棒性：五折实验对比表')
    print('[已有实验数据] 2026-10-26 实际 Linux C；以下不是本次重新跑五折。')
    study=read(ROOT/'results/mlp-study-20261026/summary.json')
    clean=study['fixed_recipe_cv'];rows=[]
    conditions=[('干净输入',clean)]+[(str(r['snr_db'])+' dB 高斯噪声',r) for r in study['noise']]
    for label,result in conditions:
        rows.append([label,f'{result["correct"]}/{result["total"]}',f'{result["accuracy"]:.2%}',
                     f'{result["macro_recall"]:.2%}',f'{(clean["accuracy"]-result["accuracy"])*100:.2f}'])
    table(['输入条件','正确/总数','五折准确率','宏平均召回','下降(pp)'],rows,right=(1,2,3,4))
    print('同一固定配方、同一390条保留折输入；仅改变测试输入，不重新训练。')
    print('高斯噪声不等于真实房间噪声；强噪声下明显下降，不能声称噪声无影响。')
    print('来源：results/mlp-study-20261026/{summary.json,noise-inputs.json}')
    if ctx.args.probe:
        pause(ctx,'按 Enter，额外运行单条 WAV 的首一秒 robust 原型')
        row,wav=ctx.example('cough');print('\n[本次运行] '+row['id'])
        print('仅取首一秒，对比 clean/20/10/5 dB；结果不是上表五折成绩。')
        ctx.run(['robust',wav],wav,require_model=False)
    pause(ctx)
if __name__=='__main__':raise SystemExit(entry(demo,'噪声五折实验表；--probe 可附加单条首秒测试'))
