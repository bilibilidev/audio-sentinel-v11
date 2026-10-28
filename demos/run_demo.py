"""Five independent console highlights; default selects one, never auto-chains."""
import sys
from common import title,table,entry
from classify_demo import demo as classify
from robust_demo import demo as robust
from model_demo import demo as model
from alarm_demo import demo as alarm
from keyword_demo import demo as keyword

SECTIONS=[('classify','五类音频事件识别',classify),('robust','噪声鲁棒性：五折实验表',robust),
          ('model','模型优化与轻量化：对比表',model),('alarm','远程告警接口：本机回环',alarm),
          ('keyword','关键词识别',keyword)]

def demo(ctx):
    section=ctx.args.section
    if section=='menu':
        title('声哨 Audio Sentinel · 五个独立亮点演示')
        table(['编号','演示内容','独立启动命令'],[
            [i,name,'bash demos/'+key+'_demo.sh'] for i,(key,name,_) in enumerate(SECTIONS,1)])
        print('每次只演示所选一项；表格读取已发布实验，功能演示执行Linux C。')
        if ctx.args.auto or not sys.stdin.isatty():return
        choice=input('\n输入1-5选择，或直接 Enter 退出：').strip()
        if not choice:return
        if choice not in ['1','2','3','4','5']:raise RuntimeError('请输入1-5。')
        section=SECTIONS[int(choice)-1][0]
    if section=='all':
        for _,_,fn in SECTIONS:fn(ctx)
    else:
        next(fn for key,_,fn in SECTIONS if key==section)(ctx)

if __name__=='__main__':raise SystemExit(entry(demo,'五项独立亮点；默认菜单，--section all 可按序检查'))
