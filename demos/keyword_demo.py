import re
from common import ROOT,read,sha,title,table,pause,entry

def demo(ctx):
    title('亮点 5/5 · 关键词识别')
    print('合成语音示例：救命 / 帮助 / 报警，调用 Linux C 的 MFCC + DTW。')
    if not ctx.keyword_dir:raise RuntimeError('请配置关键词示例目录 --keyword-dir。')
    rows=[];results=[]
    for k,name in enumerate(['救命','帮助','报警']):
        for variant in range(3):
            filename=f'kw-{k}-{variant}.wav'
            source=next(r for r in ctx.manifest['keywords'] if r['path'].endswith(filename));wav=ctx.keyword_dir/filename
            if not wav.is_file() or sha(wav)!=source['sha256']:raise RuntimeError('关键词示例缺失或内容不符：'+filename)
            print(f'运行：audiodetect keyword {filename}',flush=True);ctx.play(wav)
            text,elapsed=ctx.run(['keyword',wav],wav,require_model=False,echo=False)
            match=re.search(r'Keyword: (\S+)(?: \(([0-9.]+)%\))?',text)
            if not match:raise RuntimeError('C程序没有返回关键词结果。')
            prediction=match[1];score=float(match[2]) if match[2] else None
            rows.append([filename,name,prediction,'-' if score is None else f'{score:.1f}%'])
            results.append({'id':filename,'expected':name,'prediction':prediction,'match_score_percent':score,'elapsed_ms':elapsed,'synthetic':True})
        pause(ctx)
    for filename in ['silence.wav','low-noise.wav']:
        source=next(r for r in ctx.manifest['keywords'] if r['path'].endswith(filename));wav=ctx.keyword_dir/filename
        if not wav.is_file() or sha(wav)!=source['sha256']:raise RuntimeError('关键词负例缺失或内容不符：'+filename)
        text,elapsed=ctx.run(['keyword',wav],wav,require_model=False,echo=False);prediction=re.search(r'Keyword: (\S+)',text)[1]
        rows.append([filename,'none',prediction,'-']);results.append({'id':filename,'expected':'none','prediction':prediction,'elapsed_ms':elapsed,'synthetic':True})
    table(['示例','关键词','识别','匹配得分'],rows,right=(3,))
    positives=sum(r['expected']==r['prediction'] for r in results[:9]);negatives=sum(r['expected']==r['prediction'] for r in results[9:])
    print(f'本次结果：关键词 {positives}/9，静音与低噪声 {negatives}/2。')
    ctx.save_result('keyword',{'cases':results,'positive_correct':positives,'positive_total':9,'negative_correct':negatives,'negative_total':2})
    def records():
        title('实验记录 · 合成关键词（2026-10-26）')
        reference=read(ROOT/'results/mlp-study-20261026/linux/auxiliary.json')
        positive=reference['keyword_synthetic_cases'];negative=reference['keyword_negative_cases']
        table(['输入','实验结果'],[['合成关键词',f"{sum(r['matched'] for r in positive)}/{len(positive)}"],['静音与低噪声',f'{negative}/{negative}']],right=(1,))
        print('来源：results/mlp-study-20261026/linux/auxiliary.json')
    ctx.show_records(records)

if __name__=='__main__':raise SystemExit(entry(demo,'现场运行合成关键词与负例'))
