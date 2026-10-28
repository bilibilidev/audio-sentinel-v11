from common import sha,title,pause,entry
def demo(ctx):
    title('亮点 5/5 · 关键词识别：MFCC + DTW')
    print('关键词：救命 / 帮助 / 报警。以下是合成语音回归样例，不是真人语音准确率。')
    if not ctx.keyword_dir:raise RuntimeError('缺少合成关键词目录；请设置 --keyword-dir 或 demos/local_paths.json。')
    names=['救命','帮助','报警']
    for k,name in enumerate(names):
        row=next(r for r in ctx.manifest['keywords'] if r['path'].endswith(f'kw-{k}-0.wav'))
        p=ctx.keyword_dir/f'kw-{k}-0.wav'
        if not p.is_file() or sha(p)!=row['sha256']:raise RuntimeError('关键词文件缺失或SHA不一致：'+p.name)
        print('\n[期望] '+name+'（合成语音）');ctx.play(p)
        text,_=ctx.run(['keyword',p],p,require_model=False)
        if 'Keyword: '+name not in text:raise RuntimeError('关键词回归结果与预期不同。')
        pause(ctx)
    row=next(r for r in ctx.manifest['keywords'] if r['path'].endswith('silence.wav'));p=ctx.keyword_dir/'silence.wav'
    if not p.is_file() or sha(p)!=row['sha256']:raise RuntimeError('缺少已冻结的静音负例。')
    print('\n[负例] 数字静音，应输出 none');text,_=ctx.run(['keyword',p],p,require_model=False)
    if 'Keyword: none' not in text:raise RuntimeError('静音负例未通过。')
    print('[通过] 本次3条合成正例 + 1条静音负例；完整历史回归为9正例 + 2负例。')
if __name__=='__main__':raise SystemExit(entry(demo,'关键词识别独立演示'))
