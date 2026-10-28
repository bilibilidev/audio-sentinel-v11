"""Compile and run a C measurement harness, then display prior experiments."""
import json,math,statistics,subprocess
from common import ROOT,read,NAMES,title,table,entry
from audio_examples import read_pcm,write_pcm

def percentile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p;low=math.floor(position);high=math.ceil(position)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)

def demo(ctx):
    title('亮点 3/5 · 模型轻量化与处理速度')
    print('编译C测速程序，读取实际模型数组大小，并处理三秒WAV。',flush=True)
    binary=ctx.build_benchmark();_,source=ctx.example('background')
    pcm=read_pcm(source)[:48000]
    if len(pcm)<48000:pcm.extend([0]*(48000-len(pcm)))
    wav=ctx.work_dir/'model-3s.wav';write_pcm(wav,pcm)
    print(f'运行：model-benchmark model-3s.wav {ctx.args.repeats}',flush=True)
    completed=subprocess.run([str(binary),str(wav),str(ctx.args.repeats)],capture_output=True,text=True,timeout=120)
    if completed.returncode:raise RuntimeError('C测速程序退出：'+completed.stderr[-500:])
    measured=json.loads(completed.stdout);times=measured['times_ms']
    assert measured['samples']==48000 and len(times)==ctx.args.repeats
    assert measured['parameters']==ctx.model['parameters']
    measured['mean_ms']=statistics.mean(times);measured['p95_ms']=percentile(times,.95)
    measured['cli_elf_bytes']=ctx.binary.stat().st_size
    ctx.executions.append({'command':['model-benchmark','model-3s.wav',str(ctx.args.repeats)],'stdout':completed.stdout,'stderr':completed.stderr,'exit_code':completed.returncode})
    print('本次分类：'+NAMES[measured['prediction']]+f"  置信度：{measured['confidence']:.1%}")
    table(['本次C程序读取','结果'],[
        ['网络结构',f"{measured['input_dim']} -> {measured['hidden_dim']} -> {measured['classes']}"],
        ['参数数量',f"{measured['parameters']:,}"],['INT8权重/偏置',f"{measured['weight_bytes']:,} B"],
        ['归一化及scale',f"{measured['scale_bytes']:,} B"],['Linux CLI ELF文件',f"{measured['cli_elf_bytes']:,} B"],
        ['单次三秒网络MAC',f"{measured['network_macs']:,}"],
        ['C进程当前RSS',f"{measured['rss_kib']:,} KiB"],
        ['C进程峰值RSS(VmHWM)',f"{measured['peak_rss_kib']:,} KiB"],
    ],right=(1,))
    table(['本次三秒WAV处理','结果'],[
        ['预热次数',measured['warmup']],['测量次数',len(times)],
        ['均值',f"{measured['mean_ms']:.3f} ms"],['P95',f"{measured['p95_ms']:.3f} ms"],
        ['最小 / 最大',f"{min(times):.3f} / {max(times):.3f} ms"],
    ],right=(1,))
    print('计时范围：文件读取、特征提取和MLP推理。')
    ctx.save_result('model',measured)
    def records():
        title('实验记录 · 轻量化、消融与速度对比（2026-10-26）')
        study=read(ROOT/'results/mlp-study-20261026/summary.json');resource=read(ROOT/'results/mlp-study-20261026/resources.json')
        mlp=resource['mlp'];panns=resource['panns'];perf=read(ROOT/'results/mlp-study-20261026/linux/performance.json')
        baseline=next(b for b in study['baselines'] if b['name']=='PANNs_Cnn6_linearSVM')
        table(['指标','Tiny MLP','PANNs Cnn6 + SVM'],[
            ['网络参数',f"{mlp['parameters']:,}",f"{panns['learned_network_parameters']:,}"],
            ['5秒网络MAC',f"{mlp['network_MAC']:,}",f"{panns['network_MAC_conv2d_linear']:,}"],
            ['ESC五折',f"{study['fixed_recipe_cv']['accuracy']:.2%}",f"{baseline['held_out']['accuracy']:.2%}"],
            ['外部开发集',f"{study['external_final']['accuracy']:.2%}",f"{baseline['external_final']['accuracy']:.2%}"],
            ['预训练','无','AudioSet'],
        ],right=(1,2))
        table(['权重/偏置存储','字节','相对FP32'],[
            ['FP32核算',f"{mlp['parameters']*4:,}",'100%'],['INT8导出',f"{mlp['int8_weights_bias_bytes']:,}",'25%'],
        ],right=(1,2))
        no_asinh=next(a for a in study['ablations'] if a['variant']=='no_asinh')
        table(['特征处理','ESC五折'],[['无asinh',f"{no_asinh['held_out']['correct']}/390"],['含asinh',f"{study['fixed_recipe_cv']['correct']}/390"]],right=(1,))
        table(['三秒WAV记录','均值(ms)','P95(ms)'],[
            ['分类函数',f"{perf['fixed3s_core']['mean_ms']:.3f}",f"{perf['fixed3s_core']['p95_ms']:.3f}"],
            ['CLI进程',f"{perf['fixed3s_cli_process']['mean_ms']:.3f}",f"{perf['fixed3s_cli_process']['p95_ms']:.3f}"],
        ],right=(1,2))
        print('网络MAC按Conv/Linear核算；PANNs使用额外预训练。功耗另行测量。')
        print('来源：results/mlp-study-20261026/')
    ctx.show_records(records)

if __name__=='__main__':raise SystemExit(entry(demo,'现场C数组规模与WAV处理测速'))
