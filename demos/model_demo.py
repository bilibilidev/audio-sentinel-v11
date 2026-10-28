"""Highlight 3: source-backed footprint, ablation and timing comparisons."""
from common import ROOT,read,title,table,pause,entry

def demo(ctx):
    title('亮点 3/5 · 模型优化与轻量化：实验数据对比')
    print('[已有实验数据] 正式 Tiny MLP 92 -> 96 -> 5，从随机初始化训练。')
    study=read(ROOT/'results/mlp-study-20261026/summary.json')
    resources=read(ROOT/'results/mlp-study-20261026/resources.json')
    performance=read(ROOT/'results/mlp-study-20261026/linux/performance.json')
    mlp=resources['mlp'];panns=resources['panns']
    baseline=next(b for b in study['baselines'] if b['name']=='PANNs_Cnn6_linearSVM')
    current=study['fixed_recipe_cv'];external=study['external_final']
    print('\n表1：轻量网络与预训练基线（网络 MAC 以5秒输入核算）')
    table(['指标','Tiny MLP','PANNs Cnn6 + SVM'],[
        ['可学习网络参数',f'{mlp["parameters"]:,}',f'{panns["learned_network_parameters"]:,}'],
        ['网络 Conv/Linear MAC',f'{mlp["network_MAC"]:,}',f'{panns["network_MAC_conv2d_linear"]:,}'],
        ['额外预训练','无','AudioSet'],
        ['ESC五折准确率',f'{current["correct"]}/{current["total"]} ({current["accuracy"]:.2%})',
         f'{baseline["held_out"]["correct"]}/{baseline["held_out"]["total"]} ({baseline["held_out"]["accuracy"]:.2%})'],
        ['外部开发集',f'{external["correct"]}/{external["total"]} ({external["accuracy"]:.2%})',
         f'{baseline["external_final"]["correct"]}/{baseline["external_final"]["total"]} ({baseline["external_final"]["accuracy"]:.2%})'],
    ],right=(1,2))
    print(f'参数约为1/{panns["learned_network_parameters"]/mlp["parameters"]:,.0f}；'
          f'网络 MAC 约为1/{panns["network_MAC_conv2d_linear"]/mlp["network_MAC"]:,.0f}。')
    print('PANNs精度更高且有额外预训练；MLP取舍为端侧轻量化。')
    print('MAC不含双方前端/非线性等；PANNs另有SVM系数，未做同板部署。')
    pause(ctx)
    print('\n表2：同一网络的权重/偏置存储核算（不是运行内存）')
    int8=mlp['int8_weights_bias_bytes'];fp32=mlp['parameters']*4
    table(['存储形式','权重/偏置字节','相对FP32'],[
        ['FP32（每参数4B核算）',f'{fp32:,}','100.00%'],
        ['INT8（导出C数组）',f'{int8:,}',f'{int8/fp32:.2%}'],
    ],right=(1,2))
    print(f'权重/偏置减少{1-int8/fp32:.2%}；另有归一化/asinh/量化scale {mlp["normalization_and_asinh_and_quantization_scale_bytes"]:,} B。')
    print('前40维时域特征做asinh压缩，后52维为MFCC/Delta；DSP与神经计算仍为浮点。')
    no_asinh=next(a for a in study['ablations'] if a['variant']=='no_asinh')
    table(['同轮消融','ESC五折正确数','准确率'],[
        ['无asinh',f'{no_asinh["held_out"]["correct"]}/{no_asinh["held_out"]["total"]}',f'{no_asinh["held_out"]["accuracy"]:.2%}'],
        ['正式配方（含asinh）',f'{current["correct"]}/{current["total"]}',f'{current["accuracy"]:.2%}'],
    ],right=(1,2))
    print('消融采用同一组种子，展示本轮控制变量的实验观察。')
    pause(ctx)
    print('\n表3：实际 Ubuntu x86_64 / GCC 11.4 的三秒 WAV 历史测速')
    table(['计时口径','均值(ms)','P95(ms)','次数'],[
        ['分类函数（含读取/DSP/推理）',f'{performance["fixed3s_core"]["mean_ms"]:.3f}',
         f'{performance["fixed3s_core"]["p95_ms"]:.3f}',performance['fixed3s_core']['n']],
        ['CLI进程（另含启动/stdout）',f'{performance["fixed3s_cli_process"]["mean_ms"]:.3f}',
         f'{performance["fixed3s_cli_process"]["p95_ms"]:.3f}',performance['fixed3s_cli_process']['n']],
    ],right=(1,2,3))
    print('预热3次后测30次；不含采集/UI/USB，不是真板或声音发生至告警时间。')
    print('外部65条为已见开发集；功耗未实测，参数/MAC不等于实测能耗比。')
    print('来源：results/mlp-study-20261026/{summary.json,resources.json,linux/performance.json}')
    pause(ctx)

if __name__=='__main__':raise SystemExit(entry(demo,'轻量化、asinh消融和Linux C耗时对比表'))
