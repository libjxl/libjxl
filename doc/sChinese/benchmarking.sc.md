# 性能基准测试

若想对单一图片进行单线程或多线程解码速度基准测试，可用`djxl`打印解码速度信息。请参考`djxl --help`以获取有关解码选项的更多细节，请注意，输出图像对基准测试来说不是必须的。
若想获取一份有关多重选项的压缩密度详细对比，请使用`benchmark_xl`工具。

## 使用benchmark_xl进行基准测试

我们推荐简单易用的`build/tools/benchmark_xl`来读取图片或图片序列，使用不同编码方式（jpeg jxl png webp）对其进行编码，对结果进行解码，并对计算客观质量矩阵。一个调用例子如下：

```bash
build/tools/benchmark_xl --input "/path/*.png" --codec jxl:wombat:d1,jxl:cheetah:d2
```

这里可以加入多个使用逗号分隔的编码。 : 后的字母为编码的参数，由冒号分隔，在本例中，其指定了最大目标心理视觉距离（psychovisual distance）分别为1和2（更高的数值代表耕地的质量）以及编码器工作模式（见下文）其他常见的参数有`r0.5`（以每像素0.5比特的比特率为目标）以及`q92`（质量参数92，范围0-100，越高越好）。`jxl`编码支持下列额外的参数：

速度：`lightning`，`thunder`，`falcon`，`cheetah`，`hare`，`wombat`，`squirrel`，`kitten`，`tortoise` 由少到多控制编码器工作量。这也会影响内存占用量：使用复杂度更低的工作模式一般会减少编码期间的内存占用。

*   `lightning` 和`thunder` 是适用于无损模式的高速模式（Modular 模式）。
*   `falcon` 会禁用接下来提到的工具。
*   `cheetah` 会使用coefficient reordering，context clustering以及用于选择DCT尺寸和量化步长的启发法。
*   `hare` 会启用Gaborish filtering，chroma from luma并会为量化步长估计一个初始值。
*   `wombat` 会启用error diffusion quantization以及full DCT size selection启发法。
*   `squirrel` (默认) 会启用dots，patches，spline detection以及full context clustering。
*   `kitten` 会对心里视觉矩阵（psychovisual metric）的自适应量化（adaptive quantization）进行优化
*   `tortoise` 会启用一个更仔细的自适应量化搜素哦（adaptive quantization search）。

模式：JPEG XL具有两个模式。默认是Var-DCT模式，适合有损压缩。另一个是Modular模式，适用于无损压缩。Modular模式也可以用在有损压缩（比如 `jxl:m:q50`）。

*   `m` 会激活modular模式。

benchmark_xl的其他参数包括：

*   `--save_compressed`: 将代码流（codestreams）保存到`output_dir`。
*   `--save_decompressed`: 将解压缩的输出保存到 `output_dir`。
*   `--output_extension`: 选择输出解码图像的格式。
*   `--num_threads`: 独立编、解码图像的编码实例数量，或为0。
*   `--inner_threads`: 每个实例应为并行编、解码使用的线程数量，或为0。
*   `--encode_reps`/`--decode_reps`: 为获得更如一的测量结果，重复编、解码每张图片的次数（我们推荐10）。

基准测试的输出以如下报头开始：

```
Compr              Input    Compr            Compr       Compr  Decomp  Butteraugli
Method            Pixels     Size              BPP   #    MP/s    MP/s     Distance    Error p norm           BPP*pnorm   Errors
```

`ComprMethod` 会列出每个逗号分隔的编码。 `InputPixels` 是输入图片中像素的个数。`ComprSize`是代码流的字节大小，`ComprBPP`代表比特率。 `Compr MP/s` 和`Decomp MP/s`是压缩、解压的吞吐量，单位为Megapixels每秒。`Butteraugli Distance` 代表解码图片中的最大心理视觉错误（psychovisual error）（越大越差）。`Error p norm`是类似的、对心理视觉错误的总结，但更接近平均值，给小的低质量区域分配了更低的权重。`BPP*pnorm`是`ComprBPP` 和`Error p norm`的积，代表编码的性能（越小越好）。`Errors`代表加载、编、解码图像过程中出现的错误。