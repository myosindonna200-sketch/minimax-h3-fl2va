# 嵩山杯 MiniMax H3 FL2VA 八卡复现

「Anna说的都队」在 InfiniFabric 的 `scnet-dcu` 环境完成了赛事首帧样例的八卡生成。文本编码器和 Transformer 由 InfiniLM 在单节点做八路张量并行，独立作业读取去噪存档，用 Diffusers 的视频和音频 VAE 导出 MP4。视频经逐帧解码验证，共 124 帧，有一条音轨。

## 环境和输入

- 计算环境为 InfiniFabric `scnet-dcu`，版本 `524820f2`，8 张海光 BW DCU，架构 `gfx936`。Python 3.11.9，PyTorch 2.9.0，HIP 6.3.26093，DTK 26.04，XMake 3.0.5。
- `IFAB_AFS_HOME` 由平台设置，编译和推理分别用临时目录，日志、运行包、潜变量和视频写入个人 AFS 的 `minimax-h3-work/`。共享权重路径为 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`。
- 将赛事报告模板的原始 PNG 放在项目根目录，文件名 `first_frame.png`，尺寸 800 × 965，SHA-256 `b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54`。提示词 `A dog falls asleep`，随机种子 42，50 个采样点，124 帧。

## 离线软件包

在个人 AFS 根目录解包 `minimax-h3-offline-bootstrap.tar.gz`，得到 `h3-offline-bootstrap/`。此目录包括 InfiniCore 源码 `f32e2fbb2a9b2ca912b5711f4f3836c403196a74`（含构建所需子模块）、`InfiniLM-minimax_h3-full.tar.gz`、Linux x86_64 XMake 3.0.5 和编译用 Python wheels。InfiniLM 源码包 SHA-256 为 `0a06f8d95855ccfc595d187aa742a76d3bb4ae343c48011c26eb34f79183bc05`。

将 `h3-fl2va-wheels-part1.tar.gz`、`h3-fl2va-wheels-part2.tar.gz` 和 `h3-fl2va-compat.tar.gz` 原样放在个人 AFS 根目录。推理入口会校验和解包。关键依赖为 Diffusers 0.40.0、Transformers 5.12.1、Accelerate 1.13.0、Torchvision 0.24.0+cpu、NumPy 1.26.4、PyAV 16.1.0 和 scikit-image 0.25.2。

## 批处理入口

上传本仓库的三个 `.sh` 文件和赛事首帧到同一个项目。每行对应一份 8 卡批作业，时限 1800 秒。入口文件和参数按下表选择。

| 阶段 | 入口 | 程序参数 | 日志标记 |
| --- | --- | --- | --- |
| 编译 InfiniCore | `build_h3_batch.sh` | `core` | `NEXT_ARGS=` |
| 编译 InfiniLM | `build_h3_batch.sh` | `lm` 加 `NEXT_ARGS` 给出的 Core 归档名 | `RUNTIME_ARCHIVE=` |
| 重编 Softmax | `build_h3_batch.sh` | `softmax` 加上一步给出的运行包文件名 | `H3_SOFTMAX_LONG_INDEX_OK` 和新的 `RUNTIME_ARCHIVE=` |
| 生成潜变量 | `run_fl2va_batch.sh` | `--generate-only` 加修复后的运行包文件名 | `VAE_INPUT_CHECKPOINT=` |
| 独立解码 | `run_fl2va_batch.sh` | `--decode-only` 加结果目录名 `fl2va-ID` | `VIDEO_FRAMES=124 AUDIO_STREAMS=1` 和 `VIDEO_SAVED=` |
| 参考视频对比 | `run_fl2va_batch.sh` | `--compare-only` 加修复后的运行包文件名和同一个 `fl2va-ID` | `Size`、`Length`、`Average SSIM` |

取 `RUNTIME_ARCHIVE` 的最后一个路径分量作为运行包文件名。取 `RESULT_DIR` 的最后一个路径分量作为 `fl2va-ID`。如果生成作业已保存潜变量后因解码显存不足而退出，独立解码作业可直接读取该存档。

参考视频对比复用该结果目录的 `infinilm_fl2va.mp4`。如果组委会提供相同输入和参数的参考 MP4，将它放在个人 AFS 根目录并在对比参数末尾追加文件名。否则对比入口使用上游生成脚本生成 PyTorch 参考视频。参考路径使用 Qwen3-VL eager 注意力及分块计算的 Math SDPA。每次去噪迭代后保存潜变量和当前步数，日志出现 `REFERENCE_RESUBMIT_SAME_ARGS` 时，用同样的参数继续提交。两份视频齐备后，官方对比结果写到 `compare_fl2va.log`。

## 实测记录

长序列 BF16 Softmax 在每卡 `7 × 27570 × 27570` 的形状通过 64 位索引验证。正式生成完成 49 次去噪迭代，进度日志用时 7 分 08 秒。独立 VAE 解码与导出用时 67.693 秒，两项计时各有独立口径。完整运行记录的 `use_ntops` 为 `False`。对比结果以官方脚本的实测日志为准。

[赛题说明](https://www.infinitensor.com/news/news-72) · [InfiniLM FL2VA 对比脚本](https://github.com/InfiniTensor/InfiniLM/blob/minimax_h3/examples/minimax_h3_compare.py)
