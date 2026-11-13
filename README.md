# 嵩山杯 MiniMax H3 FL2VA 八卡复现

「Anna说的都队」在 InfiniFabric 的 `scnet-dcu` 环境完成了赛事首帧样例的八卡生成。文本编码器和 Transformer 由 InfiniLM 在单节点做八路张量并行，独立作业读取去噪存档，用 Diffusers 的视频和音频 VAE 导出 MP4。视频经逐帧解码验证，共 124 帧，有一条音轨。

## 源码和附件

- [`sources/InfiniCore/`](sources/InfiniCore) 和 [`sources/InfiniLM-minimax_h3-full/`](sources/InfiniLM-minimax_h3-full) 是本次构建所用源码快照。InfiniCore 基线为 `f32e2fbb2a9b2ca912b5711f4f3836c403196a74`，InfiniLM 基线为 `65deb91f840ebaef752d028fad2f3217ce0950f4`。所用源码的 `spdlog` 和 `json` 头文件随对应源码提供。
- [`patches/`](patches) 列出作业实际应用的改动，包含海光 MHA、Softmax 长序列索引、离线构建配置、Diffusers 解码和 PyTorch 参考续跑。批处理脚本会自动应用这些改动，按下面步骤运行即可。
- [`offline/`](offline) 提供构建和推理用的四个离线包以及 SHA-256 校验清单。`minimax-h3-offline-bootstrap.tar.gz` 内附脚本实际解包的两份源码归档。`first_frame.png` 是赛事首帧原图。
- [`evidence/compare-fl2va-excerpt.txt`](evidence/compare-fl2va-excerpt.txt) 是原八卡作业对比结果的三行摘录。

## 环境和输入

- 计算环境为 InfiniFabric `scnet-dcu`，版本 `524820f2`，8 张海光 BW DCU，架构 `gfx936`。Python 3.11.9，PyTorch 2.9.0，HIP 6.3.26093，DTK 26.04，XMake 3.0.5。
- `IFAB_AFS_HOME` 由平台设置，编译和推理分别用临时目录，日志、运行包、潜变量和视频写入个人 AFS 的 `minimax-h3-work/`。共享权重路径为 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`。
- 将赛事报告模板的原始 PNG 放在项目根目录，文件名 `first_frame.png`，尺寸 800 × 965，SHA-256 `b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54`。提示词 `A dog falls asleep`，随机种子 42，50 个采样点，124 帧。

## 离线软件包

从本仓库按报告所列 commit 下载源码。可在 GitHub 打开该提交对应的文件页，选择 **Code → Download ZIP**，并解压到本地。`offline/` 内含 `minimax-h3-offline-bootstrap.tar.gz`、`h3-fl2va-wheels-part1.tar.gz`、`h3-fl2va-wheels-part2.tar.gz`、`h3-fl2va-compat.tar.gz` 及 `SHA256SUMS`。

在个人 AFS 根目录解包 `minimax-h3-offline-bootstrap.tar.gz`，得到 `h3-offline-bootstrap/`。此目录包括 InfiniCore 源码 `f32e2fbb2a9b2ca912b5711f4f3836c403196a74`（含构建所需子模块）、`InfiniLM-minimax_h3-full.tar.gz`、Linux x86_64 XMake 3.0.5 和编译用 Python wheels。InfiniLM 源码包 SHA-256 为 `0a06f8d95855ccfc595d187aa742a76d3bb4ae343c48011c26eb34f79183bc05`。

将 `h3-fl2va-wheels-part1.tar.gz`、`h3-fl2va-wheels-part2.tar.gz` 和 `h3-fl2va-compat.tar.gz` 原样放在个人 AFS 根目录。推理入口会校验和解包。关键依赖为 Diffusers 0.40.0、Transformers 5.12.1、Accelerate 1.13.0、Torchvision 0.24.0+cpu、NumPy 1.26.4、PyAV 16.1.0 和 scikit-image 0.25.2。

在本地从仓库根目录校验待上传的归档。

```bash
(cd offline && shasum -a 256 -c SHA256SUMS)
```

在 InfiniFabric 将四个 `.tar.gz` 上传至当前账号的个人 AFS 根目录（`$IFAB_AFS_HOME`）。新建项目，在项目根目录上传 `reproduce_fl2va.sh`、`build_h3_batch.sh`、`run_fl2va_batch.sh`、`run_fl2va.sh` 和 `first_frame.png`。在项目中创建 `offline/SHA256SUMS`，粘贴仓库中同名文件的内容并保存。

打开该项目的登录工作区。在项目目录确认赛方共享模型可读，再校验并解包依赖。此步骤使用登录节点。

```bash
test -r /public/home/aclsylqidf/shared/MiniMax-H3/diffusers/model_index.json &&
  bash reproduce_fl2va.sh review01 prepare
```

### 使用另一个模型目录

赛方可提供同一 Diffusers 模型目录的其他挂载路径。在可联网且有足够存储空间的环境中，也可从 [MiniMax 官方仓库](https://huggingface.co/MiniMaxAI/MiniMax-H3) 下载 FL2VA 所用的 Diffusers 文件。以下命令使用 Hugging Face CLI `hf`，将权重放到计算节点可读的持久存储。按运行环境填写 `MODEL_DIR` 的实际绝对路径。

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 0919fb684102ef115126612f3fbb84d1027e7317 \
  --include model_index.json modular_model_index.json 'processor/*' \
    'tokenizer/*' 'text_encoder/*' 'transformer/*' 'vae/*' 'audio_vae/*' \
    'scheduler/*' 'audio_scheduler/*' \
  --local-dir "$MODEL_DIR"
bash reproduce_fl2va.sh review01 prepare "$MODEL_DIR"
```

指定模型目录时，请用新的运行编号开始 `prepare`，并在后续作业中沿用这个编号。`prepare` 检查模型索引和组件目录，将路径写入个人 AFS。生成、解码和对比作业都读取这个路径。官方权重的文件布局已核对，八卡实测记录使用的是赛方共享权重。

## 批处理入口

选择 `reproduce_fl2va.sh` 作为批作业入口。按以下顺序分别提交，每份作业时限 1800 秒。`core`、`lm` 和 `softmax` 使用 `scnet-dcu` 的 `gfx936` 环境，可申请 2 卡。`softmax` 的算子验证在第 0 张卡上运行。`generate` 和 `compare` 申请单节点 8 卡，`decode` 使用第 0 张卡。程序参数只填写脚本名后的内容，例如 `review01 core`。同一轮使用同一个自选运行编号，重做整轮时换一个编号。日志出现 `STAGE_COMPLETE=` 后再提交下一阶段。参考视频分段续跑时会显示 `REFERENCE_RESUBMIT_SAME_ARGS`，此时重复提交 `review01 compare`，直到显示 `STAGE_COMPLETE=compare`。

```bash
bash reproduce_fl2va.sh review01 core
bash reproduce_fl2va.sh review01 lm
bash reproduce_fl2va.sh review01 softmax
bash reproduce_fl2va.sh review01 generate
bash reproduce_fl2va.sh review01 decode
bash reproduce_fl2va.sh review01 compare
```

入口根据实际日志保存产物文件名，并在下一阶段读取。状态与逐阶段日志保存在 `$IFAB_AFS_HOME/minimax-h3-work/reproduce/<运行编号>/`。生成阶段如遇 VAE 显存不足，脚本检查已保存的潜变量后允许独立解码。入口内部调用的原八卡脚本如下。

| 阶段 | 入口 | 程序参数 | 日志标记 |
| --- | --- | --- | --- |
| 编译 InfiniCore | `build_h3_batch.sh` | `core` | `NEXT_ARGS=` |
| 编译 InfiniLM | `build_h3_batch.sh` | `lm` 加 `NEXT_ARGS` 给出的 Core 归档名 | `RUNTIME_ARCHIVE=` |
| 重编 Softmax | `build_h3_batch.sh` | `softmax` 加上一步给出的运行包文件名 | `H3_SOFTMAX_LONG_INDEX_OK` 和新的 `RUNTIME_ARCHIVE=` |
| 生成潜变量 | `run_fl2va_batch.sh` | `--generate-only` 加修复后的运行包文件名 | `VAE_INPUT_CHECKPOINT=` |
| 独立解码 | `run_fl2va_batch.sh` | `--decode-only` 加结果目录名 `fl2va-ID` | `VIDEO_FRAMES=124 AUDIO_STREAMS=1` 和 `VIDEO_SAVED=` |
| 参考视频对比 | `run_fl2va_batch.sh` | `--compare-only` 加修复后的运行包文件名和同一个 `fl2va-ID` | `Size`、`Length`、`Average SSIM` |

取 `RUNTIME_ARCHIVE` 的最后一个路径分量作为运行包文件名。取 `RESULT_DIR` 的最后一个路径分量作为 `fl2va-ID`。如果生成作业已保存潜变量后因解码显存不足而退出，独立解码作业可直接读取该存档。

在平台项目页的「程序参数」中填实际文件名或目录名。前两个阶段产物以本轮日志为准。本次已测推理的运行包名是 `runtime-Y15Y6M.tar.gz`，结果目录名是 `fl2va-XWBoHy`。新账号从 `core` 开始，使用自己本轮的产物名。

参考视频对比复用该结果目录的 `infinilm_fl2va.mp4`。如果组委会提供相同输入和参数的参考 MP4，将它放在个人 AFS 根目录并在对比参数末尾追加文件名。否则对比入口使用上游生成脚本生成 PyTorch 参考视频。参考路径使用 Qwen3-VL eager 注意力及分块计算的 Math SDPA。每次去噪迭代后保存潜变量和当前步数，日志出现 `REFERENCE_RESUBMIT_SAME_ARGS` 时，用同样的参数继续提交。两份视频齐备后，官方对比结果写到 `compare_fl2va.log`。

## 实测记录

长序列 BF16 Softmax 在 `7 × 27570 × 27570` 的形状通过 64 位索引验证。正式生成完成 49 次去噪迭代，进度日志用时 7 分 08 秒。独立 VAE 解码与导出用时 67.693 秒，两项计时各有独立口径。海光 MHA 使用 C++ 组合实现，运行记录的 `use_ntops` 为 `False`。原八卡样例尺寸 768×928、帧数 124 均通过对比，平均 SSIM 为 5.403%。

上述实测数据来自原八卡作业。验收时从报告所列 GitHub 提交下载文件，在新项目中使用新运行编号，从 `prepare` 跑到 `STAGE_COMPLETE=compare`。检查本轮的 `infinilm_fl2va.mp4` 和 `compare_fl2va.log`，记录视频尺寸、帧数和平均 SSIM。

[赛题说明](https://www.infinitensor.com/news/news-72) · [InfiniLM FL2VA 对比脚本](https://github.com/InfiniTensor/InfiniLM/blob/minimax_h3/examples/minimax_h3_compare.py)
