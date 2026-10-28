# Qwen3-ASR-1.7B on BW + native vLLM（Session010 技术报告）

## 1. 简介

T04 · 已完成发布基线数值对照、完整原始证据独立复算，以及从已保存镜像新建实例的完整复现。技术投递材料已整理；主办方验收和最终赛事表单提交尚未发生。

代码：[光合 session010](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-qwen3-asr-bw-vllm-session010)。镜像：[guoxin-publisher-eval-bw009](https://www.scnet.cn/ui/console/index.html#/image-management/detail/2114587646006431746)，版本 `dtk2604-vllm018-asr-trident-20261026`。模型固定 revision：`7278e1e70fe206f11671096ffdd38061171dd6e5`。源码、上游许可、失败轮次和原始报告均保留。

## 2. 验证环境

华中一区 A 区，BW 64 GB × 1、15 CPU、59 GB 内存；DTK 26.04、Ubuntu 22.04、Python 3.10。新实例实测 Torch 2.10.0、HIP 6.3.26093、vLLM 0.18.1+das.fa71803.dtk2604、Transformers 5.5.0。平台镜像继承的 PyTorch 2.9.0 标签不代表实际运行版本。

独立实例 ID `2114808632296972290`，容器 `crdnotebook-2114808632296972290-ac3p4mowhz-24051`；镜像 ID `2114587646006431746`。模型、环境、数据分别位于镜像的 `/opt/guoxin-asr009`。实例从镜像新建，完整运行日志记录于 UTC 2026-10-26T20:09:34.937352+00:00 至 2026-10-26T20:31:55.923907+00:00。

## 3. 适配优化方案

使用原生 vLLM、HCU 插件与任务内 Transformers 5.5 兼容注册；保留原 checkpoint、tokenizer、BF16 和 greedy 解码。本次增加交付验证和报告，没有修改已测 serving 的 Python / shell 代码。

本轮逐项核验原始仓库 1075 个清单文件；所有原始 Python / shell 文件与已测 session009 版本一致，见 `evidence/replay010/unchanged-serving-code.json`。

## 4. 服务启动脚本

在指定镜像中进入本仓库根目录执行，输出目录使用新的唯一名称：

```bash
source env.sh
RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
bash env_check.sh
"$PYTHON_BIN" verify.py --selfcheck
bash run.sh --run-id "$RUN_ID"
bash smoke.sh
"$PYTHON_BIN" verify.py --output /root/verify-asr-new-unique
bash bench.sh --output /root/bench-asr-new-unique
bash stop.sh --run-id "$RUN_ID"
```

服务仅监听本实例回环地址；启动与停止使用同一运行标识。新实例实际执行的环境检查、自检、启动、smoke、全量评测和停止阶段均返回 0；ASR 另有独立 SSE 和 bench 阶段。复现驱动及逐阶段返回值保存在 `evidence/replay010/`。

## 5. Smoke 测试

新实例的 smoke 返回正常；另测实际 SSE 增量输出，收到 13 个事件及 `[DONE]`，以 `stop` 正常结束。该测试输入是一段完整音频，验证的是输出流式，不扩称分块实时音频输入。

完整准确率与四轮性能均零请求失败。原件 `results.zip` 中保存实际响应，未使用假模型或固定返回值。

## 6. 精度测评

完整 LibriSpeech test-clean 2,620 条。

原始轮与新实例复现的 WER 均为 **1.638787787354%**。发布基线 1.63%，按相对劣化 2% 的上限为 **1.6626%**，本次相对劣化 **0.539128%**，通过。

新实例共 53,027 个参考词、869 个编辑错误：656 替换、144 删除、69 插入。全部 2,620 条原音频 SHA、原始返回、规范化文本及错误距离已用独立 NLTK 实现复算；不删除失败或高错误率样本。

发布方未公开完整文本归一化实现。本轮在评测前固定 OpenAI Whisper EnglishTextNormalizer，并保留 MIT 许可、原文、规范化结果和源码 SHA；不宣称与未公开实现逐字节相同。

依据：[Qwen 固定模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/README.md)、[OpenSLR LibriSpeech](https://www.openslr.org/12/)。独立核验结果为 `evidence/replay010/independent-audit.json`。评测器原始 JSON 中的 `submission_ready=false` 是评测阶段的保守范围字段，原样保留；本次完成的工程检查以 `DELIVERY_STATUS_010.json` 和新实例回执为准，不将它改写成主办方验收。

## 7. 性能参考

固定哈希选取 24 个完整输入，c1/c4 各两轮；共 96/96 请求成功。沿用运行前声明的第二轮作为主要对照，两轮原件全部保留。单位为 音频秒/墙钟秒（RTFX），包含端到端 HTTP 开销。

| 并发 | 轮次 | 成功/请求 | 墙钟秒 | 速率 | p50 秒 | p95 秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 24/24 | 21.420252 | 6.866866 | 0.873765 | 1.590581 |
| 1 | 2 | 24/24 | 21.005854 | 7.002334 | 0.865703 | 1.542596 |
| 4 | 1 | 24/24 | 5.803966 | 25.343014 | 0.888057 | 1.564420 |
| 4 | 2 | 24/24 | 5.819554 | 25.275133 | 0.889590 | 1.573136 |

第二轮速率与原始轮比较：c1 变化 +10.547%; c4 变化 +3.534%，均在 20% 以内。该比较用于复现一致性，不作为已获得后续优化奖励的声明。

## 8. 注意事项

- 每个模型单独投递本报告及对应光合、镜像链接；当前没有最终赛事提交回执，也没有主办方验收回执。
- 两项代码库保持私有；账号内可访问与评审账号已授权是不同事实，评审访问尚未独立验证。
- 新实例完整复现已完成，服务已停止；本轮平台全部实例也已关机，证据见 `evidence/replay010/all-gpu-stopped.png`。
- 新编代码继续保留版权说明；上游许可保留，不擅自替换许可证。
- 原始 session009 资料以历史版本保存在 `provenance/frozen009/`；本报告及 `publisher-delivery-manifest010.json` 是本轮材料入口。
