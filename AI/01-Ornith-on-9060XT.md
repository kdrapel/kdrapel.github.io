# Tuning Ornith-1.5-35B-A3B on a 16 GB Radeon RX 9060 XT with llama.cpp

I spent some time tuning **Ornith-1.5-35B-A3B** using the **Vulkan build of `llama.cpp` master**, at commit `cc231cb0da565440cf6a3e5b55dfeba477972cb6` (build 10701), on a **16 GB AMD Radeon RX 9060 XT**, with a particular focus on finding a configuration that remains fast and stable at long context sizes.
On this setup, performance shows a very sharp VRAM-related threshold. Once that threshold is crossed, prefill speed can collapse even though the model still technically fits in VRAM.
The final configuration I settled on is conservative rather than the absolute fastest at 32k/64k, because I sometimes use contexts close to **120k tokens**.

Important: this is the version without the MTP. I will prepare another article about this.

---

## Model and backend

- **Model:** `Ornith-1.5-35B-A3B-AD-IQ3_S-IQ3_XXS.gguf` (AtomicChat quants)
- **Model size:** ~14.44 GiB
- **Parameters:** 34.66B total, 8 experts active per token
- **Architecture:** Qwen3.5 MoE hybrid architecture
- **Backend:** `llama.cpp` Vulkan build
- **llama.cpp revision:** master commit `cc231cb0da565440cf6a3e5b55dfeba477972cb6` (build 10701)
- **GPU:** AMD Radeon RX 9060 XT, 16 GB VRAM
- **CPU:** AMD Ryzen 7 7800X3D
- **Context size:** 131,072
- **Parallel slots:** 1
- **Flash Attention:** enabled

The model has 40 layers and 256 experts, with 8 experts active per token. Only 10 layers use full attention, so the KV cache is much smaller than it would be for a conventional 40-layer full-attention transformer.

---

## Benchmark command

I used [`llama-benchy`](https://github.com/william-murray1204/llama-benchy) against `llama-server`.

Example for the 120k test:

```powershell
uvx llama-benchy --base-url http://localhost:9090/v1 \
  --model Ornith-1.5-35B-A3B \
  --depth 120000 \
  --latency-mode none \
  --pp 0 \
  --tg 256 \
  --runs 1
```

Most of the tuning runs below used one run per point, so the results should be treated as practical tuning measurements rather than statistically rigorous benchmarks.

---

## What I tuned

The main parameters explored were:

- `--n-cpu-moe`
- `-b`
- `-ub`
- KV cache precision (`q8_0` vs `q4_0`)
- context depth up to 120k

The other important parameters were kept stable unless explicitly noted.

---

# 1. Baseline

Initial configuration:

```text
--n-cpu-moe 26
-fa on
-t 8
-b 2048
-ub 1024
--cache-type-k q8_0
--cache-type-v q8_0
--ctx-size 131072
```

Results:

| Test | Throughput |
|---|---:|
| Prefill @ 32k | 1084.13 t/s |
| Decode @ 32k | 29.55 t/s |
| Prefill @ 64k | 862.53 t/s |
| Decode @ 64k | 26.47 t/s |

This became the reference point for the later experiments.

---

# 2. Large batches were worse

I first tried increasing the batch sizes substantially.

### `--n-cpu-moe 20`, `-b 16384`, `-ub 2048`

| Test | Result |
|---|---:|
| Prefill @ 32k | 918.71 t/s |
| Decode @ 32k | 32.35 t/s |
| Prefill @ 64k | 742.62 t/s |
| Decode @ 64k | 29.96 t/s |

A repeat with `-t 8` produced:

| Test | Result |
|---|---:|
| Prefill @ 32k | 905.39 t/s |
| Decode @ 32k | 33.26 t/s |
| Prefill @ 64k | 744.78 t/s |
| Decode @ 64k | 30.97 t/s |

The larger batches improved decode somewhat but hurt prefill badly.

Returning to:

```text
-b 2048
-ub 1024
```

with `--n-cpu-moe 20` gave:

| Test | Result |
|---|---:|
| Prefill @ 32k | 1019.30 t/s |
| Decode @ 32k | 33.76 t/s |
| Prefill @ 64k | 816.66 t/s |
| Decode @ 64k | 32.11 t/s |

So for this model/backend combination, **very large batches were clearly counterproductive**.

---

# 3. Moving more MoE experts to the GPU

Reducing `--n-cpu-moe` moves more expert weights from host memory to GPU memory.

## `--n-cpu-moe 14`, Q8 KV

```text
--n-cpu-moe 14
-b 2048
-ub 1024
--cache-type-k q8_0
--cache-type-v q8_0
```

Results:

| Test | Result |
|---|---:|
| Prefill @ 32k | **1089.55 t/s** |
| Decode @ 32k | **34.45 t/s** |
| Prefill @ 64k | **860.16 t/s** |
| Decode @ 64k | **31.59 t/s** |

This was a very good balanced result: baseline-like prefill with substantially faster decode.

---

# 4. The first performance cliff: `--n-cpu-moe 10` + Q8 KV

I then reduced `--n-cpu-moe` to 10 while keeping the Q8 KV cache.

```text
--n-cpu-moe 10
-b 2048
-ub 1024
--cache-type-k q8_0
--cache-type-v q8_0
```

Results:

| Test | Result |
|---|---:|
| Prefill @ 32k | **277.71 t/s** |
| Decode @ 32k | 29.02 t/s |
| Prefill @ 64k | **261.35 t/s** |
| Decode @ 64k | 27.67 t/s |

Prefill collapsed by roughly 70-75%.

At first this looked like `--n-cpu-moe 10` itself was simply too aggressive. It turned out that conclusion was wrong.

---

# 5. Q4 KV completely changed the result

I repeated `--n-cpu-moe 10`, but switched the KV cache from Q8 to Q4:

```text
--n-cpu-moe 10
-b 2048
-ub 1024
--cache-type-k q4_0
--cache-type-v q4_0
```

The result was dramatically different:

| Test | Q8 KV | Q4 KV |
|---|---:|---:|
| Prefill @ 32k | 277.71 | **1134.53 t/s** |
| Decode @ 32k | 29.02 | **37.49 t/s** |
| Prefill @ 64k | 261.35 | **910.70 t/s** |
| Decode @ 64k | 27.67 | **39.42 t/s** |

This was the fastest 32k/64k configuration tested.

### Why did Q4 matter so much?

At `--n-cpu-moe 10`, the model placement was approximately:

```text
Vulkan0 model buffer : 11207.78 MiB
Vulkan Host buffer   :  3575.31 MiB
```

With Q8 KV, the full 131k KV allocation was roughly:

```text
KV cache : 1360 MiB
K q8_0   :  680 MiB
V q8_0   :  680 MiB
```

With Q4 KV:

```text
KV cache : 720 MiB
K q4_0   : 360 MiB
V q4_0   : 360 MiB
```

That recovered about **640 MiB of VRAM**.

The resulting performance difference was enormous. This strongly suggests that the system had crossed a **VRAM / memory-management performance threshold**, rather than simply running out of memory.

The model could technically fit with Q8 KV, but it ran much more slowly.

---

# 6. Pushing further: `--n-cpu-moe 6`

With Q4 KV freeing additional memory, I tried:

```text
--n-cpu-moe 6
-b 2048
-ub 1024
--cache-type-k q4_0
--cache-type-v q4_0
```

It immediately crossed the cliff again.

Live cumulative prefill measurements were already poor:

```text
14k  ~347.6 t/s
16k  ~342.2 t/s
18k  ~336.7 t/s
20k  ~331.6 t/s
```

There was no reason to continue treating this as a candidate configuration.

The useful conclusion was that the performance curve is **not smooth**. There is a relatively narrow range where more GPU-resident experts help, followed by a sudden collapse.

---

# 7. Why I chose `--n-cpu-moe 14`

Although `--n-cpu-moe 10 + Q4 KV` was faster at 32k and 64k, I occasionally use contexts approaching 120k.

I therefore chose the more conservative:

```text
--n-cpu-moe 14
```

This leaves substantially more VRAM headroom and has shown very predictable scaling at long context lengths.

With Q4 KV and `--n-cpu-moe 14`, llama.cpp reported approximately:

```text
GPU model       :  9983 MiB
Context         :   782 MiB
Compute         :   512 MiB
Accounted total : 11278 MiB
Projected free  :  4036 MiB
```

That is much more comfortable on a 16 GB card.

---

# 8. Long-context performance

Final long-context configuration:

```text
--n-cpu-moe 14
-fa on
-t 8
-b 2048
-ub 1024
--cache-type-k q4_0
--cache-type-v q4_0
--ctx-size 131072
```

The cumulative prefill speed decreased smoothly as context grew.

```text
Context    Cumulative prefill
--------------------------------
   6k       ~1499 t/s
   8k       ~1450 t/s
  10k       ~1407 t/s
  12k       ~1368 t/s
  14k       ~1331 t/s
  16k       ~1299 t/s
  18k       ~1270 t/s
  20k       ~1242 t/s
  22k       ~1215 t/s
  24k       ~1189 t/s
  26k       ~1165 t/s
  28k       ~1142 t/s
  30k       ~1120 t/s
  32k       ~1097 t/s
  35k       ~1077 t/s
  37k       ~1056 t/s
  39k       ~1037 t/s
  41k       ~1019 t/s
  43k       ~1001 t/s
  45k        ~983 t/s
  47k        ~966 t/s
  49k        ~950 t/s
  51k        ~935 t/s
  53k        ~919 t/s
  55k        ~905 t/s
  57k        ~890 t/s
  59k        ~876 t/s
  61k        ~863 t/s
  63k        ~849 t/s
  65k        ~836 t/s
  68k        ~823 t/s
  70k        ~809 t/s
  72k        ~796 t/s
  74k        ~784 t/s
  76k        ~771 t/s
  78k        ~760 t/s
  80k        ~749 t/s
  82k        ~738 t/s
  84k        ~727 t/s
  86k        ~717 t/s
  88k        ~707 t/s
  90k        ~698 t/s
  92k        ~689 t/s
  94k        ~680 t/s
  96k        ~672 t/s
  98k        ~664 t/s
 100k        ~655 t/s
 103k        ~644 t/s
```

The important point is the shape of the curve: **there is no abrupt performance cliff**.

## 120k benchmark

Command:

```powershell
uvx llama-benchy --base-url http://localhost:9090/v1 --model Ornith-1.5-35B-A3B --depth 120000 --latency-mode none --pp 0 --tg 256 --runs 1
```

Result:

| Test | Throughput | Peak | TTFR |
|---|---:|---:|---:|
| Prefill @ 120k | **633.35 t/s** | - | 162069.64 ms |
| Decode @ 120k | **31.05 t/s** | **32.00 t/s** | - |

That corresponds to roughly **162 seconds / 2 min 42 s** of prompt processing for the benchmark workload.

Decode speed remains remarkably stable at long context.

For comparison:

| Context | Prefill | Decode |
|---:|---:|---:|
| 32k, `moe 14 + q8` | 1089.55 t/s | 34.45 t/s |
| 64k, `moe 14 + q8` | 860.16 t/s | 31.59 t/s |
| 120k, `moe 14 + q4` | **633.35 t/s** | **31.05 t/s** |

The prefill cost grows as expected, while token generation remains around 31 t/s even near 120k.

---

# 9. uBatch tuning at 120k

I also tested different `-ub` values while keeping:

```text
--n-cpu-moe 14
-b 2048
Q4 KV
120k benchmark depth
```

Results:

| `-ub` | Prefill @ 120k | Decode @ 120k | Peak decode | TTFR |
|---:|---:|---:|---:|---:|
| 512 | 568.52 t/s | **31.24 t/s** | 32.00 | 180.56 s |
| **1024** | **633.35 t/s** | 31.05 t/s | **32.00** | **162.07 s** |
| 1536 | 579.94 t/s | 30.29 t/s | 31.00 | 176.90 s |

`-ub 1024` was clearly the best overall setting.

Compared with `1024`:

- `512` lost about **10.2% prefill throughput**
- `1536` lost about **8.4% prefill throughput**
- decode differences were comparatively small

So the useful sweet spot on this system is:

```text
-b 2048
-ub 1024
```

---

# Complete run summary

| `n-cpu-moe` | KV | `b / ub` | PP 32k | TG 32k | PP 64k | TG 64k | PP 120k | TG 120k | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 26 | Q8 | 2048 / 1024 | 1084.13 | 29.55 | 862.53 | 26.47 | - | - | Baseline |
| 20 | Q8 | 16384 / 2048 | 918.71 | 32.35 | 742.62 | 29.96 | - | - | Large batch |
| 20 | Q8 | 16384 / 2048 | 905.39 | 33.26 | 744.78 | 30.97 | - | - | Repeat with `t=8` |
| 20 | Q8 | 2048 / 1024 | 1019.30 | 33.76 | 816.66 | 32.11 | - | - | Better balanced |
| 14 | Q8 | 2048 / 1024 | 1089.55 | 34.45 | 860.16 | 31.59 | - | - | Strong Q8 result |
| 10 | Q8 | 2048 / 1024 | 277.71 | 29.02 | 261.35 | 27.67 | - | - | Performance cliff |
| 10 | Q4 | 2048 / 1024 | **1134.53** | **37.49** | **910.70** | **39.42** | - | - | Fastest 32k/64k |
| 6 | Q4 | 2048 / 1024 | - | - | - | - | - | - | Cliff already visible by ~20k |
| 14 | Q4 | 2048 / 512 | - | - | - | - | 568.52 | 31.24 | Slower prefill |
| **14** | **Q4** | **2048 / 1024** | - | - | - | - | **633.35** | **31.05** | **Final long-context config** |
| 14 | Q4 | 2048 / 1536 | - | - | - | - | 579.94 | 30.29 | Slower than ub=1024 |

All throughput values are tokens/second.

---

# Final configuration

This is the `llama-server` command line I currently use:

```powershell
llama-server.exe -m "D:\40-49 Assets\44 AI Models\44.99 Misc\Ornith-1.5-35B-A3B-AD-IQ3_S-IQ3_XXS.gguf" -np 1 -lv 4 -ngl 999 --n-cpu-moe 14 -fa on -t 8 -b 2048 -ub 1024 --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --ctx-size 131072 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --reasoning on --jinja -n -1 --host 127.0.0.1 --port 9090 --alias Ornith-1.5-35B-A3B
```

For readability, the same command split across lines in PowerShell is:

```powershell
llama-server.exe `
  -m "D:\40-49 Assets\44 AI Models\44.99 Misc\Ornith-1.5-35B-A3B-AD-IQ3_S-IQ3_XXS.gguf" `
  -np 1 `
  -lv 4 `
  -ngl 999 `
  --n-cpu-moe 14 `
  -fa on `
  -t 8 `
  -b 2048 `
  -ub 1024 `
  --no-mmap `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --ctx-size 131072 `
  --temp 0.6 `
  --top-p 0.95 `
  --top-k 20 `
  --min-p 0.0 `
  --presence-penalty 0.0 `
  --repeat-penalty 1.0 `
  --reasoning on `
  --jinja `
  -n -1 `
  --host 127.0.0.1 `
  --port 9090 `
  --alias Ornith-1.5-35B-A3B
```

> **Note:** In the llama.cpp build used for these tests, `--mmap` / `--no-mmap` is reported as deprecated in favor of the newer load-mode option. The command above preserves the exact option used during the benchmark runs for reproducibility.

---

# Conclusions

A few things stood out from these experiments.

### 1. VRAM headroom matters even before OOM

The most important result was the enormous difference between `--n-cpu-moe 10` with Q8 and Q4 KV.
The Q8 configuration technically fit, but prefill collapsed to around 260-280 t/s. Freeing only ~640 MiB by switching the KV cache to Q4 restored performance to more than 1100 t/s at 32k.
So on this Vulkan/RDNA4 setup, **"it fits in VRAM" is not enough**. Leaving adequate headroom can have a major performance impact.

### 2. More GPU offload is not monotonically better

Moving more experts to the GPU improved performance until a threshold was crossed. Beyond that point, performance collapsed instead of degrading gradually.
This means tuning `--n-cpu-moe` should be done empirically rather than simply choosing the lowest value that loads successfully.

### 3. Q4 KV is very useful for long context

For this hybrid architecture, Q4 KV keeps the full 131k KV cache to roughly 720 MiB, versus about 1360 MiB with Q8.
That extra VRAM margin is particularly valuable when trying to keep more MoE experts resident on the GPU.

### 4. `-ub 1024` is a strong sweet spot

Both `512` and `1536` were slower at 120k. On this system, `-ub 1024` gave the best prefill throughput by a clear margin.

### 5. Optimize for the context size you actually use

`--n-cpu-moe 10 + Q4` produced the fastest 32k/64k results, but I prefer `--n-cpu-moe 14` because I sometimes work near 120k context.
The slightly more conservative expert placement gives more VRAM headroom and a very smooth long-context performance curve.
For my workload, **predictable 120k behavior is more useful than extracting the last few percent at 32k**.

## Final result

On a 16 GB RX 9060 XT, the conservative configuration achieves approximately:

```text
32k class prefill : ~1,100 t/s
64k class prefill : ~850-900 t/s
120k prefill      : 633 t/s
120k decode       : 31 t/s
```

For a 35B MoE model running locally on a 16 GB consumer GPU, that is a usable result.
