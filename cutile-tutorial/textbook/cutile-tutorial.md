# cuTile 编程教程 / cuTile Programming Tutorial

> 双语版本 — Bilingual Edition
>
> cuTile 是 NVIDIA 推出的基于 Python 的 GPU 编程库，采用 Tile（分块）抽象模型，让开发者无需编写底层 CUDA C++ 即可高效利用 GPU 硬件。
>
> cuTile is a Python-based GPU programming library from NVIDIA that uses a Tile abstraction model, enabling developers to efficiently utilize GPU hardware without writing low-level CUDA C++.

---

## 目录 / Table of Contents

1. [向量加法 / Vector Addition](#第-1-章向量加法--chapter-1-vector-addition)
2. [执行模型 / Execution Model](#第-2-章执行模型--chapter-2-execution-model)
3. [数据模型 / Data Model](#第-3-章数据模型--chapter-3-data-model)
4. [矩阵转置 / Matrix Transpose](#第-4-章矩阵转置--chapter-4-matrix-transpose)
5. [进阶技巧 / Advanced Tips](#第-5-章进阶技巧--chapter-5-advanced-tips)
6. [性能调优：Tile 大小 / Performance Tuning: Tile Size](#第-6-章性能调优tile-大小--chapter-6-performance-tuning-tile-size)
7. [注意力机制 / Attention Mechanism](#第-7-章注意力机制--chapter-7-attention-mechanism)

---

## 环境准备 / Environment Setup

运行本教程需要以下环境：

To run this tutorial, the following environment is required:

- NVIDIA GPU（Blackwell 架构，计算能力 10.x 或 12.x / Blackwell architecture, compute capability 10.x or 12.x）
- NVIDIA 驱动 >= r580
- CUDA Toolkit >= 13.1
- Python 3.11 + 虚拟环境 / Python 3.11 + virtual environment

```bash
# 创建虚拟环境并安装依赖 / Create venv and install dependencies
bash setup-cutile-env.sh

# 验证环境 / Verify environment
.venv/bin/python 0-environment/check.py
```

所有示例均使用以下导入：

All examples use these imports:

```python
import cupy as cp        # GPU 数组操作 / GPU array operations
import numpy as np       # CPU 验证 / CPU verification
import cuda.tile as ct   # cuTile 库 / cuTile library
```

---

# 第 1 章：向量加法 / Chapter 1: Vector Addition

> 源码 / Source: `1-vectoradd/vectoradd.py`

## 概述 / Overview

向量加法是 GPU 编程的 "Hello World"。我们将两个数组 `a` 和 `b` 逐元素相加，结果存入 `c`。

Vector addition is the "Hello World" of GPU programming. We add two arrays `a` and `b` element-wise, storing the result in `c`.

## 核函数 / The Kernel

cuTile 中的核函数用 `@ct.kernel` 装饰器标记，在 GPU 上并行执行。

In cuTile, kernels are marked with the `@ct.kernel` decorator and execute in parallel on the GPU.

```python
@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # 获取当前 tile 的 ID / Get current tile's Block ID
    pid = ct.bid(0)

    # 从全局内存加载 tile / Load tiles from global memory
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # 逐元素相加（在 GPU 寄存器中并行执行）
    # Element-wise addition (executes in parallel on GPU registers)
    result = a_tile + b_tile

    # 将结果写回全局内存 / Store result back to global memory
    ct.store(c, index=(pid,), tile=result)
```

### 关键概念 / Key Concepts

**`ct.bid(0)`** — 块索引（Block ID）。`bid(0)` 返回当前 tile 在第 0 维度的索引。若启动了 256 个 tile，则 `bid(0)` 的范围为 0 到 255。

**`ct.bid(0)`** — Block ID. Returns the current tile's index in dimension 0. If 256 tiles are launched, `bid(0)` ranges from 0 to 255.

**`ct.load(array, index, shape)`** — 从全局内存（慢、大）加载数据到 Tile 内存（快、寄存器/共享内存）。`index` 是 tile 级别的索引，`shape` 是加载的数据形状。

**`ct.load(array, index, shape)`** — Loads data from global memory (slow, large) to tile memory (fast, registers/shared memory). `index` is tile-level indexing, `shape` is the tile shape to load.

**`ct.store(array, index, tile)`** — 将计算结果从 Tile 内存写回全局内存。

**`ct.store(array, index, tile)`** — Stores computed results from tile memory back to global memory.

**`ct.Constant[int]`** — 编译时常量。标记为 `ct.Constant` 的参数允许编译器进行循环展开和寄存器分配优化。

**`ct.Constant[int]`** — Compile-time constant. Parameters marked with `ct.Constant` allow the compiler to optimize loop unrolling and register allocation.

## 主机代码 / Host Code

主机代码在 CPU 上运行，负责准备数据和启动核函数：

The host code runs on the CPU, preparing data and launching the kernel:

```python
vector_size = 2**12   # 4096 个元素 / 4096 elements
tile_size = 32        # 每个 tile 处理 32 个元素 / 32 elements per tile

# 计算网格大小：需要多少个 tile 覆盖整个向量
# Grid calculation: how many tiles to cover the entire vector
grid = (ct.cdiv(vector_size, tile_size), 1, 1)  # ct.cdiv = 向上取整除法 / ceiling division

# 在 GPU 上分配数据（使用 CuPy）/ Allocate data on GPU (using CuPy)
a = cp.random.uniform(-1, 1, vector_size)
b = cp.random.uniform(-1, 1, vector_size)
c = cp.zeros_like(a)

# 启动核函数 / Launch kernel
ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, c, tile_size))
```

**`ct.cdiv(x, y)`** — 向上取整除法。$4096 / 32 = 128$，所以启动 128 个 tile。

**`ct.cdiv(x, y)`** — Ceiling division. $4096 / 32 = 128$, so we launch 128 tiles.

**`ct.launch(stream, grid, kernel, args)`** — 启动核函数。`stream` 用于同步，`grid` 指定并行维度，`args` 是传给核函数的参数。

**`ct.launch(stream, grid, kernel, args)`** — Launches the kernel. `stream` for synchronization, `grid` specifies parallel dimensions, `args` are kernel arguments.

## 验证 / Verification

```python
# 将 GPU 数据拷贝回 CPU 进行验证 / Copy GPU data back to CPU for verification
a_np, b_np, c_np = cp.asnumpy(a), cp.asnumpy(b), cp.asnumpy(c)
expected = a_np + b_np
np.testing.assert_array_almost_equal(c_np, expected)
```

```bash
.venv/bin/python 1-vectoradd/vectoradd.py
```

---

# 第 2 章：执行模型 / Chapter 2: Execution Model

> 源码 / Source: `2-execution-model/sigmoid_1d.py`, `2-execution-model/grid_2d.py`

## Grid 与 Tile / Grid and Tile

传统 CUDA 使用 Grid → Block → Thread 三级层次结构。cuTile 将抽象级别提升到 **Tile** 层面：你为一个"数据分块"编写代码，而非为单个线程编写。

Traditional CUDA uses a Grid → Block → Thread hierarchy. cuTile raises the abstraction to the **Tile** level: you write code for a "chunk of data" rather than individual threads.

- **Grid（网格）**：整个问题空间，可以是 1D、2D 或 3D 排列。
- **Tile（分块）**：计算的基本单位。每个 tile 独立处理一块数据。

- **Grid**: The entire problem space, arranged in 1D, 2D, or 3D.
- **Tile**: The fundamental unit of computation. Each tile independently processes a chunk of data.

## 1D 示例：Sigmoid 函数 / 1D Example: Sigmoid Function

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
@ct.kernel
def sigmoid_kernel(input, output, tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    x_tile = ct.load(input, index=(pid,), shape=(tile_size,))

    # cuTile 提供 ct.exp() 进行逐元素指数运算
    # cuTile provides ct.exp() for element-wise exponential
    exp_neg_x = ct.exp(-x_tile)
    sigmoid_tile = 1.0 / (1.0 + exp_neg_x)

    ct.store(output, index=(pid,), tile=sigmoid_tile)
```

此示例展示了 cuTile 的数学函数（`ct.exp`）以及标量与 tile 的混合运算。

This example demonstrates cuTile's math functions (`ct.exp`) and mixed scalar-tile operations.

## 2D 网格示例 / 2D Grid Example

处理二维数据（矩阵、图像）时，我们使用 2D 网格：

For 2D data (matrices, images), we use a 2D grid:

```python
# 128x128 的数组，16x16 的 tile / 128x128 array, 16x16 tiles
grid_x = ct.cdiv(width, tile_w)    # 8 个列 tile / 8 column tiles
grid_y = ct.cdiv(height, tile_h)   # 8 个行 tile / 8 row tiles
grid = (grid_x, grid_y, 1)        # 共 64 个 tile 并行执行 / 64 tiles run in parallel
```

在核函数内部获取坐标：

Inside the kernel, get coordinates:

```python
pid_x = ct.bid(0)  # 列索引 / column index
pid_y = ct.bid(1)  # 行索引 / row index
```

### 内存映射 / Memory Mapping

NumPy/CuPy 数组的形状是 `(rows, cols)` 即 `(y, x)`。因此访问内存时：

NumPy/CuPy arrays have shape `(rows, cols)` i.e. `(y, x)`. So when accessing memory:

```python
# index=(行索引, 列索引) / index=(row_idx, col_idx)
ct.load(data, index=(pid_y, pid_x), shape=(tile_h, tile_w))
```

## 要点 / Key Takeaways

1. **SIMT vs SIMD**：传统 CUDA 是 SIMT（单指令多线程），cuTile 更接近 SIMD（单指令多数据），其中"数据"是一个 Tile。
2. **坐标**：使用 `ct.bid(n)` 获取网格中第 n 维的坐标。
3. **并行性**：网格中的所有 tile 独立且可并行运行。

1. **SIMT vs SIMD**: Traditional CUDA is SIMT (Single Instruction, Multiple Threads). cuTile is more like SIMD (Single Instruction, Multiple Data) where "Data" is a Tile.
2. **Coordinates**: Use `ct.bid(n)` to get your coordinate in dimension n of the grid.
3. **Parallelism**: All tiles in the grid are independent and can run in parallel.

---

# 第 3 章：数据模型 / Chapter 3: Data Model

> 源码 / Source: `3-data-model/data_types.py`

## 编译时常量 / Compile-time Constants

```python
def mixed_precision_scale(..., tile_size: ct.Constant[int]):
```

**为什么需要 `ct.Constant`？**

GPU 编译器需要在**编译时**知道循环边界和数组大小，以便进行循环展开和寄存器分配优化。标记为 `ct.Constant` 的参数告诉 cuTile："我保证这个值在调用 `ct.launch` 时是已知的，不会改变。"

**Why `ct.Constant`?**

GPU compilers need to know loop bounds and array sizes at **compile time** for loop unrolling and register allocation optimizations. Marking a parameter as `ct.Constant` tells cuTile: "I promise this value will be known when I call `ct.launch` and it won't change."

## 混合精度 / Mixed Precision

深度学习和高性能计算经常使用 FP16（半精度）来节省内存和提高速度。cuTile 支持混合精度计算：

Deep learning and HPC often use FP16 (half precision) to save memory and increase speed. cuTile supports mixed precision:

```python
@ct.kernel
def mixed_precision_scale(input_ptr, output_ptr, scale_factor,
                          tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    # 1. 加载 FP16 数据 / Load FP16 data
    input_tile = ct.load(input_ptr, index=(pid,), shape=(tile_size,))

    # 2. 在 FP32 中计算（自动类型提升）/ Compute in FP32 (automatic promotion)
    result_fp32 = input_tile * scale_factor  # FP16 * FP32 → FP32

    # 3. 显式转换回 FP16 并存储 / Explicit cast back to FP16 and store
    result_fp16 = ct.astype(result_fp32, ct.float16)
    ct.store(output_ptr, index=(pid,), tile=result_fp16)
```

### 类型提升规则 / Type Promotion Rules

| 运算 / Operation | 结果 / Result | 说明 / Notes |
|-----------|--------|-------|
| `float16 * float32` | `float32` | 隐式提升（安全）/ Implicit promotion (safe) |
| `int32 + float32` | `float32` | 隐式提升（安全）/ Implicit promotion (safe) |
| `float32 → float16` | **错误 / Error** | 需要 `ct.astype()` / Requires `ct.astype()` |

**重要**：从高精度到低精度的转换必须使用 `ct.astype()` 显式完成，cuTile 不会自动降精度。

**Important**: Conversions from higher to lower precision must be done explicitly with `ct.astype()`. cuTile will not automatically downcast.

## 支持的类型 / Supported Types

| 类别 / Category | 类型 / Types |
|------|-------|
| 整数 / Integer | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64` |
| 浮点 / Float | `float16`, `bfloat16`, `float32`, `float64` |
| 特殊 / Special | `tfloat32`, `float8_e4m3fn`, `float8_e5m2` |
| 布尔 / Boolean | `bool_` |

---

# 第 4 章：矩阵转置 / Chapter 4: Matrix Transpose

> 源码 / Source: `4-transpose/transpose_2d.py`

## 算法 / Algorithm

矩阵转置交换行和列：`output[j][i] = input[i][j]`。

Matrix transpose swaps rows and columns: `output[j][i] = input[i][j]`.

基于 Tile 的转置分三步：

Tiled transpose has three steps:

1. **加载 / Load**：从输入矩阵的 `(row=pid_y, col=pid_x)` 位置加载 tile
2. **转置 / Transpose**：使用 `ct.transpose(tile)` 转置 tile 内容
3. **存储 / Store**：在输出矩阵的**交换位置** `(row=pid_x, col=pid_y)` 存储

1. **Load** a tile from input at `(row=pid_y, col=pid_x)`
2. **Transpose** tile contents with `ct.transpose(tile)`
3. **Store** at the **swapped** position `(row=pid_x, col=pid_y)` in output

```python
@ct.kernel
def transpose_kernel(input_arr, output_arr,
                     tile_w: ct.Constant[int],
                     tile_h: ct.Constant[int]):
    pid_x = ct.bid(0)  # 列 tile 索引 / column tile index
    pid_y = ct.bid(1)  # 行 tile 索引 / row tile index

    # 从输入 (height, width) 加载，位置 (行, 列)
    # Load from input (height, width) at (row, col)
    tile = ct.load(input_arr, index=(pid_y, pid_x), shape=(tile_h, tile_w))

    # 转置 tile 内容: (tile_h, tile_w) -> (tile_w, tile_h)
    # Transpose tile contents: (tile_h, tile_w) -> (tile_w, tile_h)
    tile_T = ct.transpose(tile)

    # 存储到输出 (width, height)，位置交换
    # Store to output (width, height) at swapped position
    ct.store(output_arr, index=(pid_x, pid_y), tile=tile_T)
```

### 网格设置 / Grid Setup

```python
height, width = 128, 256
tile_h, tile_w = 16, 32

grid_x = ct.cdiv(width, tile_w)    # 256/32 = 8
grid_y = ct.cdiv(height, tile_h)   # 128/16 = 8
grid = (grid_x, grid_y, 1)

# 输出形状是转置后的维度 / Output shape is the transposed dimensions
output = cp.zeros((width, height), dtype=cp.int32)
```

### 关键点 / Key Points

- **`ct.transpose(tile)`** 交换 tile 的最后两个维度。对于 2D tile `(H, W)` → `(W, H)`。
- **坐标交换**是转置的核心：加载用 `(pid_y, pid_x)`，存储用 `(pid_x, pid_y)`。
- **非方阵**同样适用：输出形状必须是 `(width, height)`。

- **`ct.transpose(tile)`** swaps the last two dimensions. For 2D tile `(H, W)` → `(W, H)`.
- **Coordinate swapping** is the essence of transpose: load at `(pid_y, pid_x)`, store at `(pid_x, pid_y)`.
- **Non-square matrices** work the same way: output shape must be `(width, height)`.

```bash
.venv/bin/python 4-transpose/transpose_2d.py
```

---

# 第 5 章：进阶技巧 / Chapter 5: Advanced Tips

> 源码 / Source: `5-secret-notes/README.md`

本章汇集了 cuTile 编程中不容易从文档直接发现的技巧和陷阱。

This chapter collects tips and gotchas that aren't immediately obvious from the documentation.

## 5.1 Tile 维度必须是 2 的幂 / Tile Dimensions Must Be Powers of 2

Tile 的每个维度**必须**是 2 的幂（1, 2, 4, 8, 16, 32, 64, ...）。

Every dimension of a Tile **must** be a power of 2 (1, 2, 4, 8, 16, 32, 64, ...).

```python
# 正确 / OK
tile = ct.load(arr, index=(pid,), shape=(32,))
tile = ct.load(arr, index=(py, px), shape=(16, 64))

# 错误 — 编译时报错 / BAD — fails at compile time
tile = ct.load(arr, index=(pid,), shape=(48,))    # 48 不是 2 的幂 / not power of 2
```

如果数据大小不是 tile 大小的整数倍，cuTile 会自动处理边界：越界加载返回填充值，越界存储被静默忽略。

If data size isn't a multiple of tile size, cuTile handles edges automatically: out-of-bounds loads return padding values, out-of-bounds stores are silently ignored.

## 5.2 ct.Constant 对性能至关重要 / ct.Constant is Critical for Performance

没有 `ct.Constant`，编译器无法：展开循环、高效分配寄存器、在编译时确定 tile 形状。

Without `ct.Constant`, the compiler cannot: unroll loops, allocate registers efficiently, or determine tile shapes at compile time.

**规则**：凡是用在 `shape=`、`ct.zeros()`、`ct.full()`、`range()` 等编译时构造中的值，**必须**是 `ct.Constant`。

**Rule**: Any value used in `shape=`, `ct.zeros()`, `ct.full()`, `range()`, or other compile-time constructs **must** be `ct.Constant`.

## 5.3 内存访问顺序 / Memory Access Order

`ct.load()` / `ct.store()` 的 `order` 参数控制内存访问模式：

The `order` parameter in `ct.load()` / `ct.store()` controls memory access pattern:

- `order='C'`（默认）：行优先 — 同一行的元素在内存中连续 / Row-major — elements within a row are contiguous
- `order='F'`：列优先 — 同一列的元素在内存中连续 / Column-major — elements within a column are contiguous

当算法按列访问数据时，使用 `order='F'` 可以显著提高内存带宽。

When your algorithm accesses data column-by-column, using `order='F'` can dramatically improve memory bandwidth.

## 5.4 边界处理：PaddingMode / Edge Handling: PaddingMode

```python
ct.load(arr, index=(pid,), shape=(tile_size,),
        padding_mode=ct.PaddingMode.ZERO)       # 越界 → 0 / Out-of-bounds → 0
```

可用模式 / Available modes:
- `UNDETERMINED`（默认 / default）：填充值未指定 / unspecified
- `ZERO`：填充 0 / pad with 0
- `NAN`：填充 NaN
- `NEG_INF` / `POS_INF`：填充 ±∞ / pad with ±infinity

## 5.5 ct.mma vs @ 运算符 / ct.mma vs @ Operator

| 特性 / Feature | `@`（`ct.matmul`） | `ct.mma(a, b, acc)` |
|---------|----------------------|---------------------|
| 运算 / Operation | `a @ b` | `a @ b + acc`（融合 / fused） |
| 硬件映射 / Hardware | 取决于编译器 / Compiler decides | 显式使用 MMA 指令 / Explicitly targets MMA |
| 累加器 / Accumulator | 创建新 tile / Creates new tile | 复用已有累加器 / Reuses existing |
| 适用场景 / Best for | 单次矩阵乘 / One-off matmul | 内层循环 / Inner loops (GEMM, attention) |

```python
# 内层循环中推荐使用 ct.mma / Preferred in inner loops
acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
for k in range(num_k_tiles):
    a = ct.load(A, index=(pid_m, k), shape=(tile_m, tile_k))
    b = ct.load(B, index=(k, pid_n), shape=(tile_k, tile_n))
    acc = ct.mma(a, b, acc)
```

## 5.6 调试：ct.printf / Debugging with ct.printf

```python
@ct.kernel
def debug_kernel(data, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    ct.printf("Block %d starting\n", pid)
```

**注意**：`ct.printf` 会序列化输出，严重影响性能，仅用于调试。

**Warning**: `ct.printf` serializes output and significantly affects performance. Use only for debugging.

## 5.7 核函数编译器选项 / Kernel Compiler Options

```python
@ct.kernel(num_ctas=4, occupancy=8, opt_level=3)
def optimized_kernel(...):
    ...
```

| 参数 / Parameter | 范围 / Range | 效果 / Effect |
|-----------|-------|--------|
| `num_ctas` | 1–16（2 的幂） | CGA 中的 CTA 数量 / CTAs in a CGA |
| `occupancy` | 1–32 | 每个 SM 的预期活跃 CTA 数 / Expected active CTAs per SM |
| `opt_level` | 0–3 | 编译优化级别 / Compiler optimization level |

针对不同架构的调优 / Architecture-specific tuning:

```python
from cuda.tile import ByTarget

@ct.kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
def arch_tuned_kernel(...):
    ...
```

## 5.8 归约操作与 keepdims / Reductions with keepdims

```python
# 行最大值: (M, N) -> (M, 1) / Row-wise max
row_max = ct.max(tile, axis=1, keepdims=True)

# 行求和: (M, N) -> (M, 1) / Row-wise sum
row_sum = ct.sum(tile, axis=1, keepdims=True)

# 数值稳定的 softmax 技巧 / Numerically stable softmax trick
shifted = tile - ct.max(tile, axis=1, keepdims=True)
```

`keepdims=True` 保留被归约的维度（大小为 1），使得结果可以与原始 tile 广播运算。

`keepdims=True` preserves the reduced dimension as size 1, enabling broadcasting with the original tile.

## 5.9 常见错误信息 / Common Error Messages

| 错误 / Error | 原因 / Cause | 修复 / Fix |
|-------|-------|-----|
| `TileSyntaxError` | 核函数中使用了不支持的 Python 语法 | 简化代码，避免类/生成器等 |
| `TileTypeError` | 类型不匹配（如 fp32 → fp16） | 使用 `ct.astype()` 显式转换 |
| `TileValueError` | 常量值无效 | 检查 `ct.Constant` 参数 |
| `TileCompilerTimeoutError` | 核函数过于复杂 | 减小 tile 大小或简化逻辑 |

## 5.10 其他实用功能 / Other Useful Features

**原子操作 / Atomic Operations:**
```python
ct.atomic_add(output, indices=(row, col), update=value,
              memory_order=ct.MemoryOrder.RELAXED,
              memory_scope=ct.MemoryScope.DEVICE)
```

**非连续访问 / Irregular Access (gather/scatter):**
```python
values = ct.gather(array, (row_indices, col_indices), padding_value=0)
ct.scatter(array, (row_indices, col_indices), values)
```

**形状操作 / Shape Manipulation:**
```python
flat = ct.reshape(tile_2d, (-1,))           # 展平 / flatten
expanded = ct.expand_dims(tile_1d, axis=0)  # (N,) -> (1, N)
combined = ct.cat([tile_a, tile_b], axis=0) # 拼接 / concatenate
```

---

# 第 6 章：性能调优：Tile 大小 / Chapter 6: Performance Tuning: Tile Size

> 源码 / Source: `6-performance-tuning/autotune_benchmark.py`

## 权衡 / The Trade-off

选择"正确"的 tile 大小需要平衡多个硬件约束：

Choosing the "right" tile size balances several hardware constraints:

**1. 并行性（占用率）/ Parallelism (Occupancy)**
- **小 tile**：产生更多 block，有利于填满 GPU，但调度开销更大。
- **大 tile**：block 更少，如果 block 数少于 SM 数量，GPU 将无法充分利用。

- **Small tiles**: Create more blocks, good for filling the GPU, but higher scheduling overhead.
- **Large tiles**: Fewer blocks. If blocks < SMs, the GPU will be underutilized.

**2. 寄存器压力 / Register Pressure**
- 每个 tile 消耗寄存器。大 tile 通常需要更多寄存器，可能导致每个 SM 上能同时运行的 block 减少。

- Each tile consumes registers. Large tiles need more registers, potentially reducing blocks per SM.

**3. 内存合并 / Memory Coalescing**
- 较大的 tile 通常允许更宽、更高效的内存事务。

- Larger tiles often allow wider, more efficient memory transactions.

## 自动调优 / Autotuning

由于难以从数学上预测最优大小，业界标准做法是**自动调优**：

Since it's hard to predict the optimal size mathematically, the industry standard is **autotuning**:

```python
@ct.kernel
def math_kernel(data, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    r = ct.load(data, index=(pid,), shape=(tile_size,))
    res = r * r
    res = res + r
    res = res * 0.5
    res = res * res
    ct.store(out, index=(pid,), tile=res)
```

对不同的 tile 大小进行基准测试：

Benchmark different tile sizes:

```python
candidate_sizes = [32, 64, 128, 256, 512, 1024]

for size in candidate_sizes:
    grid = (ct.cdiv(N, size), 1, 1)
    # ... 计时并比较 / time and compare
```

典型结果：中等大小（如 128）通常表现最佳；过小（32）有调度开销，过大（1024）有寄存器压力。

Typical result: Medium sizes (e.g., 128) usually perform best; too small (32) has scheduling overhead, too large (1024) has register pressure.

```bash
.venv/bin/python 6-performance-tuning/autotune_benchmark.py
```

---

# 第 7 章：注意力机制 / Chapter 7: Attention Mechanism

> 源码 / Source: `7-attention/attention.py`

## 概述 / Overview

本章将前面所学的所有概念综合起来，实现 Transformer 中使用的**注意力机制（Attention Mechanism）**的简化版本。

This chapter brings together all the concepts learned so far to implement a simplified version of the **Attention Mechanism** used in Transformers (like LLMs).

## 数学公式 / The Math

标准注意力定义为：

Standard attention is defined as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

在我们的简化实现中（为了避免在教程中引入在线 softmax 归一化的复杂性），我们实现：

In our simplified implementation (to avoid the complexity of online-softmax normalization logic in a tutorial), we implement:

$$ \text{Out} = \sum_j \exp\left(\frac{QK_j^T}{\sqrt{d_k}}\right) V_j $$

## 核函数 / The Kernel

```python
@ct.kernel
def simple_attention(Q, K, V, Out,
                     seq_len_k: int,
                     d_head: ct.Constant[int],
                     tile_size_m: ct.Constant[int],
                     tile_size_n: ct.Constant[int]):

    pid_m = ct.bid(0)

    # 加载 Query tile: (tile_m, D)
    # Load Query tile: (tile_m, D)
    q_tile = ct.load(Q, index=(pid_m, 0), shape=(tile_size_m, d_head))

    # 初始化累加器 / Initialize accumulator
    acc_tile = q_tile * 0.0

    # K/V tile 数量 / Number of K/V tiles
    num_k_tiles = ct.cdiv(seq_len_k, tile_size_n)

    # 遍历 K/V 的所有 tile / Loop over all K/V tiles
    for k_id in range(num_k_tiles):
        # 加载 K tile 和 V tile / Load K and V tiles
        k_tile = ct.load(K, index=(k_id, 0), shape=(tile_size_n, d_head))
        v_tile = ct.load(V, index=(k_id, 0), shape=(tile_size_n, d_head))

        # 1. 注意力分数: Q @ K.T → (tile_m, tile_n)
        # 1. Attention scores: Q @ K.T → (tile_m, tile_n)
        k_tile_T = ct.transpose(k_tile)
        score_tile = q_tile @ k_tile_T

        # 2. 缩放 / Scale: 1/sqrt(d)
        scale = d_head ** -0.5
        score_tile = score_tile * scale

        # 3. 指数化（简化版，跳过归一化）
        # 3. Exponentiate (simplified, skip normalization)
        exp_score = ct.exp(score_tile)

        # 4. 加权求和: A @ V → (tile_m, D)
        # 4. Weighted sum: A @ V → (tile_m, D)
        weighted_val = exp_score @ v_tile

        # 累加 / Accumulate
        acc_tile = acc_tile + weighted_val

    # 存储结果 / Store result
    ct.store(Out, index=(pid_m, 0), tile=acc_tile)
```

## 关键概念 / Key Concepts Demonstrated

1. **复杂数据流**：同时加载三个不同的矩阵（Q、K、V）。
2. **Tiled 矩阵乘法**：
   - `score = q_tile @ k_tile_T`（计算相似度）
   - `weighted = exp_score @ v_tile`（加权求和）
3. **核函数内循环**：按 tile_n 大小的块遍历 K/V 序列长度，同时将 Q 保持在寄存器中。这是 **FlashAttention** 的基础。
4. **数学函数**：使用 `ct.exp()` 对 tile 进行逐元素指数运算。

1. **Complex Data Flow**: Loading three different matrices (Q, K, V).
2. **Tiled Matrix Multiplication**:
   - `score = q_tile @ k_tile_T` (compute similarity)
   - `weighted = exp_score @ v_tile` (weighted sum)
3. **Kernel Loops**: Iterating over the Key/Value sequence in tile_n chunks while holding Q in registers. This is the foundation of **FlashAttention**.
4. **Math Functions**: Using `ct.exp()` for element-wise exponential on tiles.

## 执行流程 / Execution Flow

1. **Grid 设置**：计算覆盖所有 Query 所需的 tile 数。每个 block 处理 tile_m 个 query。
2. **循环**：在核函数内部，按 tile_n 步长遍历 K/V 序列。
3. **计算**：
   - 加载一个 K/V 块
   - 计算 Q 块与当前 K 块的相似度
   - 应用非线性函数（Exp）
   - 乘以 V 块
   - 累加到结果中
4. **存储**：将累加结果写回全局内存。

1. **Grid**: Calculate how many tiles to cover all Queries. Each block handles tile_m queries.
2. **Loop**: Inside the kernel, loop over K/V in steps of tile_n.
3. **Compute**:
   - Load a K/V block
   - Compute similarity between Q block and current K block
   - Apply non-linearity (Exp)
   - Multiply by V block
   - Accumulate into result
4. **Store**: Write accumulated result back to global memory.

这种 "Tiling" 策略允许我们为非常长的序列计算 Attention，而无需在全局内存中创建巨大的 $(M \times N)$ 注意力矩阵，从而节省大量内存。

This "Tiling" strategy allows us to compute Attention for very long sequences without creating the massive $(M \times N)$ attention matrix in global memory, saving huge amounts of memory.

## 主机代码 / Host Code

```python
M = 128  # Query 数量 / Number of Queries
N = 128  # Key/Value 数量 / Number of Keys/Values
D = 64   # Head 维度 / Head Dimension

TILE_M = 32
TILE_N = 32

# 数据准备 / Data preparation
q = cp.random.normal(0, 1, (M, D)).astype(cp.float32)
k = cp.random.normal(0, 1, (N, D)).astype(cp.float32)
v = cp.random.normal(0, 1, (N, D)).astype(cp.float32)
out = cp.zeros((M, D), dtype=cp.float32)

# Grid: 覆盖所有 Query / Cover all Queries
grid = (ct.cdiv(M, TILE_M), 1, 1)

# 启动核函数 / Launch kernel
ct.launch(cp.cuda.get_current_stream(), grid,
          simple_attention, (q, k, v, out, N, D, TILE_M, TILE_N))
```

## 验证 / Verification

```python
# CPU 参考实现 / CPU reference implementation
h_q, h_k, h_v = cp.asnumpy(q), cp.asnumpy(k), cp.asnumpy(v)
scores = (h_q @ h_k.T) / np.sqrt(D)
attn = np.exp(scores)
expected = attn @ h_v

h_out = cp.asnumpy(out)
np.testing.assert_allclose(h_out, expected, rtol=1e-3, atol=1e-3)
```

### 注意 / Note

本实现是**简化版**，没有 softmax 归一化。在实际应用中，应使用 **FlashAttention**（在线 softmax）来获得正确的归一化注意力输出和数值稳定性。

This implementation is **simplified** without softmax normalization. In production, use **FlashAttention** (online softmax) for correct normalized attention output and numerical stability.

```bash
.venv/bin/python 7-attention/attention.py
```

---

## 附录：cuTile API 速查表 / Appendix: cuTile API Quick Reference

### 核心 / Core

| 函数 / Function | 说明 / Description |
|------|------|
| `@ct.kernel` | 标记 GPU 核函数 / Mark GPU kernel |
| `ct.launch(stream, grid, kernel, args)` | 启动核函数 / Launch kernel |
| `ct.bid(axis)` | 获取 Block ID / Get Block ID |
| `ct.cdiv(x, y)` | 向上取整除法 / Ceiling division |

### 数据传输 / Data Transfer

| 函数 / Function | 说明 / Description |
|------|------|
| `ct.load(arr, index, shape)` | 从全局内存加载 tile / Load tile from global memory |
| `ct.store(arr, index, tile)` | 将 tile 写回全局内存 / Store tile to global memory |
| `ct.gather(arr, indices)` | 非连续加载 / Non-contiguous load |
| `ct.scatter(arr, indices, values)` | 非连续存储 / Non-contiguous store |

### 创建 / Creation

| 函数 / Function | 说明 / Description |
|------|------|
| `ct.zeros(shape, dtype)` | 全零 tile / Zero-filled tile |
| `ct.ones(shape, dtype)` | 全一 tile / One-filled tile |
| `ct.full(shape, value, dtype)` | 指定值填充 / Fill with value |
| `ct.arange(size, dtype=)` | 等差序列 / Range sequence |

### 数学 / Math

| 函数 / Function | 说明 / Description |
|------|------|
| `ct.exp(x)` | 指数 / Exponential |
| `ct.log(x)` / `ct.log2(x)` | 对数 / Logarithm |
| `ct.sqrt(x)` / `ct.rsqrt(x)` | 平方根 / 倒数平方根 / Square root / Reciprocal |
| `ct.sin` / `ct.cos` / `ct.tanh` | 三角 / 双曲函数 / Trig / Hyperbolic |
| `ct.floor(x)` / `ct.ceil(x)` | 取整 / Rounding |

### 矩阵与归约 / Matrix and Reduction

| 函数 / Function | 说明 / Description |
|------|------|
| `a @ b` / `ct.matmul(a, b)` | 矩阵乘法 / Matrix multiply |
| `ct.mma(a, b, acc)` | 融合乘累加 / Fused multiply-accumulate |
| `ct.transpose(tile)` | 转置 / Transpose |
| `ct.sum(x, axis, keepdims)` | 求和 / Sum |
| `ct.max(x, axis, keepdims)` | 最大值 / Maximum |
| `ct.min(x, axis, keepdims)` | 最小值 / Minimum |

### 类型与形状 / Type and Shape

| 函数 / Function | 说明 / Description |
|------|------|
| `ct.astype(x, dtype)` | 类型转换 / Type conversion |
| `ct.reshape(x, shape)` | 重塑 / Reshape |
| `ct.expand_dims(x, axis)` | 添加维度 / Add dimension |
| `ct.broadcast_to(x, shape)` | 广播 / Broadcast |
| `ct.where(cond, x, y)` | 条件选择 / Conditional select |
| `ct.maximum(x, y)` / `ct.minimum(x, y)` | 逐元素最大/最小 / Element-wise max/min |
