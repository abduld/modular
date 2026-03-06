# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Mandelbrot set computation on CUDA GPU using Mojo.

This example demonstrates how to use Mojo's GPU programming capabilities to
compute the Mandelbrot set in parallel on an NVIDIA CUDA GPU. The kernel
launches a 2D grid of threads where each thread computes the escape iteration
count for a single pixel in the complex plane.

Features:
- LayoutTensor-based GPU kernel for type-safe tensor access
- 2D thread grid mapping to pixel coordinates
- Configurable resolution, iteration depth, and complex plane bounds
- ASCII art visualization of the resulting fractal
"""

from std.math import ceildiv
from std.sys import has_accelerator

from std.complex import ComplexScalar
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Output grid dimensions (width x height in characters/pixels).
comptime WIDTH = 120
comptime HEIGHT = 40

# Data types used on the GPU.
comptime float_dtype = DType.float32
comptime int_dtype = DType.int32

# Bounds of the complex plane region to render.
comptime MIN_X: Scalar[float_dtype] = -2.0
comptime MAX_X: Scalar[float_dtype] = 0.7
comptime MIN_Y: Scalar[float_dtype] = -1.12
comptime MAX_Y: Scalar[float_dtype] = 1.12

# Maximum number of iterations before a point is considered to be in the set.
comptime MAX_ITERATIONS = 200

# CUDA thread block size (16x16 = 256 threads per block is a common choice).
comptime BLOCK_SIZE = 16

# LayoutTensor layout for the output buffer (row-major, height x width).
comptime layout = Layout.row_major(HEIGHT, WIDTH)


# ---------------------------------------------------------------------------
# GPU Kernel
# ---------------------------------------------------------------------------


fn mandelbrot_kernel(
    output: LayoutTensor[int_dtype, layout, MutAnyOrigin],
):
    """Compute escape iterations for one pixel of the Mandelbrot set.

    Each GPU thread is mapped to a unique (row, col) coordinate in the output
    grid via its 2D global index. The thread converts that coordinate to a
    complex number *c* and iterates z = z² + c until |z|² > 4 or the
    iteration limit is reached.

    Args:
        output: Device tensor to store the per-pixel iteration counts.
    """
    var row = global_idx.y
    var col = global_idx.x

    # Guard against threads that fall outside the grid when the grid dimensions
    # are not exact multiples of the block size.
    if row >= HEIGHT or col >= WIDTH:
        return

    # Map the (col, row) pixel coordinate to a point in the complex plane.
    comptime SCALE_X = (MAX_X - MIN_X) / WIDTH
    comptime SCALE_Y = (MAX_Y - MIN_Y) / HEIGHT

    var cx = MIN_X + Float32(col) * SCALE_X
    var cy = MIN_Y + Float32(row) * SCALE_Y
    var c = ComplexScalar[float_dtype](cx, cy)

    # Iterate z = z² + c and count how many steps until escape.
    var z = ComplexScalar[float_dtype](0, 0)
    var iters = Scalar[int_dtype](0)

    var in_set_mask = Scalar[DType.bool](True)
    for _ in range(MAX_ITERATIONS):
        if not any(in_set_mask):
            break
        in_set_mask = z.squared_norm().le(4)
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    output[row, col] = iters


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_mandelbrot(tensor: LayoutTensor[int_dtype, layout, ...]) raises:
    """Render the Mandelbrot set as ASCII art to stdout.

    Characters are chosen from a palette based on the iteration count so that
    regions that escape quickly appear as lighter characters and the interior
    of the set (points that never escape) are rendered as spaces.

    Args:
        tensor: Host-accessible tensor of per-pixel iteration counts.
    """
    # Gradient palette – denser characters represent faster escape.
    comptime palette = StringSlice(" .:-=+*#%@")
    comptime palette_len = len(palette)

    print("Mandelbrot Set (CUDA GPU) —", WIDTH, "x", HEIGHT, "pixels,", MAX_ITERATIONS, "max iterations")
    print(String("─") * WIDTH)

    for row in range(HEIGHT):
        for col in range(WIDTH):
            var v = tensor[row, col]
            if v >= MAX_ITERATIONS:
                # Point is (likely) in the Mandelbrot set.
                print(" ", end="")
            else:
                var idx = Int(v % Int32(palette_len))
                print(palette[byte=idx], end="")
        print("")

    print(String("─") * WIDTH)


# ---------------------------------------------------------------------------
# Main – host-side orchestration
# ---------------------------------------------------------------------------


def main() raises:
    # Ensure a GPU accelerator is available at compile time.
    comptime assert (
        has_accelerator()
    ), "This example requires a CUDA-capable GPU"

    # Acquire the default GPU device context.
    var ctx = DeviceContext()
    print("GPU device:", ctx.name())

    # Allocate the output buffer on the device.
    var dev_buf = ctx.enqueue_create_buffer[int_dtype](comptime (layout.size()))
    var out_tensor = LayoutTensor[int_dtype, layout](dev_buf)

    # Calculate grid dimensions so that every pixel is covered by a thread.
    comptime COL_BLOCKS = ceildiv(WIDTH, BLOCK_SIZE)
    comptime ROW_BLOCKS = ceildiv(HEIGHT, BLOCK_SIZE)

    # Launch the kernel on the GPU.
    ctx.enqueue_function[mandelbrot_kernel, mandelbrot_kernel](
        out_tensor,
        grid_dim=(COL_BLOCKS, ROW_BLOCKS),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )
    ctx.synchronize()

    # Map device memory back to the host and visualize.
    with dev_buf.map_to_host() as host_buf:
        var host_tensor = LayoutTensor[int_dtype, layout](host_buf)
        draw_mandelbrot(host_tensor)
