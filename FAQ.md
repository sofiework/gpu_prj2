# CS 7295 — Project 2 FAQ (from Ed Discussion)

Source: Ed Discussion, "Project 2 - FAQ #33" (pinned, Scott Madeira) + recent thread replies.
Captured: 2026-09-11.

## Official FAQ (pinned post)

**Q. Am I required to use a Bitonic sort? There are a variety of parallel sorting algorithms.**
Your solution is required to implement the Bitonic sorting algorithm. Failure to do this could result in a zero for the project.

**Q. Are there resources to learn about Bitonic Sort?**
- Visual explanation: https://www.youtube.com/watch?v=uEfieI0MumY
- Wikipedia: https://en.wikipedia.org/wiki/Bitonic_sorter

**Q. Possible performance instability due to Cluster GPU**
The cluster GPU is shared, so numbers vary run to run (especially near the deadline). Grading is done when the GPU is idle. It's OK to compare speedup/throughput/occupancy with classmates in the "Project 2 - Performance" thread.

**Q. Resources for Nsight Compute analysis?**
Nsight Compute | NVIDIA Developer. The linked videos give a basic walkthrough plus examples of using it to improve performance.

**Q. Does bitonic sort have to handle any array size?**
The algorithm itself needs power-of-2 lengths, but the implementation must work for any input size. Pad the input up to the next power of 2 (e.g. 1000 → 1024) before running bitonic sort.

**Q. Required citation format?**
Any format is fine.

**Q. "Break down your kernel into two: one with shared memory and another with global memory" — do we submit both but call only one?**
The two-phase split is suggested, not enforced: a shared-memory kernel sorts as many integers as fit in shared memory, then a global kernel gathers and sorts those sorted runs. **Both kernels must be invoked** to get a fully sorted array.

**Q. Compiler options and performance**
Do **not** use `-g` / `-G` when measuring performance — these debug flags slow the code down.

**Q. ICE cluster configuration (used for grading)**
- Nvidia GPU **H100**
- Nodes: 1
- Cores per node: 2
- GPUs per core: 1
- Memory per core: 8GB

For functional development you can pick "Nvidia (First Available CUDA 13)" to get a session more easily, but tune performance with the grading configuration above.

**Q. Report format**
Standard report, format of your choosing, roughly a research-paper structure:
- Brief intro and summary of the problem and your approach
- Important implementation details
- Base performance data (GPU vs CPU, GPU across data sizes, etc.)
- Optimizations made (good and bad), effects, and analysis
- Observations from the project
- Citations for anything beyond course materials
- Possible next steps

Times Roman or Arial 11pt, **3 pages / ~750 words** of text plus figures/tables. Graders may stop reading at 750 words. Graduate-level work expected — properly formatted, substantive, no one-liners.

**Q. Notes on padding the GPU array**
- Padding must happen inside one of the functions in `bitonic.cu` — it exists only to make the GPU sort work. The CPU sort runs on the raw generated data (2,000 / 10,000 / …).
- Be creative: a plain `for` loop is probably not the best choice. Look at C memory-buffer functions and their CUDA equivalents.

**Q. Code execution observations**
GPU code is SIMT — one instruction across all threads in a warp. Divergent conditionals serialize: one branch's threads execute while the others wait, reducing parallelism.

**Q. Areas to look at to increase performance** (non-exhaustive; some may cancel each other out — try one GPU-code optimization plus one data-transfer optimization first)
- Shared memory
- Pinned memory
- Asynchronous operations
- Multiple kernels
- Memory transfer optimization
- Data type optimization

**Q. Can I use the sorting algorithms in the CUDA SDK?**
You may reference other sources for inspiration, but you must write your own bitonic sort. External sources must be properly cited and the influenced portions of code clearly indicated.

**Q. Other resources on sorting and GPUs** (working links from the requirements doc)
- Bitonic Sort
- Prof. Vuduc's bitonic sort lecture: items 23 (Comparator networks) through 28
- Batcher's Odd-Even Merge Sort
- Improved GPU Sorting
- PACE ICE cluster guide
- NVIDIA CUDA Toolkit Documentation
- CUDA Programming Guide

**Q. Tips to speed up development**
Debug with small arrays and block sizes — array size 32 with block size 4 makes each step easy to inspect.

**Q. Restrictions on code development?**
Any CUDA features are allowed. The major rule: **all code written because you are using a GPU must live in one of the provided functions in `bitonic.cu`**. Every line affecting the GPU must be "inside the timers" and counted in your performance. Writing plain C reduces the risk of tripping this rule. Doing GPU work in a static constructor before `main()` to dodge the timers results in a **zero**.

## Thread replies

**Transfer-time fluctuation between sessions** (Daniel David Schiopucie, 2d)
Saw H2D swing from ~8ms to ~20ms across sessions, with D2H swinging too; grader output went 18 → 21, and throughput 1120 → 890 MEPS day to day.
- *Scott Madeira:* This is fairly consistent with the cluster. Grading runs the code three times, on multiple days, to get the best result. Try an **H200** — performance is very similar to the H100 for this project and may be more stable.
- Follow-up referenced thread `#82`.

**Markdown → PDF for the report?** (Dane Michael Murphy, 3d)
*Scott Madeira:* We don't care how you generate the PDF, as long as you follow the report guidelines and the 750-word / 3-page limit plus graphs and charts.

**PACE GPU hardware issues** (Ruth Ran Tian, 4d)
Saw `Volatile Uncorr. ECC = 2` across two sessions; `nvidia-smi --gpu-reset` needs permissions they don't have.
*Scott Madeira:* Open a ticket with PACE support to report it and get a different GPU. Try requesting an H200.

**Which modules to study first?** (Rishi Soni, 4d)
Student guessed modules #4–#7.
*Scott Madeira:* View whichever modules you think will help — you have to do all of them eventually, so none of it is wasted.

**Are all array sizes graded for performance?** (Daniel David Schiopucie, 7d)
*Scott Madeira:* The Grading Rubric section in the project README is explicit about what is graded. (README states performance is checked on 100M.)

**Where do we submit the report?** (Qingwen Zhou, 1w)
Gradescope — upload the report and the zip as multiple files in the same submission. Gradescope submission is now available.
