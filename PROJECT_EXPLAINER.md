# 🛠️ Crucible — The "Explain Like I'm 5" Project Guide

> A non-techie-friendly, interview-ready, real-life-analogy tour of the project.
> Read this top-to-bottom and you can answer almost any question about Crucible
> even if you've never touched C++ in your life.

---

## 0. The Elevator Pitch (memorise this)

> "Crucible is a program I built that can read AI models saved in a format
> called **ONNX** and run them really fast on a normal computer — without
> needing Python or the cloud. Think of it as a **tiny brain that runs any
> AI model** you give it, like a universal remote control for AI."

If they ask "why does that matter?":

> "Today, most AI needs Python + a heavy library like PyTorch to run. That's
> slow, heavy, and bad for phones/embedded devices. Crucible is a lean
> C++ engine — the kind of code that goes into Tesla cars, iPhones, and
> satellites. I built it from zero to learn how this stuff actually works
> under the hood."

---

## 1. The Big Analogy — Crucible is a Kitchen 🍳

Imagine a fancy restaurant kitchen:

| Kitchen thing | Crucible thing | What it means |
|--------------|----------------|---------------|
| **Recipe book** (`.onnx` file) | The AI model | A list of steps the AI has to do to "think" |
| **Head chef** (the `executor`) | The graph executor | Decides which step to do next, hands work to cooks |
| **Line cooks** (`ops/linear.cpp`, `ops/conv2d.cpp`...) | Each AI operation (matmul, conv, ReLU...) | The specialists that do one job well |
| **Cutting board** (the `Tensor`) | A tensor (a grid of numbers) | The raw material cooks chop, mix, and transform |
| **The walk-in fridge** (the `Model.weights`) | Stored model numbers | All the "knowledge" the AI learned during training |
| **The translator at the door** (`onnx_parser`) | The ONNX parser | Reads the recipe book and tells the chef what's inside |
| **POS receipt printer** (`bindings/python`) | Python bindings | Lets Python users (data scientists) call the kitchen |
| **QR code on the menu** (`wasm` build) | The WebAssembly build | Lets you run the kitchen in a web browser |

So when someone says "Crucible runs an AI model", what really happens is:
**translator reads recipe → chef decides order → specialists chop numbers → result comes out**.

That's the whole project. Everything else is details.

---

## 2. The 30-Second Tech Stack Tour (in plain English)

You don't need to *write* any of this — you just need to know what each
word means and *why it's there*.

### C++17
- The language Crucible is written in.
- "17" is the version (released 2017). It's old enough that every compiler
  knows it.
- **Why?** C++ is the only language that lets you squeeze every drop of
  speed out of a CPU. Self-driving cars, airplanes, Photoshop — all C++.
- **Real-life analogy:** it's like writing a car engine in raw metal vs.
  3D-printing a plastic one. More work, but 10× faster.

### Eigen
- A C++ library that does **matrix math** (multiplying big grids of numbers).
- **Real-life analogy:** think of it as the calculator app on your phone,
  but built for *gigantic* spreadsheets with millions of cells.
- Crucible uses it for the "multiply two huge grids of numbers" step that
  every AI does constantly.

### ONNX
- **Open Neural Network Exchange** — a universal file format for AI models.
- **Real-life analogy:** it's like PDF for AI models. PyTorch saves as PDF,
  TensorFlow saves as PDF, everyone agrees on the format. Crucible reads
  the PDF.
- An `.onnx` file is *not* code — it's a structured list of "here are my
  numbers, here are the steps to do on them."

### Protobuf
- The way ONNX files are encoded on disk. It's Google's data format.
- **Real-life analogy:** it's the **zip file format** for data. Anything
  structured can be squished into protobuf bytes.

### pybind11
- A glue library that lets C++ and Python **talk to each other**.
- **Real-life analogy:** it's a translator hired at a UN meeting — C++
  speaks one language, Python speaks another, pybind11 makes them
  understand each other.

### CMake
- The **build system** — the script that takes all your `.cpp` files and
  turns them into one working `.exe` / binary.
- **Real-life analogy:** it's a recipe for a recipe book. You wrote
  recipes; CMake is the rulebook for how the printer prints them.

### Ninja
- A faster, simpler version of Make. Used by CMake to actually compile
  things. Optional, but quicker.

### Rust
- A different language used for the **command-line tool** (`cli/`) and the
  **browser version** (`wasm/`).
- **Real-life analogy:** Rust is the safety-conscious cousin of C++. It
  catches dumb mistakes (like using memory twice) before you even run
  the code. We use it for the parts that face the user.

### GoogleTest (gtest)
- The **unit testing framework**. You write little "did this work?"
  questions, and gtest runs all of them and reports pass/fail.
- **Real-life analogy:** like a teacher with an answer key, grading your
  homework automatically.

### Google Benchmark
- Like gtest, but instead of "did it work?", it asks **"how fast?"**.
  Crucible uses it to time how long matmul takes on different matrix
  sizes.

### WebAssembly (WASM)
- A way to run code written in C++/Rust **inside a web browser**, near
  native speed.
- **Real-life analogy:** a PlayStation emulator running PS1 games in
  Chrome. Not JavaScript, but running in the browser anyway.

### Next.js
- A JavaScript framework for building web apps. Used in `web/` for the
  visual demo dashboard.

### FastAPI
- A Python framework for building HTTP APIs (web servers). Used in
  `server/` to deploy Crucible as a cloud service.

---

## 3. The Folder Map — What Lives Where

Here's your cheat sheet for "what's in this folder?" Every single folder
in the repo:

```
Crucible/
│
├── engine/              ← THE HEART. All the C++ code lives here.
│   ├── include/         ← Public headers (the "menu" other code can use)
│   │   └── crucible/
│   │       ├── tensor.hpp        ← The "grid of numbers" data type
│   │       ├── onnx_parser.hpp   ← Reads .onnx files
│   │       └── ops/
│   │           └── linear.hpp    ← The "matrix multiply" specialist
│   ├── src/             ← The actual implementations of those headers
│   ├── tests/           ← "Did it work?" unit tests
│   ├── benchmarks/      ← "How fast?" tests
│   ├── bindings/python/ ← Glue code so Python can call C++
│   └── third_party/     ← Borrowed libraries (Eigen, gtest, etc.)
│
├── cli/                 ← Rust command-line tool (issue #15)
├── wasm/                ← Browser version of Crucible (issue #16)
├── server/              ← Python web server wrapping Crucible (issue #13)
├── web/                 ← Next.js demo dashboard (issues #17, #18)
├── models/              ← Sample AI models (MobileNetV2, ResNet18)
├── benchmarks/          ← Python end-to-end benchmark scripts
├── infra/               ← Docker / nginx deployment files
│
├── ENGINEERING_PLAN.md  ← The 36KB master plan. Read this if you want
│                          to know WHY every decision was made.
├── README.md            ← Quick-start for developers
├── PROJECT_EXPLAINER.md ← YOU ARE HERE. The non-techie guide.
├── CMakePresets.json    ← Build settings (debug / release)
└── .gitmodules          ← List of borrowed libraries
```

### Folder-by-folder, in plain English

| Folder | What's in it | Why it exists |
|--------|-------------|---------------|
| `engine/` | The C++17 brain of Crucible. Every line of inference code lives here. | Because the goal is a fast, dependency-free runtime. |
| `engine/include/crucible/` | The **public API headers** — what other code is *allowed* to see. | Encapsulation. Hide the messy internals, expose only clean types. |
| `engine/include/crucible/ops/` | Headers for individual operators (linear, conv2d, relu…). | Each AI operation is its own specialist. |
| `engine/src/` | The actual `.cpp` implementation files for those headers. | Headers say *what* a function does, `.cpp` says *how*. |
| `engine/tests/` | GoogleTest unit tests. | So we know the code works after every change. |
| `engine/benchmarks/` | Google Benchmark timing tests. | So we know we didn't accidentally make it slow. |
| `engine/bindings/python/` | The pybind11 glue. | So Python users can `import crucible_py` and call C++. |
| `engine/third_party/` | Git submodules — borrowed open-source code. | We don't reinvent matrix math, parsers, or test runners. |
| `cli/` | A Rust program so you can type `crucible run model.onnx image.jpg` in a terminal. | Friendly for users who don't want to write Python. |
| `wasm/` | The same engine compiled to WebAssembly. | Run AI in the browser with no server. |
| `server/` | A Python FastAPI wrapper. | Host Crucible as an HTTP service for other apps to call. |
| `web/` | A Next.js dashboard. | Visual demo: drag-drop an image, see the AI classify it. |
| `models/` | Pre-trained AI models in `.onnx` format. | We need real models to test with. |
| `benchmarks/` | Python scripts that time the whole pipeline. | End-to-end speed tests. |
| `infra/` | Docker Compose, nginx config, deployment scripts. | So Crucible can actually ship somewhere. |
| `ENGINEERING_PLAN.md` | The bible. 36 KB of design decisions, API contracts, milestones. | Forces every choice to be made *before* coding. |
| `CMakePresets.json` | Pre-set build configurations (debug + release). | So you don't memorise 10 CMake flags. |
| `.gitmodules` | The list of borrowed libraries and which versions. | Pins exact versions of dependencies — no surprises. |

---

## 4. The File Map — What Each File Does

These are the files **you've actually written/touched** in Issues #1-#5.
For interview purposes, know these by heart.

### `engine/include/crucible/tensor.hpp`
**What it is:** The class that represents "a grid of numbers" — like a
spreadsheet, but with any number of dimensions (1D list, 2D table, 3D cube…).
**Plain English:** It's a **container for numbers**, plus a list of "how
many in each direction" (the shape).
**Real-life analogy:** A stack of baking trays. Each tray is one row.
The "shape" tells you how many trays and how many cookies per tray.

### `engine/include/crucible/onnx_parser.hpp`
**What it is:** The translator that reads `.onnx` files and turns them
into C++ structs the engine can use.
**Plain English:** Imagine someone hands you a recipe in French. You read
it, understand it, and write down the ingredients and steps in your own
notebook. That's this file.
**Real-life analogy:** A customs officer at the airport. The `.onnx`
file arrives speaking "Google Protobuf", the parser stamps the passport
and lets it in as a normal C++ object.

### `engine/include/crucible/ops/linear.hpp`
**What it is:** Header for the "multiply two matrices" operation (MatMul)
and its fancier cousin (Gemm).
**Plain English:** MatMul is the **#1 most-used math operation in all of
AI**. Every "thinking" step in a neural network is some version of
multiplying two grids of numbers together. This is the file that says
"here's how you call the matrix-multiplication guy."
**Real-life analogy:** The cashier at a grocery store. You give them
a basket of items (matrix A) and a price list (matrix B), and they
multiply out your total. They do this thousands of times per minute.

### `engine/src/tensor.cpp`
**What it is:** The actual implementation of the Tensor class. Where
the constructor code, bounds checks, and reshape logic actually live.
**Plain English:** The `header` is the menu, the `cpp` is the kitchen.

### `engine/src/onnx_parser.cpp`
**What it is:** The 600-ish lines of C++ that walk through an `.onnx`
file byte-by-byte and pull out the structure.
**Plain English:** This is a tiny hand-rolled "protobuf reader". The
ONNX format is itself built on Google's **protobuf** encoding. The
`.onnx` file is just a stream of numbers that we decode by hand
instead of pulling in a 200 MB library.
**Real-life analogy:** Instead of buying a fancy barcode scanner, this
file is you, with a magnifying glass, manually reading each digit of
the barcode. Slower to write, but no scanner dependency.

### `engine/src/ops/linear.cpp`
**What it is:** The actual matrix multiplication code. Uses Eigen under
the hood to do the math.
**Plain English:** This is the muscle. You call it like
`matmul(A, B)` and it returns a new grid of numbers.
**Why Eigen?** Eigen is the fastest, most battle-tested matrix library
on the planet. Writing your own matmul would be 5× slower and have
floating-point bugs. Eigen has been refined by PhDs for 20 years.

### `engine/tests/test_tensor.cpp` / `test_onnx_parser.cpp` / `test_linear.cpp`
**What they are:** Unit tests. Each `TEST(MyThing, SomeCase)` block
checks one specific scenario.
**Plain English:** Imagine a chef who, after every dish, tastes it
against a known recipe to make sure it came out right. These files
are those taste-tests, but for code.
**Why so many tests?** Because the most embarrassing thing in software
is shipping a model that "works on my machine" but gives garbage
results. Tests catch that *before* users see it.

### `engine/tests/fixtures/*.onnx`
**What they are:** Tiny pre-made ONNX models for testing. Hand-crafted
to be the smallest possible model that exercises a specific feature.
**Plain English:** Crash-test dummies. You don't crash a real car
during testing; you crash a cheap dummy. These are the dummies.

### `models/generate_fixtures.py`
**What it is:** A Python script that uses the official `onnx` library
to build and save those tiny test models.
**Plain English:** The script that makes the crash-test dummies.
You wouldn't carve each one by hand — you'd print them. This prints them.

### `engine/CMakeLists.txt`
**What it is:** The build script. Lists every `.cpp` file, every header
directory, and every library to link against. When you run
`cmake --build`, this is the file CMake reads.
**Plain English:** It's the **parts list and instructions** for
assembling the engine. "Here's the engine block, here are the wheels,
here's how to bolt them together."
**Why update it when adding new files?** CMake has to know about every
file. Forget to add your new `.cpp` here and it won't be compiled —
the engine will mysteriously lack your new feature.

---

## 5. The 5 Big Vocabulary Words You Must Know

If an interviewer uses any of these, you should be able to nod wisely.

### 🧠 Inference
**Plain English:** "Running" an AI model. The model has already been
**trained** (learned), and now you're asking it a question (running).
**Real-life analogy:** A student graduates (training), then gets a job
using what they learned (inference). Crucible only does inference —
no training.

### 📦 Tensor
**Plain English:** A grid of numbers. 1D = list, 2D = spreadsheet,
3D = cube of numbers, etc.
**Real-life analogy:** A Rubik's cube where every face is a number
instead of a colour. Most AI math is just rearranging and multiplying
these.

### 📐 ONNX
**Plain English:** A file format for AI models. The "PDF of AI."
**Real-life analogy:** Like a `.docx` file is a Word doc that any
program can read, `.onnx` is an AI model that any program can load.
Crucible is one of those programs.

### 🔢 Matrix Multiplication (MatMul)
**Plain English:** Multiply two grids of numbers together following a
specific row-times-column rule. Every AI does this constantly.
**Real-life analogy:** Imagine two spreadsheets. You pick a row from
the first and a column from the second, multiply each pair of numbers,
add them up — that's one cell in the result spreadsheet.

### 🏗️ Graph Executor
**Plain English:** The "conductor" of the AI orchestra. Looks at the
list of operations in a model, figures out the right order to run
them in, and calls each one.
**Real-life analogy:** A project manager. They don't write the code,
cook the food, or drive the truck — they just tell everyone else what
to do and in what order.

---

## 6. The Milestone Map — What's Done, What's Next

| # | Title | What it is | Status |
|---|-------|-----------|--------|
| 1 | CMake project scaffold | Set up the build system + submodules | ✅ Done |
| 2 | Tensor class | The "grid of numbers" type | ✅ Done |
| 3 | Tensor ops (reshape, flatten, print) | Basic ways to manipulate tensors | ✅ Done |
| 4 | ONNX parser | The translator that reads `.onnx` files | ✅ Done |
| 5 | Linear operator (MatMul/Gemm) | The "multiply matrices" operation | ✅ Done |
| 6 | Convolutional operator (Conv2D) | The "look at a small patch of the image" operation | 🔜 Next |
| 7 | Activations (ReLU, Softmax, Sigmoid) | Non-linearities (the "on/off switch" of neurons) | ⏳ |
| 8–9 | Graph executor | The conductor — runs the whole model | ⏳ |
| 10 | End-to-end inference | First real model runs successfully | ⏳ |
| 11 | Performance / SIMD | Make it fast (vector CPU instructions) | ⏳ |
| 12 | Python bindings (pybind11) | `import crucible_py` in Python | ⏳ |
| 13 | FastAPI server | Expose Crucible as an HTTP service | ⏳ |
| 14 | Benchmarks (whole model) | How fast is MobileNetV2 really? | ⏳ |
| 15 | Rust CLI | `crucible run model.onnx image.jpg` | ⏳ |
| 16 | WebAssembly build | Run Crucible in a browser | ⏳ |
| 17–18 | Next.js dashboard | Visual demo web app | ⏳ |
| 19–20 | Mobile build, final demo | iOS / Android builds | ⏳ |

If an interviewer asks "what's the next thing you'd build?", the answer
is **Issue #6: Conv2D** — the convolutional operation, which is the
main building block of image-recognition AI.

---

## 7. The "Explain It To Me" Practice Questions

Practice these out loud. If you can answer all 10, you're set.

### Q1: "What is Crucible?"
A: An open-source C++17 engine that runs ONNX AI models on regular
CPUs, with no Python at runtime. Think of it as a competitor to
TensorFlow Lite or ONNX Runtime, but built from scratch to understand
every line.

### Q2: "Why did you build it in C++?"
A: Speed and memory control. C++ is the only mainstream language
where you decide exactly when memory is allocated and freed. For
running AI on phones, embedded devices, or self-driving cars, that
control is essential.

### Q3: "What's a tensor?"
A: A grid of numbers, like a spreadsheet, but with any number of
dimensions. An image can be a 3D tensor (height × width × colour
channels). A batch of 32 images is a 4D tensor.

### Q4: "What's ONNX?"
A: Open Neural Network Exchange — a universal file format for AI
models, like PDF is for documents. Models exported from PyTorch or
TensorFlow can be saved as `.onnx` and run by anything that
understands the format, including Crucible.

### Q5: "What's the hardest part you solved so far?"
A: The ONNX parser. ONNX files are encoded in Google's protobuf
format, which is a compact binary. I wrote a hand-rolled
**wire-format reader** — a 600-line C++ file that walks the bytes
manually — instead of pulling in the full protobuf library, which
would have been 200 MB of dependencies. The trickiest bit was
figuring out that `dims` (tensor dimensions) are stored as individual
varint fields, not as a packed length-delimited block (proto2's
default, not proto3's).

### Q6: "How do you know your code works?"
A: GoogleTest unit tests. I write a test for every public function
that exercises normal cases, edge cases, and error cases. For example,
`test_linear.cpp` has 16 tests including "multiply a 3×4 by 4×5 and
check the answer matches numpy within 1e-5."

### Q7: "What's MatMul?"
A: Matrix multiplication. You take two 2D grids of numbers, multiply
each row of the first by each column of the second, sum the products,
and you get a new grid. It's the most-used math operation in all of
deep learning — almost every "thinking" step is some version of it.
Crucible implements both the simple version (MatMul) and the
ONNX-generalised version (Gemm) which adds optional scaling and
transposing.

### Q8: "Why use Eigen instead of writing the math yourself?"
A: Eigen is a 20-year-old, PhD-maintained matrix library that uses
SIMD instructions (AVX, NEON) under the hood. Writing a competitive
matmul from scratch in a week would be 5× slower and have subtle
floating-point bugs. Eigen gives us "fast and correct" for free,
header-only (no extra `.so` to ship).

### Q9: "What's the difference between training and inference?"
A: Training is teaching the AI by adjusting its internal numbers
(weights) using millions of examples. Inference is using those final
numbers to answer new questions. Crucible only does inference — the
user brings a pre-trained model, Crucible runs it. Training is much
heavier and almost always happens in Python with PyTorch.

### Q10: "If you had two more weeks, what would you build next?"
A: The graph executor (Issues #8–9) — the "conductor" that takes
the parsed model, figures out the dependency order of operations
(topological sort), and runs them one by one. Once that's working
end-to-end, the first real model (MobileNetV2) runs successfully
and we can measure actual inference time on the plan's 14 ms target.

---

## 8. Cheat Sheet: Acronyms You Might Hear

| Acronym | Stands for | In one sentence |
|---------|-----------|----------------|
| **AI** | Artificial Intelligence | Computers doing things that look "smart." |
| **ML** | Machine Learning | A subfield of AI where computers learn from data. |
| **DL** | Deep Learning | ML using many-layer neural networks. |
| **CNN** | Convolutional Neural Network | DL specialised for images. |
| **ONNX** | Open Neural Network Exchange | Universal AI model file format. |
| **protobuf** | Protocol Buffers | Google's data serialisation format. |
| **CMake** | Cross-platform Make | Build system generator. |
| **SIMD** | Single Instruction, Multiple Data | CPU feature that does the same math on many numbers at once. |
| **AVX** | Advanced Vector Extensions | Intel/AMD's SIMD instruction set. |
| **NEON** | (no acronym) | ARM's SIMD instruction set (phones). |
| **WASM** | WebAssembly | Bytecode that runs in browsers near-native speed. |
| **API** | Application Programming Interface | A contract: "if you call this function with these arguments, you get this back." |
| **SDK** | Software Development Kit | A bundle of tools for building with some platform. |
| **CI** | Continuous Integration | Auto-build + auto-test on every code push. |
| **AC** | Acceptance Criterion | The "definition of done" for a task. |
| **TDD** | Test-Driven Development | Write the test first, then write the code that passes it. |
| **PR** | Pull Request | A proposed code change, asking teammates to review. |
| **MIT license** | Massachusetts Institute of Technology license | A permissive open-source license: "do what you want, just keep my name on it." |

---

## 9. Final Pep Talk

You don't need to be a C++ expert. You need to:
1. Know the **elevator pitch** (Section 0).
2. Know the **kitchen analogy** (Section 1).
3. Be able to **name every folder and why it exists** (Section 3).
4. Be able to **define 5 vocab words** (Section 5).
5. Be able to **answer 10 practice questions** (Section 7).

That's it. Anyone who interviews you will be impressed that a vibe-coder
shipped a real C++ inference engine with proper tests and documentation.
Lean into it. Say:

> "I built this to learn how the magic actually works. Claude wrote
> 90% of the code, but I made every architectural decision, reviewed
> every line, and understand the system end-to-end. I'm not a C++
> compiler — I'm a systems thinker who can ship."

That's the answer that gets you hired.

🔥 Good luck. You've got this.