## `gnnmath::vector` – Vector Operations for GNN-based 3D Mesh Simplification

The `gnnmath::vector` namespace in the **MeshGNN** project provides mathematical operations for vectors, specifically tailored for **Graph Neural Network (GNN)** applications in **3D mesh simplification**. It supports computations on node and edge feature vectors, enabling feature transformations and edge scoring in GNNs using the C++17 standard library.

---

### 📦 Overview

* Operates on vectors represented as sequences of real numbers (`double`).
* Includes vector arithmetic, inner products, norms, and activation functions.
* Implements **parallelized** operations and **numerical stability** checks.

---

### 📚 Dependencies

* **C++17** standard library:

  * `<vector>`, `<cmath>`, `<algorithm>`, `<stdexcept>`, `<execution>`
* ✅ No external libraries required

---

### 📐 Mathematical Definitions

Given vectors:

$$
\mathbf{a} = [a_1, a_2, \dots, a_n], \quad \mathbf{b} = [b_1, b_2, \dots, b_n]
$$

#### 🔢 Vector Operations

* **Vector Addition**

  $$
  \mathbf{a} + \mathbf{b} = [a_1 + b_1, a_2 + b_2, \dots, a_n + b_n]
  $$

* **Vector Subtraction**

  $$
  \mathbf{a} - \mathbf{b} = [a_1 - b_1, a_2 - b_2, \dots, a_n - b_n]
  $$

* **In-place Addition/Subtraction**

  $$
  \mathbf{a} \gets \mathbf{a} + \mathbf{b}, \quad \mathbf{a} \gets \mathbf{a} - \mathbf{b}
  $$

* **Scalar Multiplication**

  $$
  c \cdot \mathbf{a} = [c a_1, c a_2, \dots, c a_n]
  $$

* **Dot Product (Inner Product)**

  $$
  \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i
  $$

* **Euclidean Norm (L2 Norm)**

  $$
  \|\mathbf{a}\|_2 = \sqrt{\sum_{i=1}^n a_i^2}
  $$

---

#### 🧠 Activation Functions (Element-wise)

* **ReLU**:

  $$
  \text{ReLU}(x) = \max(0, x)
  $$

* **Sigmoid**:

  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$

* **Mish**:

  $$
  \text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))
  $$

* **Softmax**:

  $$
  \text{Softmax}(\mathbf{a})_i = \frac{e^{a_i}}{\sum_{j=1}^n e^{a_j}}
  $$

* **Softplus**:

  $$
  \text{Softplus}(x) = \ln(1 + e^x)
  $$

* **GELU (Gaussian Error Linear Unit)**:

  $$
  \text{GELU}(x) = x \cdot \Phi(x), \quad \Phi(x) = \frac{1}{2} \left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
  $$

* **SiLU (Sigmoid Linear Unit)**:

  $$
  \text{SiLU}(x) = x \cdot \sigma(x)
  $$

* **Softsign**:

  $$
  \text{Softsign}(x) = \frac{x}{1 + |x|}
  $$

---

### 🧩 Usage in MeshGNN

* Feature transformation in GNN layers (e.g., ReLU, softmax)
* Dot product for similarity computation
* Euclidean norm for feature normalization
* Softmax-based edge scoring in mesh simplification

---

### 🛠️ Build & Integration

#### 🔧 Clone the Repository

```bash
git clone https://github.com/username/MeshGNN.git
cd MeshGNN
```

#### 🏗️ Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

#### 📌 Include in Your Project

* Add `include/gnnmath` to your include path
* Link against the `GNNMath` library (static or shared)

---

### 📦 Requirements

* C++17-compliant compiler (e.g., **GCC 7+**, **Clang 5+**)
* **CMake** ≥ 3.10

---

### ✅ Testing

Unit tests are available in:

```
tests/test_vector.cpp
```

Run them with:

```bash
cd build
make test
```

---

### 🤝 Contributing

* Submit issues and pull requests via [MeshGNN repository](https://github.com/buniatyann/MeshGNN).
* Follow the style in `vector.hpp`.
* Include unit tests for all new features.

---

### 📄 License

MIT License — see [`LICENSE`](./LICENSE) for details.

---
