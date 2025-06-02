---

## `gnnmath::matrix` – Matrix Operations for GNN-Based 3D Mesh Simplification

The `gnnmath::matrix` namespace in the **MeshGNN** project provides operations for **dense and sparse matrices**, enabling efficient **graph processing** and **feature transformations** for **Graph Neural Networks (GNNs)** in 3D mesh simplification. Built with the C++17 standard library.

---

### 📦 Overview

* Provides dense matrix and **Compressed Sparse Row (CSR)** sparse matrix structures.
* Supports:

  * Matrix arithmetic
  * Transposition
  * Graph operations (adjacency, Laplacian)
* Designed for numerical **stability** and **parallel performance**.

---

### 📚 Dependencies

* **C++17** standard library:
  `<vector>`, `<cmath>`, `<algorithm>`, `<stdexcept>`, `<execution>`
* [`gnnmath::vector`](#gnnmathvector) for vector operations
* ✅ No external libraries required

---

### 📐 Mathematical Definitions

Let:

* $A = [a_{ij}]$ be an $m \times n$ dense matrix
* $B = [b_{ij}]$ be a $p \times q$ dense matrix
* $\mathbf{x} = [x_1, x_2, \dots, x_n]$ be a vector
* CSR format stores: non-zero values, column indices, and row pointers

#### 🔢 Dense Matrix Operations

* **Matrix-Vector Multiplication**

  $$
  A \mathbf{x} = \left[ \sum_{j=1}^n a_{ij} x_j \right]_{i=1}^m
  $$

* **Matrix-Matrix Multiplication**

  $$
  (A B)_{ik} = \sum_{j=1}^n a_{ij} b_{jk}
  $$

* **Transpose**

  $$
  (A^T)_{ji} = a_{ij}
  $$

* **Element-wise Addition/Subtraction**

  $$
  (A \pm B)_{ij} = a_{ij} \pm b_{ij}
  $$

* **In-Place Addition/Subtraction**

  $$
  A \gets A \pm B
  $$

* **Element-wise Multiplication**

  $$
  (A \odot B)_{ij} = a_{ij} \cdot b_{ij}
  $$

* **Identity Matrix**

  $$
  I_{ij} = \begin{cases}
  1 & \text{if } i = j \\
  0 & \text{otherwise}
  \end{cases}
  $$

* **Frobenius Norm**

  $$
  |A|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2}
  $$

* **Diagonal Extraction**

  $$
  \text{diag}(A) = [a_{11}, a_{22}, \dots, a_{nn}]
  $$

---

#### 🧠 Sparse Matrix Operations (CSR Format)

* **Sparse Matrix-Vector Multiplication**

  $$
  A \mathbf{x} = \left[ \sum_{j : a_{ij} \neq 0} a_{ij} x_j \right]_{i=1}^m
  $$

* **Sparse Matrix-Matrix Multiplication**
  Combine non-zero elements using row-column matchings.

* **Sparse Transpose**
  Rearranges non-zero values and updates indices.

* **Sparse Addition/Subtraction**

  $$
  (A \pm B)_{ij} = a_{ij} \pm b_{ij}, \quad \text{for non-zero terms}
  $$

* **In-Place Sparse Addition/Subtraction**
  Modifies $A$ by sparse addition/subtraction with $B$

---

#### 🌐 Graph-Specific Operations

* **Adjacency Matrix (A)**

  $$
  A_{ij} = \begin{cases}
  1 & \text{if } (i, j) \text{ is an edge} \\
  0 & \text{otherwise}
  \end{cases}
  $$

* **Degree Vector (d)**

  $$
  d_i = \sum_{j=1}^V A_{ij}
  $$

* **Laplacian Matrix (L = D - A)**

  $$
  L_{ij} = \begin{cases}
  d_i & \text{if } i = j \\
  -1 & \text{if } (i,j) \text{ is an edge} \\
  0 & \text{otherwise}
  \end{cases}
  $$

* **Normalized Laplacian ( $L_{\text{norm}} = D^{-1/2} L D^{-1/2}$ )**

  $$
  (L_{\text{norm}})_{ij} = \begin{cases}
  1 & \text{if } i = j \text{ and } d_i \neq 0 \\
  -\frac{1}{\sqrt{d_i d_j}} & \text{if } (i, j) \text{ is an edge} \\
  0 & \text{otherwise}
  \end{cases}
  $$

* **Validation**
  Check if edge indices satisfy:

  $$
  u, v < V
  $$

* **Sparse to Dense Conversion**
  Reconstruct full matrix from CSR representation.

* **Symmetry Check**

  $$
  A = A^T
  $$

---

### 🧩 Usage in MeshGNN

* Build **adjacency** and **Laplacian** matrices for mesh graphs.
* Apply **GNN weight matrices** for node/edge features.
* Score **edges** using sparse vector products.
* Analyze mesh graphs with tools like **Frobenius norm** or **symmetry check**.

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
* Link against the `GNNMath` static/shared library

---

### 📦 Requirements

* **C++17** compiler (e.g., GCC 7+, Clang 5+)
* **CMake** ≥ 3.10

---

### ✅ Testing

Unit tests for `gnnmath::matrix` are in:

```
tests/test_matrix.cpp
```

Run tests with:

```bash
cd build
make test
```

---

### 🤝 Contributing

* Submit issues or pull requests via [MeshGNN repository](https://github.com/buniatyann/MeshGNN)
* Follow the coding style in `matrix.hpp`
* Include unit tests for all new features

---

### 📄 License

MIT License — see [`LICENSE`](./LICENSE) for details.

---
