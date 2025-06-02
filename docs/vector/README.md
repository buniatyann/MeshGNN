## `gnnmath::vector` – Vector Operations for GNN-based 3D Mesh Simplification

The `gnnmath::vector` namespace provides operations for vectors used in GNNs, such as node and edge feature transformations.

### 📦 Overview

- Operates on real-valued vectors (`double`)
- Supports:
  - Vector arithmetic (add, sub, dot)
  - Norms (L2)
  - Activation functions (ReLU, Sigmoid, Softmax, etc.)
- Designed for **GNN layers** and **mesh feature processing**
- Uses parallel algorithms and stability checks

### 📚 Dependencies

- C++17 standard library: `<vector>`, `<cmath>`, `<algorithm>`, `<stdexcept>`, `<execution>`
- ✅ No external libraries required

### 📐 Mathematical Definitions

Let \( \mathbf{a} = [a_1, \dots, a_n] \), \( \mathbf{b} = [b_1, \dots, b_n] \):

- **Addition**:  
  \[ \mathbf{a} + \mathbf{b} = [a_1 + b_1, \dots, a_n + b_n] \]
- **Subtraction**:  
  \[ \mathbf{a} - \mathbf{b} = [a_1 - b_1, \dots, a_n - b_n] \]
- **In-place Operations**:  
  \[ \mathbf{a} \gets \mathbf{a} \pm \mathbf{b} \]
- **Scalar Multiplication**:  
  \[ c \cdot \mathbf{a} = [c a_1, \dots, c a_n] \]
- **Dot Product**:  
  \[ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i \]
- **L2 Norm**:  
  \[ |\mathbf{a}|_2 = \sqrt{\sum_{i=1}^n a_i^2} \]

### 🔁 Activation Functions

- **ReLU**: \( 	ext{ReLU}(x) = \max(0, x) \)
- **Sigmoid**: \( \sigma(x) = rac{1}{1 + e^{-x}} \)
- **Mish**: \( x \cdot 	anh(\ln(1 + e^x)) \)
- **Softmax**: \( 	ext{Softmax}(\mathbf{a})_i = rac{e^{a_i}}{\sum_j e^{a_j}} \)
- **Softplus**: \( \ln(1 + e^x) \)
- **GELU**: \( x \cdot \Phi(x), \quad \Phi(x) = rac{1}{2}(1 + 	ext{erf}(rac{x}{\sqrt{2}})) \)
- **SiLU**: \( x \cdot \sigma(x) \)
- **Softsign**: \( rac{x}{1 + |x|} \)

### 🧩 Usage

Used for:

- Node/edge feature transformations
- Dot products for similarity
- Feature normalization
- Edge scoring for simplification

### 🧪 Testing

```bash
cd build
make test
```

---

## `gnnmath::matrix` – Matrix Operations

The `gnnmath::matrix` namespace provides dense and sparse matrix operations, including GNN-relevant graph algorithms.

### 📦 Overview

- Dense matrix (double-based)
- Sparse CSR matrix (non-zero values, indices, row pointers)
- Graph-based operations: Adjacency, Laplacian, etc.
- Optimized for GNN mesh processing

### 📚 Dependencies

- C++17 standard library:  
  `<vector>`, `<cmath>`, `<algorithm>`, `<stdexcept>`, `<execution>`
- `gnnmath::vector`

### 📐 Mathematical Definitions

#### 🔢 Dense Matrix Operations

- **Matrix-Vector**:  
  \[ A \mathbf{x} = [\sum_j a_{ij} x_j]_i \]
- **Matrix-Matrix**:  
  \[ (AB)_{ik} = \sum_j a_{ij} b_{jk} \]
- **Transpose**: \( A^T_{ji} = a_{ij} \)
- **Element-wise Add/Sub**: \( (A \pm B)_{ij} = a_{ij} \pm b_{ij} \)
- **In-Place**: \( A \gets A \pm B \)
- **Element-wise Mul**: \( (A \odot B)_{ij} = a_{ij} b_{ij} \)
- **Identity**: \( I_{ij} = \delta_{ij} \)
- **Frobenius Norm**: \( \|A\|_F = \sqrt{\sum a_{ij}^2} \)
- **Diagonal**: \( 	ext{diag}(A) = [a_{11}, \dots, a_{nn}] \)

#### 🧠 Sparse Matrix (CSR)

- **Matrix-Vector**:  
  \[ A \mathbf{x} = [\sum_{j \in 	ext{nz}} a_{ij} x_j]_i \]
- **Matrix-Matrix**, Transpose, Add/Sub, In-Place

#### 🌐 Graph Operations

- **Adjacency Matrix**: \( A_{ij} = 1 	ext{ if edge } (i,j) \)
- **Degree Vector**: \( d_i = \sum_j A_{ij} \)
- **Laplacian**: \( L = D - A \)
- **Normalized Laplacian**:  
  \[ L_{ij} = egin{cases}
  1 & i = j \
  -rac{1}{\sqrt{d_i d_j}} & 	ext{if } (i,j) 	ext{ edge} \
  0 & 	ext{otherwise}
  \end{cases} \]
- **Validation**: Check if all edge indices are valid
- **To Dense**: Expand CSR to dense
- **Symmetry Check**: \( A = A^T \)?

### 🧩 Usage

- Graph construction (adjacency, Laplacian)
- GNN weight transformations
- Edge scoring (multiplication)
- Mesh diagnostics (norms, symmetry)

### 🧪 Testing

```bash
cd build
make test
```

---

## 🔨 Building & Integration

```bash
git clone https://github.com/username/MeshGNN.git
cd MeshGNN
mkdir build && cd build
cmake ..
make
```

- Add `include/gnnmath` to your include path
- Link with the `GNNMath` library

### Requirements

- C++17 compiler (GCC 7+, Clang 5+)
- CMake ≥ 3.10

---

## 🤝 Contributing

- Use issues and PRs on GitHub
- Follow `matrix.hpp`/`vector.hpp` conventions
- Include unit tests for all new features

## 📄 License

MIT License. See `LICENSE` for details.
