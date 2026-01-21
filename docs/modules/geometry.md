# Geometry Module

The geometry module handles 3D mesh representation, file I/O, geometric feature extraction, and mesh simplification algorithms.

## Files

- `include/gnnmath/geometry/obj_loader.hpp` / `src/geometry/obj_loader.cpp` - OBJ file parsing
- `include/gnnmath/geometry/mesh.hpp` / `src/geometry/mesh.cpp` - Mesh representation
- `include/gnnmath/geometry/feature_extraction.hpp` / `src/geometry/feature_extraction.cpp` - Geometric features
- `include/gnnmath/geometry/mesh_processor.hpp` / `src/geometry/mesh_processor.cpp` - Mesh simplification

---

## OBJ Loader (`obj_loader.hpp`)

The OBJ loader parses Wavefront OBJ files, the most common 3D mesh format. It supports vertices, texture coordinates, normals, faces, materials, and object groups.

### OBJ File Format Overview

```obj
# Comment
mtllib material.mtl          # Material library reference

o ObjectName                 # Object name
g GroupName                  # Group name

v  x y z [w]                 # Vertex position
vt u v [w]                   # Texture coordinate
vn x y z                     # Vertex normal

usemtl MaterialName          # Use material
f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3  # Face (triangle)
f v1//vn1 v2//vn2 v3//vn3           # Face (no texture)
f v1 v2 v3 v4                       # Face (quad, no tex/normals)
```

### Data Structures

#### `obj_data`

The main structure holding parsed OBJ file contents.

```cpp
struct obj_data {
    // Vertex data
    std::vector<std::array<scalar_t, 3>> vertices;    // Position (x, y, z)
    std::vector<std::array<scalar_t, 2>> texcoords;   // Texture (u, v)
    std::vector<std::array<scalar_t, 3>> normals;     // Normal (nx, ny, nz)

    // Face data - each face contains indices into above arrays
    // Index format: {vertex_idx, texcoord_idx, normal_idx}
    // -1 indicates missing component
    std::vector<std::vector<std::array<index_t, 3>>> faces;

    // Materials
    std::vector<material> materials;
    std::map<std::string, index_t> material_map;  // name -> index

    // Groups/Objects
    std::vector<group> groups;

    // Metadata
    std::vector<std::string> warnings;
    std::string filename;
};
```

#### `load_options`

Configuration for the loading process.

```cpp
struct load_options {
    bool triangulate = true;           // Convert polygons to triangles
    triangulation_method tri_method = triangulation_method::FAN;
    bool generate_normals = true;      // Auto-generate if missing
    bool flip_normals = false;         // Reverse normal direction
    bool flip_texcoords_v = false;     // Flip V coordinate (1-v)
    scalar_t scale = 1.0;              // Uniform scaling factor
    std::array<scalar_t, 3> offset = {0, 0, 0};  // Translation
    bool strict_mode = false;          // Throw on warnings
};
```

#### Triangulation Methods

```cpp
enum class triangulation_method {
    FAN,           // Fan triangulation from first vertex
    EAR_CLIPPING,  // Ear clipping for concave polygons
    DELAUNAY       // Delaunay triangulation (quality)
};
```

**FAN** (Default):
- Simplest, fastest
- Works well for convex polygons
- May produce poor triangles for concave polygons

**EAR_CLIPPING**:
- Handles concave polygons correctly
- O(n²) complexity
- Produces reasonable triangles

**DELAUNAY**:
- Best triangle quality
- Most expensive
- Maximizes minimum angles

### Loading Functions

#### `load_obj(filename, options)`

Main loading function.

```cpp
obj_data load_obj(
    const std::string& filename,
    const load_options& options = load_options{}
);
```

**Parameters:**
- `filename`: Path to OBJ file
- `options`: Loading configuration

**Returns:** Parsed `obj_data` structure

**Throws:** `std::runtime_error` on file errors or parse failures

**Example:**
```cpp
// Basic loading with defaults
auto data = gnnmath::load_obj("model.obj");

// Custom options
gnnmath::load_options opts;
opts.triangulate = true;
opts.generate_normals = true;
opts.scale = 0.01;  // Convert cm to m
auto scaled_data = gnnmath::load_obj("model.obj", opts);
```

### Material Support

The loader parses MTL (Material Library) files referenced by OBJ files.

```cpp
struct material {
    std::string name;
    std::array<scalar_t, 3> ambient;   // Ka
    std::array<scalar_t, 3> diffuse;   // Kd
    std::array<scalar_t, 3> specular;  // Ks
    scalar_t shininess;                 // Ns
    scalar_t opacity;                   // d or Tr
    std::string diffuse_map;           // map_Kd texture path
    // ... other properties
};
```

### Index Handling

OBJ files use 1-based indexing and support relative (negative) indices:

```obj
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 3      # Absolute: vertices 1, 2, 3
f -3 -2 -1   # Relative: last 3 vertices
```

The loader converts all indices to 0-based absolute indices.

---

## Mesh Representation (`mesh.hpp`)

The `mesh` class is the central data structure for 3D mesh operations, providing topology information and efficient neighbor queries.

### Class Definition

```cpp
namespace gnnmath {
    class mesh {
    private:
        // Core geometry
        std::vector<std::array<scalar_t, 3>> vertices_;
        std::vector<std::pair<index_t, index_t>> edges_;
        std::vector<std::array<index_t, 3>> faces_;

        // Optional data
        std::vector<std::array<scalar_t, 2>> texcoords_;
        std::vector<std::array<scalar_t, 3>> file_normals_;

        // Topology acceleration structures
        std::map<index_t, std::vector<index_t>> adjacency_;
        std::map<index_t, std::vector<index_t>> incident_edges_;
        std::map<std::pair<index_t, index_t>, index_t> edge_index_map_;

    public:
        // Construction
        mesh() = default;
        void load_obj(const std::string& filename);

        // Accessors
        const auto& vertices() const;
        const auto& edges() const;
        const auto& faces() const;
        index_t num_vertices() const;
        index_t num_edges() const;
        index_t num_faces() const;

        // Topology queries
        const std::vector<index_t>& get_neighbors(index_t v) const;
        const std::vector<index_t>& get_incident_edges(index_t v) const;
        index_t get_edge_index(index_t u, index_t v) const;

        // Feature computation
        feature_matrix_t compute_node_features() const;
        feature_matrix_t compute_edge_features() const;
        std::vector<std::array<scalar_t, 3>> compute_normals() const;

        // Conversion
        sparse_matrix to_adjacency_matrix() const;

        // Utilities
        bool is_valid() const;
        std::vector<index_t> sample_vertices(index_t n) const;
        void add_vertex_noise(scalar_t scale);
    };
}
```

### Topology Data Structures

#### Adjacency Map

```cpp
adjacency_[v] = {neighbors of vertex v}
```

Enables O(1) lookup of vertex neighbors (amortized).

#### Incident Edges Map

```cpp
incident_edges_[v] = {edge indices incident to vertex v}
```

Enables efficient iteration over edges connected to a vertex.

#### Edge Index Map

```cpp
edge_index_map_[{u, v}] = edge_index
```

Where u < v (canonical ordering). Enables O(log E) edge lookup.

### Mesh Construction

When loading an OBJ file, the mesh:

1. **Extracts vertices** from OBJ vertex list
2. **Builds faces** from OBJ face definitions
3. **Extracts edges** from face adjacencies (each face edge added once)
4. **Builds adjacency** map from edges
5. **Builds incident edge** map for each vertex
6. **Creates edge index** map for O(log E) lookup

```cpp
// Example: Building edge set from faces
for (const auto& face : faces) {
    for (int i = 0; i < 3; ++i) {
        index_t u = face[i];
        index_t v = face[(i + 1) % 3];
        if (u > v) std::swap(u, v);  // Canonical order
        edges.insert({u, v});
    }
}
```

### Computing Features

#### `compute_node_features()`

Returns vertex positions as feature vectors.

```cpp
feature_matrix_t compute_node_features() const;
```

**Returns:** Matrix of size [num_vertices × 3], each row is (x, y, z)

#### `compute_edge_features()`

Returns edge lengths as feature vectors.

```cpp
feature_matrix_t compute_edge_features() const;
```

**Returns:** Matrix of size [num_edges × 1], each row is edge length

#### `compute_normals()`

Computes per-vertex normals by averaging incident face normals.

```cpp
std::vector<std::array<scalar_t, 3>> compute_normals() const;
```

**Algorithm:**
1. For each face, compute face normal via cross product
2. For each vertex, accumulate normals of incident faces
3. Normalize each vertex normal

### Validation

```cpp
bool is_valid() const;
```

Checks:
- All face vertex indices are valid
- All edge vertex indices are valid
- Adjacency structure is consistent
- No degenerate faces (repeated vertices)

### Example Usage

```cpp
gnnmath::mesh m;
m.load_obj("bunny.obj");

std::cout << "Vertices: " << m.num_vertices() << "\n";
std::cout << "Edges: " << m.num_edges() << "\n";
std::cout << "Faces: " << m.num_faces() << "\n";

// Get neighbors of vertex 0
for (index_t neighbor : m.get_neighbors(0)) {
    std::cout << "Neighbor: " << neighbor << "\n";
}

// Convert to sparse adjacency for GNN
auto adj = m.to_adjacency_matrix();

// Compute features for GNN input
auto node_features = m.compute_node_features();
auto edge_features = m.compute_edge_features();
```

---

## Feature Extraction (`feature_extraction.hpp`)

The feature extraction module computes geometric features from meshes for use as GNN input.

### Gaussian Curvature

#### `compute_gaussian_curvature(mesh)`

Computes discrete Gaussian curvature at each vertex using the Gauss-Bonnet theorem.

```cpp
std::vector<scalar_t> compute_gaussian_curvature(const mesh& m);
```

**Mathematical Background:**

For a smooth surface, Gaussian curvature K is the product of principal curvatures:
```
K = κ₁ × κ₂
```

For discrete meshes, we use the **angle defect** formula:

```
K(v) = (2π - Σᵢ θᵢ) / A(v)
```

Where:
- θᵢ = angle at vertex v in incident triangle i
- A(v) = local area around vertex v (1/3 of incident triangle areas)

**Algorithm:**

```cpp
for each vertex v:
    angle_sum = 0
    area = 0

    for each incident face f:
        // Find angle at v in face f
        // Face has vertices (v, a, b)
        vec_a = normalize(a - v)
        vec_b = normalize(b - v)
        angle = acos(dot(vec_a, vec_b))
        angle_sum += angle

        // Add 1/3 of face area
        area += triangle_area(v, a, b) / 3

    // Angle defect
    curvature[v] = (2π - angle_sum) / area
```

**Interpretation:**
- K > 0: Elliptic point (bowl-like, e.g., sphere)
- K < 0: Hyperbolic point (saddle-like)
- K = 0: Parabolic point (flat or cylindrical)

**Example:**
```cpp
auto curvatures = gnnmath::compute_gaussian_curvature(mesh);

// Find high-curvature vertices (features/corners)
for (index_t i = 0; i < curvatures.size(); ++i) {
    if (std::abs(curvatures[i]) > threshold) {
        std::cout << "Feature vertex: " << i << "\n";
    }
}
```

### Node Features

#### `compute_node_features(mesh)`

Computes comprehensive per-vertex features combining position, normal, and curvature.

```cpp
feature_matrix_t compute_node_features(const mesh& m);
```

**Returns:** Matrix of size [num_vertices × 7]

**Feature Vector:**
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | x | Vertex x coordinate |
| 1 | y | Vertex y coordinate |
| 2 | z | Vertex z coordinate |
| 3 | nx | Normal x component |
| 4 | ny | Normal y component |
| 5 | nz | Normal z component |
| 6 | κ | Gaussian curvature |

**Example:**
```cpp
auto features = gnnmath::compute_node_features(mesh);

// Feature vector for vertex 0
const auto& f = features[0];
std::cout << "Position: (" << f[0] << ", " << f[1] << ", " << f[2] << ")\n";
std::cout << "Normal: (" << f[3] << ", " << f[4] << ", " << f[5] << ")\n";
std::cout << "Curvature: " << f[6] << "\n";
```

### Edge Features

#### `compute_edge_features(mesh)`

Computes per-edge features including length and normal angle.

```cpp
feature_matrix_t compute_edge_features(const mesh& m);
```

**Returns:** Matrix of size [num_edges × 2]

**Feature Vector:**
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | length | Euclidean distance between endpoints |
| 1 | θ | Angle between endpoint normals |

**Normal Angle Computation:**
```cpp
// For edge (u, v):
normal_u = vertex_normal[u]
normal_v = vertex_normal[v]
cos_theta = dot(normal_u, normal_v)
theta = acos(clamp(cos_theta, -1, 1))
```

The normal angle indicates edge sharpness:
- θ ≈ 0: Smooth edge (normals parallel)
- θ ≈ π: Sharp crease (normals opposite)

**Example:**
```cpp
auto edge_features = gnnmath::compute_edge_features(mesh);

// Find sharp edges
for (index_t i = 0; i < edge_features.size(); ++i) {
    scalar_t angle = edge_features[i][1];
    if (angle > M_PI / 4) {  // > 45 degrees
        std::cout << "Sharp edge: " << i << "\n";
    }
}
```

---

## Mesh Processor (`mesh_processor.hpp`)

The mesh processor implements mesh simplification algorithms, including GNN-driven edge collapse.

### Quadric Error Metric

#### `compute_quadric_error(mesh, u, v)`

Computes the error metric for collapsing edge (u, v).

```cpp
scalar_t compute_quadric_error(const mesh& m, index_t u, index_t v);
```

**Current Implementation:** Returns edge length as a simple proxy.

**Future Extension:** Full Garland-Heckbert quadric error matrices:

```
Q(v) = Σ Kₚ   (sum of fundamental error quadrics for incident planes)
Error(v) = vᵀ Q v
```

### GNN-Driven Simplification

#### `simplify_with_gnn_scores(mesh, target, scores)`

Simplifies mesh using GNN-predicted edge collapse priorities.

```cpp
mesh simplify_with_gnn_scores(
    mesh& m,
    index_t target_vertices,
    const std::vector<scalar_t>& scores
);
```

**Parameters:**
- `m`: Input mesh (modified in place)
- `target_vertices`: Desired vertex count after simplification
- `scores`: Per-edge priority scores from GNN (lower = collapse first)

**Algorithm:**

```
1. Initialize priority queue with (score, edge) pairs
2. Initialize version counter for each edge (for lazy deletion)

3. While num_vertices > target:
   a. Pop lowest-score edge (u, v) from queue
   b. If edge is stale (version mismatch), skip
   c. If edge no longer exists, skip

   d. Collapse edge:
      - Move v to midpoint of (u, v)
      - Redirect all edges from u to v
      - Remove u from mesh
      - Mark affected edges as stale

   e. Re-insert affected edges with updated scores

4. Remove degenerate faces (faces with repeated vertices)
5. Compact vertex indices
```

**Complexity:** O(E log E) using priority queue with lazy deletion

**Lazy Deletion Strategy:**

Instead of removing stale edges from the heap (expensive), we:
1. Keep a version number for each edge
2. Increment version when edge is modified
3. When popping, check if version matches - skip if stale

```cpp
struct edge_entry {
    scalar_t score;
    index_t edge_id;
    index_t version;
};

// When edge is modified:
edge_versions[edge_id]++;
queue.push({new_score, edge_id, edge_versions[edge_id]});

// When popping:
auto entry = queue.pop();
if (entry.version != edge_versions[entry.edge_id]) {
    continue;  // Stale entry
}
```

### Random Removal (Baseline)

#### `simplify_random_removal(mesh, target)`

Baseline simplification by random vertex removal.

```cpp
mesh simplify_random_removal(mesh& m, index_t target_vertices);
```

**Algorithm:**
1. Randomly select vertices for removal
2. Remove selected vertices and incident faces
3. No topology preservation guarantees

**Use Case:** Baseline comparison for GNN-driven simplification.

### Edge Collapse Operation

The core edge collapse operation for (u, v):

```
Before:                After:
    a                     a
   /|\                   /|
  / | \                 / |
 /  |  \               /  |
b---u---c    -->      b---v'--c
 \  |  /               \  |  /
  \ | /                 \ | /
   \|/                   \|/
    d                     d

v' = (u + v) / 2  (midpoint)
```

**Steps:**
1. Compute new position (midpoint or optimized)
2. Update v's position to new position
3. Redirect all edges (u, x) to (v, x)
4. Remove vertex u
5. Remove degenerate faces (where two vertices became the same)
6. Update adjacency structures

### Example Usage

```cpp
// Load mesh
gnnmath::mesh m;
m.load_obj("high_poly.obj");

std::cout << "Original: " << m.num_vertices() << " vertices\n";

// Compute GNN scores (lower = collapse first)
auto features = gnnmath::compute_node_features(m);
auto adj = m.to_adjacency_matrix();
auto scores = pipeline->process(features, adj);

// Flatten scores to per-edge
std::vector<scalar_t> edge_scores;
for (const auto& [u, v] : m.edges()) {
    // Combine endpoint scores
    edge_scores.push_back(scores[u][0] + scores[v][0]);
}

// Simplify to 50% vertices
index_t target = m.num_vertices() / 2;
gnnmath::simplify_with_gnn_scores(m, target, edge_scores);

std::cout << "Simplified: " << m.num_vertices() << " vertices\n";
```

---

## Geometry Pipeline Integration

The geometry module integrates with the full MeshGNN pipeline:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  OBJ File   │────▶│  OBJ Loader │────▶│    Mesh     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
           ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
           │ Node Features │          │ Edge Features │          │   Adjacency   │
           │  (7-dim)      │          │   (2-dim)     │          │   (Sparse)    │
           └───────────────┘          └───────────────┘          └───────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Graph     │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │GNN Pipeline │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Scores    │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │Mesh Process │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │Simplified   │
                                        │   Mesh      │
                                        └─────────────┘
```

---

## Performance Considerations

### OBJ Loading
- Large files benefit from memory-mapped I/O
- Triangulation adds O(n) overhead per polygon
- Normal generation is O(V + F)

### Mesh Operations
- Adjacency queries: O(1) amortized
- Edge index lookup: O(log E)
- Feature extraction: O(V + E + F)

### Simplification
- Priority queue approach: O(E log E)
- Lazy deletion avoids O(E) heap updates
- Memory: O(V + E) for version tracking

### Memory Layout
- Vertices: Contiguous array for cache efficiency
- Maps: Balanced trees for worst-case guarantees
- Consider flat hash maps for better average performance
