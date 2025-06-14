#include "include/gnnmath/mesh.hpp" // mesh
#include "include/gnnmath/feature_extraction.hpp"
#include "include/gnnmath/mesh_processor.hpp"
#include "include/gnnmath/vector.hpp"
#include <iostream>
#include <string>
#include <stdexcept>

int main(int argc, char* argv[]) {
    std::string filename;

    // Check for command-line argument
    if (argc > 1) {
        filename = argv[1];
    } 
    else {
        // Prompt user for file name
        std::cout << "Enter path to OBJ file (e.g., data/tetrahedron.obj): ";
        std::getline(std::cin, filename);
        if (filename.empty()) {
            filename = "data/tetrahedron.obj"; // Default file
            std::cout << "Using default file: " << filename << "\n";
        }
    }

    try {
        // Load mesh
        gnnmath::mesh::mesh m;
        m.load_obj(filename);
        if (!m.is_valid()) {
            throw std::runtime_error("Invalid mesh loaded");
        }
        std::cout << "Loaded mesh with " << m.n_vertices() << " vertices, "
                  << m.n_edges() << " edges, " << m.n_faces() << " faces\n";

        // Extract Gaussian curvature
        auto curvatures = gnnmath::mesh::compute_gaussian_curvature(m);
        std::cout << "Computed " << curvatures.size() << " Gaussian curvatures\n";
        if (!curvatures.empty()) {
            std::cout << "Sample curvature (vertex 0): " << curvatures[0] << "\n";
        }

        // Extract combined node features (coordinates, normals, curvature)
        std::vector<gnnmath::vector::vector> node_features = gnnmath::mesh::compute_combined_node_features(m);
        std::cout << "Extracted " << node_features.size() << " node features (dimension: "
                  << (node_features.empty() ? 0 : node_features[0].size()) << ")\n";
        if (!node_features.empty()) {
            std::cout << "Sample node feature (vertex 0): [";
            for (size_t i = 0; i < node_features[0].size(); ++i) {
                std::cout << node_features[0][i];
                if (i < node_features[0].size() - 1) std::cout << ", ";
            }
            
            std::cout << "]\n";
        }

        // Extract neighbor edge features (length, angle)
        auto edge_features = gnnmath::mesh::compute_neighbor_edge_features(m);
        std::cout << "Extracted " << edge_features.size() << " edge features (dimension: "
                  << (edge_features.empty() ? 0 : edge_features[0].size()) << ")\n";

        // Compute quadric error for a sample edge (if edges exist)
        if (m.n_edges() > 0) {
            auto edges = m.edges();
            auto [u, v] = edges[0]; // First edge
            auto error = gnnmath::mesh::compute_quadric_error(m, u, v);
            std::cout << "Quadric error for edge (" << u << ", " << v << "): " << error << "\n";
        }

        // Simplify mesh
        std::size_t target_vertices = m.n_vertices() / 2;
        gnnmath::mesh::simplify_random_removal(m, target_vertices);
        std::cout << "Simplified mesh to " << m.n_vertices() << " vertices\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}