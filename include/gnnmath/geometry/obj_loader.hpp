#ifndef GNNMATH_GEOMETRY_OBJ_LOADER_HPP
#define GNNMATH_GEOMETRY_OBJ_LOADER_HPP

#include "../core/types.hpp"
#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <optional>
#include <variant>

namespace gnnmath {
namespace mesh {

/// @brief Result of parsing a single face vertex (v/vt/vn format)
struct face_vertex {
    std::size_t vertex_idx;                      ///< Vertex position index (1-based in file, converted to 0-based)
    std::optional<std::size_t> texcoord_idx;     ///< Texture coordinate index (optional)
    std::optional<std::size_t> normal_idx;       ///< Normal index (optional)
};

/// @brief A polygon face that may have more than 3 vertices
using polygon_face = std::vector<face_vertex>;

/// @brief Material definition from MTL file
struct material {
    std::string name;
    std::array<scalar_t, 3> ambient = {0.2, 0.2, 0.2};   ///< Ka
    std::array<scalar_t, 3> diffuse = {0.8, 0.8, 0.8};   ///< Kd
    std::array<scalar_t, 3> specular = {1.0, 1.0, 1.0};  ///< Ks
    scalar_t shininess = 0.0;                             ///< Ns
    scalar_t opacity = 1.0;                               ///< d or Tr
    std::string diffuse_map;                              ///< map_Kd
    std::string normal_map;                               ///< map_Bump or bump
    std::string specular_map;                             ///< map_Ks
};

/// @brief Object group within an OBJ file
struct object_group {
    std::string name;
    std::string material_name;
    std::vector<std::size_t> face_indices;  ///< Indices into obj_data::faces
};

/// @brief Raw data parsed from an OBJ file
struct obj_data {
    std::vector<std::array<scalar_t, 3>> vertices;        ///< Vertex positions (v)
    std::vector<std::array<scalar_t, 2>> texcoords;       ///< Texture coordinates (vt)
    std::vector<std::array<scalar_t, 3>> normals;         ///< Vertex normals (vn)
    std::vector<polygon_face> faces;                       ///< Polygon faces (f)
    std::vector<std::array<std::size_t, 2>> lines;        ///< Line elements (l)
    std::vector<object_group> groups;                      ///< Object/group definitions
    std::unordered_map<std::string, material> materials;  ///< Materials loaded from MTL
    std::string mtl_filename;                              ///< MTL library filename

    /// @brief Check if the data contains texture coordinates
    bool has_texcoords() const { return !texcoords.empty(); }

    /// @brief Check if the data contains normals
    bool has_normals() const { return !normals.empty(); }

    /// @brief Get total vertex count
    std::size_t vertex_count() const { return vertices.size(); }

    /// @brief Get total face count
    std::size_t face_count() const { return faces.size(); }
};

/// @brief Statistics about loaded OBJ file
struct obj_stats {
    std::size_t vertex_count = 0;
    std::size_t texcoord_count = 0;
    std::size_t normal_count = 0;
    std::size_t face_count = 0;
    std::size_t triangle_count = 0;       ///< After triangulation
    std::size_t quad_count = 0;           ///< Original quads before triangulation
    std::size_t ngon_count = 0;           ///< Faces with >4 vertices
    std::size_t group_count = 0;
    std::size_t material_count = 0;
    std::size_t line_count = 0;
    bool has_texcoords = false;
    bool has_normals = false;
    bool has_negative_indices = false;    ///< Used relative indexing
};

/// @brief Triangulation method for polygon faces
enum class triangulation_method {
    FAN,           ///< Fan triangulation - fast but only works for convex polygons
    EAR_CLIPPING,  ///< Ear clipping - handles concave polygons correctly
    DELAUNAY       ///< Delaunay triangulation - produces well-shaped triangles
};

/// @brief Options for OBJ loading behavior
struct obj_load_options {
    bool triangulate = true;              ///< Convert polygons to triangles
    triangulation_method tri_method = triangulation_method::EAR_CLIPPING; ///< Triangulation algorithm
    bool generate_normals = false;        ///< Generate normals if missing
    bool flip_normals = false;            ///< Reverse normal direction
    bool flip_texcoords_v = false;        ///< Flip V texture coordinate (1-v)
    bool load_materials = true;           ///< Load MTL file if present
    bool ignore_groups = false;           ///< Ignore group/object definitions
    bool strict_mode = false;             ///< Throw on any warning
    scalar_t scale = 1.0;                 ///< Scale factor for vertices
    std::array<scalar_t, 3> offset = {0, 0, 0};  ///< Translation offset
};

/// @brief Comprehensive OBJ file loader supporting all format variations
class obj_loader {
public:
    /// @brief Default constructor with default options
    obj_loader() = default;

    /// @brief Constructor with custom options
    /// @param options Loading options
    explicit obj_loader(const obj_load_options& options) : options_(options) {}

    /// @brief Load an OBJ file
    /// @param filename Path to the OBJ file
    /// @return Parsed OBJ data
    /// @throws std::runtime_error If file cannot be opened or has fatal errors
    obj_data load(const std::string& filename);

    /// @brief Load an OBJ file from a string
    /// @param content OBJ file content as string
    /// @param base_path Base path for resolving MTL files (optional)
    /// @return Parsed OBJ data
    obj_data load_from_string(const std::string& content, const std::string& base_path = "");

    /// @brief Get loading statistics from the last load operation
    /// @return Loading statistics
    const obj_stats& stats() const { return stats_; }

    /// @brief Get warnings from the last load operation
    /// @return List of warning messages
    const std::vector<std::string>& warnings() const { return warnings_; }

    /// @brief Get/set loading options
    obj_load_options& options() { return options_; }
    const obj_load_options& options() const { return options_; }

private:
    /// @brief Parse a single line from OBJ file
    void parse_line(const std::string& line, obj_data& data);

    /// @brief Parse vertex position (v x y z [w])
    void parse_vertex(const std::string& line, obj_data& data);

    /// @brief Parse texture coordinate (vt u [v] [w])
    void parse_texcoord(const std::string& line, obj_data& data);

    /// @brief Parse vertex normal (vn x y z)
    void parse_normal(const std::string& line, obj_data& data);

    /// @brief Parse face (f v1 v2 v3 ... or f v1/vt1 v2/vt2 ... etc)
    void parse_face(const std::string& line, obj_data& data);

    /// @brief Parse a single face vertex (v, v/vt, v/vt/vn, or v//vn)
    face_vertex parse_face_vertex(const std::string& token, const obj_data& data);

    /// @brief Parse line element (l v1 v2 ...)
    void parse_line_element(const std::string& line, obj_data& data);

    /// @brief Parse group/object definition (g/o name)
    void parse_group(const std::string& line, obj_data& data, bool is_object);

    /// @brief Parse material library reference (mtllib filename)
    void parse_mtllib(const std::string& line, obj_data& data);

    /// @brief Parse material usage (usemtl name)
    void parse_usemtl(const std::string& line, obj_data& data);

    /// @brief Load MTL file
    void load_mtl(const std::string& filename, obj_data& data);

    /// @brief Parse MTL file content
    void parse_mtl_content(const std::string& content, obj_data& data);

    /// @brief Triangulate a polygon face using the configured method
    std::vector<std::array<face_vertex, 3>> triangulate_face(const polygon_face& face, const obj_data& data);

    /// @brief Fan triangulation - uses Delaunay for better triangle quality
    std::vector<std::array<face_vertex, 3>> triangulate_fan(const polygon_face& face, const obj_data& data);

    /// @brief Ear clipping triangulation - handles concave polygons
    std::vector<std::array<face_vertex, 3>> triangulate_ear_clipping(const polygon_face& face, const obj_data& data);

    /// @brief Delaunay triangulation - produces well-shaped triangles
    std::vector<std::array<face_vertex, 3>> triangulate_delaunay(const polygon_face& face, const obj_data& data);

    /// @brief Convert negative index to positive (OBJ supports relative indexing)
    std::size_t resolve_index(int index, std::size_t count);

    /// @brief Apply post-processing (scaling, normals generation, etc.)
    void post_process(obj_data& data);

    /// @brief Add a warning message
    void warn(const std::string& msg);

    obj_load_options options_;
    obj_stats stats_;
    std::vector<std::string> warnings_;
    std::string current_group_;
    std::string current_material_;
    std::string base_path_;
    std::size_t line_number_ = 0;
};

/// @brief Convenience function to load OBJ file with default options
/// @param filename Path to the OBJ file
/// @return Parsed OBJ data
obj_data load_obj_file(const std::string& filename);

/// @brief Convenience function to load OBJ file with custom options
/// @param filename Path to the OBJ file
/// @param options Loading options
/// @return Parsed OBJ data
obj_data load_obj_file(const std::string& filename, const obj_load_options& options);

} // namespace mesh
} // namespace gnnmath

#endif // GNNMATH_GEOMETRY_OBJ_LOADER_HPP
