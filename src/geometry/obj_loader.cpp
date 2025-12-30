#include <gnnmath/geometry/obj_loader.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>

namespace gnnmath {
namespace mesh {

namespace {

std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    
    auto end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::vector<std::string> split_whitespace(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, delim)) {
        tokens.push_back(token);
    }

    return tokens;
}

scalar_t parse_number(const std::string& str, scalar_t default_val = 0.0) {
    if (str.empty()) {
        return default_val;
    }
    
    try {
        return std::stod(str);
    } catch (...) {
        return default_val;
    }
}

int parse_int(const std::string& str, int default_val = 0) {
    if (str.empty()) {
        return default_val;
    }
    
    try {
        return std::stoi(str);
    } catch (...) {
        return default_val;
    }
}

std::string get_directory(const std::string& filepath) {
    std::filesystem::path p(filepath);
    return p.parent_path().string();
}

std::string join_path(const std::string& dir, const std::string& filename) {
    if (dir.empty()) {
        return filename;
    }
    
    std::filesystem::path p(dir);
    p /= filename;
    
    return p.string();
}

} // anonymous namespace

obj_data obj_loader::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("obj_loader: cannot open file '" + filename + "'");
    }

    base_path_ = get_directory(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();

    return load_from_string(buffer.str(), base_path_);
}

obj_data obj_loader::load_from_string(const std::string& content, const std::string& base_path) {
    obj_data data;
    warnings_.clear();
    stats_ = obj_stats{};
    line_number_ = 0;
    current_group_ = "default";
    current_material_ = "";
    base_path_ = base_path;

    // Create default group
    data.groups.push_back({"default", "", {}});

    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        ++line_number_;
        parse_line(line, data);
    }

    // Update statistics
    stats_.vertex_count = data.vertices.size();
    stats_.texcoord_count = data.texcoords.size();
    stats_.normal_count = data.normals.size();
    stats_.face_count = data.faces.size();
    stats_.group_count = data.groups.size();
    stats_.material_count = data.materials.size();
    stats_.line_count = data.lines.size();
    stats_.has_texcoords = !data.texcoords.empty();
    stats_.has_normals = !data.normals.empty();

    for (const auto& face : data.faces) {
        if (face.size() == 3) {
            ++stats_.triangle_count;
        } 
        else if (face.size() == 4) {
            ++stats_.quad_count;
        } 
        else if (face.size() > 4) {
            ++stats_.ngon_count;
        }
    }

    // Post-process (triangulation, normal generation, etc.)
    post_process(data);
    return data;
}

void obj_loader::parse_line(const std::string& line, obj_data& data) {
    std::string trimmed = trim(line);

    // empty lines and comments
    if (trimmed.empty() || trimmed[0] == '#') {
        return;
    }

    // Handle line continuation (backslash at end)
    // Note: This is handled implicitly since we read line by line

    // Get the command prefix
    auto tokens = split_whitespace(trimmed);
    if (tokens.empty()) {
        return;
    }

    const std::string& cmd = tokens[0];
    if (cmd == "v") {
        parse_vertex(trimmed, data);
    } 
    else if (cmd == "vt") {
        parse_texcoord(trimmed, data);
    } 
    else if (cmd == "vn") {
        parse_normal(trimmed, data);
    } 
    else if (cmd == "f") {
        parse_face(trimmed, data);
    } 
    else if (cmd == "l") {
        parse_line_element(trimmed, data);
    } 
    else if (cmd == "g") {
        parse_group(trimmed, data, false);
    } 
    else if (cmd == "o") {
        parse_group(trimmed, data, true);
    } 
    else if (cmd == "mtllib") {
        parse_mtllib(trimmed, data);
    } 
    else if (cmd == "usemtl") {
        parse_usemtl(trimmed, data);
    } 
    else if (cmd == "s") {
        // Smoothing group - we ignore this but don't warn
    } 
    else if (cmd == "vp") {
        // Parameter space vertex - rarely used, ignore
        warn("line " + std::to_string(line_number_) + ": ignoring parameter space vertex (vp)");
    } 
    else if (cmd[0] == 'v' || cmd[0] == 'f' || cmd[0] == 'l') {
        // Unknown vertex/face/line variant
        warn("line " + std::to_string(line_number_) + ": unknown command '" + cmd + "'");
    }
    // Silently ignore other commands (materials, etc.)
}

void obj_loader::parse_vertex(const std::string& line, obj_data& data) {
    auto tokens = split_whitespace(line);
    if (tokens.size() < 4) {
        if (options_.strict_mode) {
            throw std::runtime_error("line " + std::to_string(line_number_) +
                                     ": invalid vertex - need at least 3 coordinates");
        }

        warn("line " + std::to_string(line_number_) + ": invalid vertex format");
        return;
    }

    scalar_t x = parse_number(tokens[1]);
    scalar_t y = parse_number(tokens[2]);
    scalar_t z = parse_number(tokens[3]);

    // Handle homogeneous coordinate (w)
    if (tokens.size() > 4) {
        scalar_t w = parse_number(tokens[4], 1.0);
        if (std::abs(w) > 1e-10) {
            x /= w;
            y /= w;
            z /= w;
        }
    }

    // Validate
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        if (options_.strict_mode) {
            throw std::runtime_error("line " + std::to_string(line_number_) +
                                     ": non-finite vertex coordinate");
        }

        warn("line " + std::to_string(line_number_) + ": non-finite vertex coordinate, using (0,0,0)");
        x = y = z = 0.0;
    }

    data.vertices.push_back({x, y, z});
}

void obj_loader::parse_texcoord(const std::string& line, obj_data& data) {
    auto tokens = split_whitespace(line);
    if (tokens.size() < 2) {
        warn("line " + std::to_string(line_number_) + ": invalid texture coordinate");
        return;
    }

    scalar_t u = parse_number(tokens[1]);
    scalar_t v = tokens.size() > 2 ? parse_number(tokens[2]) : 0.0;
    // w coordinate (tokens[3]) is ignored for 2D textures
    if (options_.flip_texcoords_v) {
        v = 1.0 - v;
    }

    data.texcoords.push_back({u, v});
}

void obj_loader::parse_normal(const std::string& line, obj_data& data) {
    auto tokens = split_whitespace(line);
    if (tokens.size() < 4) {
        warn("line " + std::to_string(line_number_) + ": invalid normal");
        return;
    }

    scalar_t x = parse_number(tokens[1]);
    scalar_t y = parse_number(tokens[2]);
    scalar_t z = parse_number(tokens[3]);
    if (options_.flip_normals) {
        x = -x;
        y = -y;
        z = -z;
    }

    // Normalize
    scalar_t len = std::sqrt(x * x + y * y + z * z);
    if (len > 1e-10) {
        x /= len;
        y /= len;
        z /= len;
    }

    data.normals.push_back({x, y, z});
}

void obj_loader::parse_face(const std::string& line, obj_data& data) {
    auto tokens = split_whitespace(line);
    if (tokens.size() < 4) {  // f + at least 3 vertices
        warn("line " + std::to_string(line_number_) + ": face needs at least 3 vertices");
        return;
    }

    polygon_face face;
    face.reserve(tokens.size() - 1);
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        try {
            face.push_back(parse_face_vertex(tokens[i], data));
        } catch (const std::exception& e) {
            if (options_.strict_mode) {
                throw;
            }

            warn("line " + std::to_string(line_number_) + ": " + e.what());
            return;
        }
    }

    if (face.size() < 3) {
        warn("line " + std::to_string(line_number_) + ": degenerate face");
        return;
    }

    // Add face to data
    std::size_t face_idx = data.faces.size();
    data.faces.push_back(std::move(face));

    // Add to current group
    if (!data.groups.empty()) {
        data.groups.back().face_indices.push_back(face_idx);
    }
}

face_vertex obj_loader::parse_face_vertex(const std::string& token, const obj_data& data) {
    face_vertex fv;
    fv.vertex_idx = 0;
    auto parts = split(token, '/');
    if (parts.empty() || parts[0].empty()) {
        throw std::runtime_error("invalid face vertex '" + token + "'");
    }

    // Parse vertex index
    int v_idx = parse_int(parts[0]);
    if (v_idx == 0) {
        throw std::runtime_error("invalid vertex index in '" + token + "'");
    }

    fv.vertex_idx = resolve_index(v_idx, data.vertices.size());

    // Parse texture coordinate index (if present)
    if (parts.size() > 1 && !parts[1].empty()) {
        int vt_idx = parse_int(parts[1]);
        if (vt_idx != 0) {
            fv.texcoord_idx = resolve_index(vt_idx, data.texcoords.size());
        }
    }

    // Parse normal index (if present)
    if (parts.size() > 2 && !parts[2].empty()) {
        int vn_idx = parse_int(parts[2]);
        if (vn_idx != 0) {
            fv.normal_idx = resolve_index(vn_idx, data.normals.size());
        }
    }

    return fv;
}

void obj_loader::parse_line_element(const std::string& line, obj_data& data) {
    auto tokens = split_whitespace(line);
    if (tokens.size() < 3) {  // l + at least 2 vertices
        warn("line " + std::to_string(line_number_) + ": line element needs at least 2 vertices");
        return;
    }

    // Parse vertex indices (may include texture coords as v/vt)
    std::vector<std::size_t> indices;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        auto parts = split(tokens[i], '/');
        if (parts.empty()) {
            continue;
        }

        int v_idx = parse_int(parts[0]);
        if (v_idx == 0) {
            continue;
        }

        indices.push_back(resolve_index(v_idx, data.vertices.size()));
    }

    // Create line segments
    for (std::size_t i = 0; i + 1 < indices.size(); ++i) {
        data.lines.push_back({indices[i], indices[i + 1]});
    }
}

void obj_loader::parse_group(const std::string& line, obj_data& data, bool /*is_object*/) {
    if (options_.ignore_groups) {
        return;
    }

    auto tokens = split_whitespace(line);
    std::string name = tokens.size() > 1 ? tokens[1] : "unnamed";
    current_group_ = name;

    // Create new group
    object_group group;
    group.name = name;
    group.material_name = current_material_;
    data.groups.push_back(std::move(group));
}

void obj_loader::parse_mtllib(const std::string& line, obj_data& data) {
    if (!options_.load_materials) {
        return;
    }

    auto tokens = split_whitespace(line);
    if (tokens.size() < 2) {
        return;
    }

    // MTL filename may contain spaces, so join remaining tokens
    std::string mtl_filename;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        if (i > 1) {
            mtl_filename += " ";
        }
        
        mtl_filename += tokens[i];
    }

    data.mtl_filename = mtl_filename;

    // Try to load MTL file
    std::string mtl_path = join_path(base_path_, mtl_filename);
    try {
        load_mtl(mtl_path, data);
    } catch (const std::exception& e) {
        warn("could not load MTL file '" + mtl_path + "': " + e.what());
    }
}

void obj_loader::parse_usemtl(const std::string& line, obj_data& data) {
    auto tokens = split_whitespace(line);
    if (tokens.size() < 2) {
        return;
    }

    current_material_ = tokens[1];

    // Update current group's material
    if (!data.groups.empty()) {
        data.groups.back().material_name = current_material_;
    }
}

void obj_loader::load_mtl(const std::string& filename, obj_data& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open MTL file");
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    parse_mtl_content(buffer.str(), data);
}

void obj_loader::parse_mtl_content(const std::string& content, obj_data& data) {
    std::istringstream stream(content);
    std::string line;
    material* current_mat = nullptr;
    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        auto tokens = split_whitespace(line);
        if (tokens.empty()) {
            continue;
        }

        const std::string& cmd = tokens[0];
        if (cmd == "newmtl" && tokens.size() > 1) {
            std::string name = tokens[1];
            material new_mat;
            new_mat.name = name;
            data.materials[name] = new_mat;
            current_mat = &data.materials[name];
        } 
        else if (current_mat) {
            if (cmd == "Ka" && tokens.size() >= 4) {
                current_mat->ambient = {
                    parse_number(tokens[1]),
                    parse_number(tokens[2]),
                    parse_number(tokens[3])
                };
            } 
            else if (cmd == "Kd" && tokens.size() >= 4) {
                current_mat->diffuse = {
                    parse_number(tokens[1]),
                    parse_number(tokens[2]),
                    parse_number(tokens[3])
                };
            } 
            else if (cmd == "Ks" && tokens.size() >= 4) {
                current_mat->specular = {
                    parse_number(tokens[1]),
                    parse_number(tokens[2]),
                    parse_number(tokens[3])
                };
            } 
            else if (cmd == "Ns" && tokens.size() >= 2) {
                current_mat->shininess = parse_number(tokens[1]);
            } 
            else if ((cmd == "d" || cmd == "Tr") && tokens.size() >= 2) {
                current_mat->opacity = parse_number(tokens[1]);
                if (cmd == "Tr") {
                    // Tr is transparency (1-opacity)
                    current_mat->opacity = 1.0 - current_mat->opacity;
                }
            } 
            else if (cmd == "map_Kd" && tokens.size() >= 2) {
                current_mat->diffuse_map = tokens.back();  // Last token (path may have options before)
            } 
            else if ((cmd == "map_Bump" || cmd == "bump") && tokens.size() >= 2) {
                current_mat->normal_map = tokens.back();
            } 
            else if (cmd == "map_Ks" && tokens.size() >= 2) {
                current_mat->specular_map = tokens.back();
            }
        }
    }
}

std::vector<std::array<face_vertex, 3>> obj_loader::triangulate_face(const polygon_face& face, const obj_data& data) {
    if (face.size() < 3) {
        return {};
    }

    if (face.size() == 3) {
        return {{face[0], face[1], face[2]}};
    }

    // Choose triangulation method based on options
    switch (options_.tri_method) {
        case triangulation_method::FAN:
            return triangulate_fan(face, data);
        case triangulation_method::EAR_CLIPPING:
            return triangulate_ear_clipping(face, data);
        case triangulation_method::DELAUNAY:
            return triangulate_delaunay(face, data);
        default:
            return triangulate_fan(face, data);
    }
}

std::vector<std::array<face_vertex, 3>> obj_loader::triangulate_fan(
        const polygon_face& face, const obj_data& data) {
    return triangulate_delaunay(face, data);
}

namespace {
    inline scalar_t cross2d(scalar_t ax, scalar_t ay, scalar_t bx, scalar_t by) {
        return ax * by - ay * bx;
    }

    // Helper: check if point P is inside triangle ABC using barycentric coordinates
    bool point_in_triangle_2d(scalar_t px, scalar_t py,
                              scalar_t ax, scalar_t ay,
                              scalar_t bx, scalar_t by,
                              scalar_t cx, scalar_t cy) {
        scalar_t v0x = cx - ax, v0y = cy - ay;
        scalar_t v1x = bx - ax, v1y = by - ay;
        scalar_t v2x = px - ax, v2y = py - ay;

        scalar_t dot00 = v0x * v0x + v0y * v0y;
        scalar_t dot01 = v0x * v1x + v0y * v1y;
        scalar_t dot02 = v0x * v2x + v0y * v2y;
        scalar_t dot11 = v1x * v1x + v1y * v1y;
        scalar_t dot12 = v1x * v2x + v1y * v2y;

        scalar_t inv_denom = dot00 * dot11 - dot01 * dot01;
        if (std::abs(inv_denom) < 1e-12) {
            return false;
        }
        
        inv_denom = 1.0 / inv_denom;
        scalar_t u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        scalar_t v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        return (u >= 0) && (v >= 0) && (u + v <= 1);
    }

    // Helper: find the best projection plane for a 3D polygon
    // Returns 0 for XY, 1 for XZ, 2 for YZ
    int find_projection_plane(const std::vector<std::array<scalar_t, 3>>& vertices) {
        if (vertices.size() < 3) {
            return 0;
        }

        // Compute polygon normal using Newell's method
        scalar_t nx = 0, ny = 0, nz = 0;
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            std::size_t j = (i + 1) % vertices.size();
            const auto& vi = vertices[i];
            const auto& vj = vertices[j];
            nx += (vi[1] - vj[1]) * (vi[2] + vj[2]);
            ny += (vi[2] - vj[2]) * (vi[0] + vj[0]);
            nz += (vi[0] - vj[0]) * (vi[1] + vj[1]);
        }

        // Choose the plane with the largest normal component
        scalar_t anx = std::abs(nx), any = std::abs(ny), anz = std::abs(nz);
        if (anz >= anx && anz >= any) {
            return 0;  // XY plane
        }
        if (any >= anx && any >= anz) {
            return 1;  // XZ plane
        }
        
        return 2;  // YZ plane
    }

    // Helper: project 3D point to 2D based on plane index
    std::pair<scalar_t, scalar_t> project_to_2d(const std::array<scalar_t, 3>& p, int plane) {
        switch (plane) {
            case 0: return {p[0], p[1]};  // XY
            case 1: return {p[0], p[2]};  // XZ
            default: return {p[1], p[2]}; // YZ
        }
    }
}

std::vector<std::array<face_vertex, 3>> obj_loader::triangulate_ear_clipping(
        const polygon_face& face, const obj_data& data) {
    std::vector<std::array<face_vertex, 3>> triangles;
    const std::size_t n = face.size();
    if (n < 3) {
        return triangles;
    }
    
    triangles.reserve(n - 2);
    std::vector<std::array<scalar_t, 3>> vertices_3d(n);
    for (std::size_t i = 0; i < n; ++i) {
        vertices_3d[i] = data.vertices[face[i].vertex_idx];
    }

    // best projection plane
    int plane = find_projection_plane(vertices_3d);

    // Project to 2D
    std::vector<std::pair<scalar_t, scalar_t>> vertices_2d(n);
    for (std::size_t i = 0; i < n; ++i) {
        vertices_2d[i] = project_to_2d(vertices_3d[i], plane);
    }

    // Create index list for remaining vertices
    std::vector<std::size_t> indices(n);
    for (std::size_t i = 0; i < n; ++i) indices[i] = i;

    // Determine polygon winding (CW or CCW)
    scalar_t signed_area = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = (i + 1) % n;
        signed_area += vertices_2d[i].first * vertices_2d[j].second;
        signed_area -= vertices_2d[j].first * vertices_2d[i].second;
    }
    
    bool ccw = signed_area > 0;

    // Ear clipping loop
    while (indices.size() > 3) {
        bool ear_found = false;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            std::size_t prev = (i + indices.size() - 1) % indices.size();
            std::size_t next = (i + 1) % indices.size();

            std::size_t i0 = indices[prev];
            std::size_t i1 = indices[i];
            std::size_t i2 = indices[next];

            auto [ax, ay] = vertices_2d[i0];
            auto [bx, by] = vertices_2d[i1];
            auto [cx, cy] = vertices_2d[i2];

            // Check if this is a convex vertex (ear candidate)
            scalar_t cross = cross2d(bx - ax, by - ay, cx - bx, cy - by);
            bool is_convex = ccw ? (cross > 0) : (cross < 0);
            if (!is_convex) {
                continue;
            }

            // Check if any other vertex is inside this triangle
            bool has_point_inside = false;
            for (std::size_t j = 0; j < indices.size(); ++j) {
                if (j == prev || j == i || j == next) {
                    continue;
                }
                
                std::size_t idx = indices[j];
                auto [px, py] = vertices_2d[idx];
                if (point_in_triangle_2d(px, py, ax, ay, bx, by, cx, cy)) {
                    has_point_inside = true;
                    break;
                }
            }

            if (!has_point_inside) {
                // This is an ear, clip it
                triangles.push_back({face[i0], face[i1], face[i2]});
                indices.erase(indices.begin() + static_cast<std::ptrdiff_t>(i));
                ear_found = true;
                break;
            }
        }

        if (!ear_found) {
            // Degenerate polygon, fall back to simple fan triangulation
            warn("ear clipping failed, falling back to simple fan triangulation");
            std::vector<std::array<face_vertex, 3>> fan_result;
            fan_result.reserve(face.size() - 2);
            for (std::size_t i = 1; i + 1 < face.size(); ++i) {
                fan_result.push_back({face[0], face[i], face[i + 1]});
            }

            return fan_result;
        }
    }

    // last triangle
    if (indices.size() == 3) {
        triangles.push_back({face[indices[0]], face[indices[1]], face[indices[2]]});
    }

    return triangles;
}

std::vector<std::array<face_vertex, 3>> obj_loader::triangulate_delaunay(
        const polygon_face& face, const obj_data& data) {
    // Delaunay triangulation using Bowyer-Watson algorithm
    // For polygons, we first do ear clipping, then improve with edge flips
    const std::size_t n = face.size();
    if (n < 3) {
        return {};
    }
    
    if (n == 3) {
        return {{face[0], face[1], face[2]}};
    }

    // try ear clipping triangulation
    auto triangles = triangulate_ear_clipping(face, data);
    if (triangles.size() < 2) 
        return triangles;

    // Get 3D vertices and project to 2D
    std::vector<std::array<scalar_t, 3>> vertices_3d(n);
    for (std::size_t i = 0; i < n; ++i) {
        vertices_3d[i] = data.vertices[face[i].vertex_idx];
    }

    int plane = find_projection_plane(vertices_3d);
    std::vector<std::pair<scalar_t, scalar_t>> vertices_2d(n);
    for (std::size_t i = 0; i < n; ++i) {
        vertices_2d[i] = project_to_2d(vertices_3d[i], plane);
    }

    // Helper: check if point d is inside circumcircle of triangle abc
    auto in_circumcircle = [&](std::size_t ia, std::size_t ib, std::size_t ic, std::size_t id) -> bool {
        auto [ax, ay] = vertices_2d[ia];
        auto [bx, by] = vertices_2d[ib];
        auto [cx, cy] = vertices_2d[ic];
        auto [dx, dy] = vertices_2d[id];

        // Use the determinant method for circumcircle test
        scalar_t adx = ax - dx, ady = ay - dy;
        scalar_t bdx = bx - dx, bdy = by - dy;
        scalar_t cdx = cx - dx, cdy = cy - dy;

        scalar_t abdet = adx * bdy - bdx * ady;
        scalar_t bcdet = bdx * cdy - cdx * bdy;
        scalar_t cadet = cdx * ady - adx * cdy;

        scalar_t alift = adx * adx + ady * ady;
        scalar_t blift = bdx * bdx + bdy * bdy;
        scalar_t clift = cdx * cdx + cdy * cdy;

        return alift * bcdet + blift * cadet + clift * abdet > 0;
    };

    // from vertex index in face to local index
    std::unordered_map<std::size_t, std::size_t> vertex_to_local;
    for (std::size_t i = 0; i < n; ++i) {
        vertex_to_local[face[i].vertex_idx] = i;
    }

    // Edge flip iterations to improve triangulation (Lawson's algorithm)
    bool changed = true;
    int max_iterations = static_cast<int>(triangles.size() * 2);
    int iteration = 0;
    while (changed && iteration < max_iterations) {
        changed = false;
        ++iteration;
        for (std::size_t t1 = 0; t1 < triangles.size() && !changed; ++t1) {
            for (int edge = 0; edge < 3 && !changed; ++edge) {
                // edge vertices
                std::size_t e0_idx = triangles[t1][edge].vertex_idx;
                std::size_t e1_idx = triangles[t1][(edge + 1) % 3].vertex_idx;
                std::size_t opp_idx = triangles[t1][(edge + 2) % 3].vertex_idx;

                // Find adjacent triangle sharing this edge
                for (std::size_t t2 = t1 + 1; t2 < triangles.size() && !changed; ++t2) {
                    // Check if t2 shares edge (e0, e1)
                    int shared_edge = -1;
                    for (int e = 0; e < 3; ++e) {
                        std::size_t v0 = triangles[t2][e].vertex_idx;
                        std::size_t v1 = triangles[t2][(e + 1) % 3].vertex_idx;
                        if ((v0 == e0_idx && v1 == e1_idx) || (v0 == e1_idx && v1 == e0_idx)) {
                            shared_edge = e;
                            break;
                        }
                    }

                    if (shared_edge >= 0) {
                        std::size_t opp2_idx = triangles[t2][(shared_edge + 2) % 3].vertex_idx;

                        // Check Delaunay condition
                        auto it_opp = vertex_to_local.find(opp_idx);
                        auto it_opp2 = vertex_to_local.find(opp2_idx);
                        auto it_e0 = vertex_to_local.find(e0_idx);
                        auto it_e1 = vertex_to_local.find(e1_idx);

                        if (it_opp != vertex_to_local.end() && it_opp2 != vertex_to_local.end() &&
                            it_e0 != vertex_to_local.end() && it_e1 != vertex_to_local.end()) {
                            if (in_circumcircle(it_e0->second, it_e1->second, it_opp->second, it_opp2->second)) {
                                // Flip the edge
                                face_vertex fv_e0 = triangles[t1][edge];
                                face_vertex fv_e1 = triangles[t1][(edge + 1) % 3];
                                face_vertex fv_opp = triangles[t1][(edge + 2) % 3];
                                face_vertex fv_opp2 = triangles[t2][(shared_edge + 2) % 3];

                                // New triangles: (opp, opp2, e0) and (opp, e1, opp2)
                                triangles[t1] = {fv_opp, fv_opp2, fv_e0};
                                triangles[t2] = {fv_opp, fv_e1, fv_opp2};
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    return triangles;
}

std::size_t obj_loader::resolve_index(int index, std::size_t count) {
    if (index > 0) {
        // 1-based
        std::size_t idx = static_cast<std::size_t>(index - 1);
        if (idx >= count) {
            throw std::runtime_error("vertex index " + std::to_string(index) +
                                     " out of range (max " + std::to_string(count) + ")");
        }

        return idx;
    } 
    else if (index < 0) {
        // relative from end
        stats_.has_negative_indices = true;
        if (count == 0 || static_cast<std::size_t>(-index) > count) {
            throw std::runtime_error("negative vertex index " + std::to_string(index) +
                                     " out of range");
        }
        
        return count + static_cast<std::size_t>(index);
    } else {
        throw std::runtime_error("vertex index cannot be 0");
    }
}

void obj_loader::post_process(obj_data& data) {
    // Apply scaling and offset
    if (options_.scale != 1.0 || options_.offset[0] != 0 ||
        options_.offset[1] != 0 || options_.offset[2] != 0) {
        for (auto& v : data.vertices) {
            v[0] = v[0] * options_.scale + options_.offset[0];
            v[1] = v[1] * options_.scale + options_.offset[1];
            v[2] = v[2] * options_.scale + options_.offset[2];
        }
    }

    // Triangulate if requested
    if (options_.triangulate) {
        std::vector<polygon_face> triangulated;
        for (const auto& face : data.faces) {
            if (face.size() == 3) {
                triangulated.push_back(face);
            } 
            else {
                auto tris = triangulate_face(face, data);
                for (const auto& tri : tris) {
                    triangulated.push_back({tri[0], tri[1], tri[2]});
                }
            }
        }

        data.faces = std::move(triangulated);

        // Update triangle count after triangulation
        stats_.triangle_count = data.faces.size();
    }

    // Generate normals if requested and missing
    if (options_.generate_normals && data.normals.empty() && !data.faces.empty()) {
        // Compute face normals
        data.normals.resize(data.vertices.size(), {0, 0, 0});
        std::vector<int> counts(data.vertices.size(), 0);
        for (const auto& face : data.faces) {
            if (face.size() < 3) {
                continue;
            }

            const auto& v0 = data.vertices[face[0].vertex_idx];
            const auto& v1 = data.vertices[face[1].vertex_idx];
            const auto& v2 = data.vertices[face[2].vertex_idx];

            // face normal
            std::array<scalar_t, 3> e1 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
            std::array<scalar_t, 3> e2 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
            std::array<scalar_t, 3> normal = {
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            };

            // Accumulate to vertex normals
            for (const auto& fv : face) {
                data.normals[fv.vertex_idx][0] += normal[0];
                data.normals[fv.vertex_idx][1] += normal[1];
                data.normals[fv.vertex_idx][2] += normal[2];
                ++counts[fv.vertex_idx];
            }
        }

        for (std::size_t i = 0; i < data.normals.size(); ++i) {
            auto& n = data.normals[i];
            scalar_t len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (len > 1e-10) {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            }
        }

        // Update face vertices to reference generated normals
        for (auto& face : data.faces) {
            for (auto& fv : face) {
                fv.normal_idx = fv.vertex_idx;
            }
        }
    }
}

void obj_loader::warn(const std::string& msg) {
    warnings_.push_back(msg);
    if (options_.strict_mode) {
        throw std::runtime_error(msg);
    }
}

// Convenience functions
obj_data load_obj_file(const std::string& filename) {
    obj_loader loader;
    return loader.load(filename);
}

obj_data load_obj_file(const std::string& filename, const obj_load_options& options) {
    obj_loader loader(options);
    return loader.load(filename);
}

} // namespace mesh
} // namespace gnnmath
