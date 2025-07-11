#include "object.h"

Object::Object() : pos{0, 0, 0, 1} {}
Object::Object(float x, float y, float z) : pos{x, y, z, 1} {}

std::istream& operator>>(std::istream& in, Object& obj) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments
        std::istringstream iss(line);
        std::string c;
        iss >> c;
        if (c == "v") {
            Vec4 vertex{0, 0, 0, 1};
            iss >> vertex[0][0] >> vertex[0][1] >> vertex[0][2];
            obj.vertices.push_back(vertex);
        }
        else if (c == "f"){
            std::string v;
            std::vector<int> face;
            while (iss >> v) {
                // get vertex indices from string
                size_t ind = v.find('/');
                if (ind != std::string::npos) {
                    v = v.substr(0, ind); // ignore texture and normal indices
                }
                face.push_back(std::stoi(v) - 1); // convert to zero-based index
            }
            obj.faces.push_back(face);
        }
    }

    return in;
}