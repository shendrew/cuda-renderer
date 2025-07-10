#include "pipeline/cpu_object_manager.h"

CPUObjectManager::CPUObjectManager(const std::vector<Vec4>& verts, Camera* cam, uint32_t vbo, uint32_t vao)
: m_vertices(verts), m_camera(cam), m_vbo(vbo), m_vao(vao) {}

void CPUObjectManager::render() {
    std::vector<float> points;
    points.reserve(m_vertices.size()*2);
    for (auto &v : m_vertices) {
        Vec4 rel = v - m_camera->position;
        Vec4 proj = m_camera->perspectiveMat * rel;
        if (proj.vw() > 0) {
            proj = proj * (1.0f/proj.vw());
            points.push_back(WINDOW_WIDTH/2.0f * proj.vx());
            points.push_back(WINDOW_HEIGHT/2.0f * proj.vy());
        }
    }

    // fill opengl buffer and draw
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);    
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, points.size()*sizeof(float), points.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, points.size()/2);
    glBindVertexArray(0);
}

