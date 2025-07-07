#include "render.h"

#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 800

// --- Shader Source Code (Unchanged) ---
const char* vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec2 aPos;

    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * vec4(aPos.x, aPos.y, 0.0, 1.0);
        gl_PointSize = 1.0;
    }
)glsl";

const char* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
)glsl";


// --- Core Logic (Unchanged) ---
Matrix ProjectionMat() {
    const float NEAR = 10.0f;
    const float FAR = 50.0f;
    const double FOV = 100.0 * M_PI / 180.0;
    const double aspect = (double)WINDOW_WIDTH / (double)WINDOW_HEIGHT;

    Matrix proj = Matrix({
        {1 / (float)(aspect * tan(FOV / 2)), 0                        , 0                     , 0},
        {0                                 , 1 / (float)(tan(FOV / 2)), 0                     , 0},
        {0                                 , 0                        , -(FAR+NEAR)/(FAR-NEAR), -(2*FAR*NEAR)/(FAR-NEAR)},
        {0                                 , 0                        , -1.0                  , 0}
    });
    return proj;
}

inline void DrawPoint(std::vector<float>& points, float centerX, float centerY) {
    points.push_back(centerX); points.push_back(centerY);
}

// --- Helper function for compiling shaders (Unchanged) ---
GLuint createShaderProgram() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}


void testMatrix() {
    Matrix a = {{1, 2, 3}, {4, 5, 6}};

    Matrix b = {{1, 2}, {3, 4}, {5, 6}};

    std::cout << "A: " << a.rows << " " << a.cols << std::endl;
    a.print();

    std::cout << "B: " << b.rows << " " << b.cols << std::endl;
    b.print();

    Matrix c = a*b;

    c.print();

    Matrix* d = new Matrix(4, 1);
    d->print();
    // std::cout << d->rows << std::endl;

    delete d;

    std::cout << "---------" << std::endl;

    Vec2 v2 = {{1,2,3}, {3, 4,5 }};
    v2.print();
}

// int main() {
//     testMatrix();
// }

int main(int argc,  char* argv[]) {
    Object obj(0, -20, -20);

    if (argc > 1) {
        std::ifstream file(argv[1]);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << argv[1] << std::endl;
            return -1;
        }
        file >> obj;
        file.close();
        
        // Print object vertices for debugging
        std::cout << "Object vertices:" << obj.vertices.size() << std::endl;
        std::cout << "faces: " << obj.faces.size() << std::endl;
        // return 0;
    } else {
        std::cerr << "Usage: " << argv[0] << " <path_to_object_file>" << std::endl;
        return -1;
    }

    // --- Initialization ---
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        return -1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_Window* window = SDL_CreateWindow(
        "OpenGL Renderer (GLEW)",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
    );

    if (!window) {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        SDL_Log("Failed to create OpenGL context: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // disable vsync
    SDL_GL_SetSwapInterval(0);
    
    // --- GLEW Initialization ---
    // This must be done *after* creating the OpenGL context.
    glewExperimental = GL_TRUE; // Needed for core profile
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
        return -1;
    }


    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // --- Modern OpenGL Setup (Unchanged) ---
    GLuint shaderProgram = createShaderProgram();
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    // --- Application State (Unchanged) ---
    bool running = true;
    SDL_Event event;
    Camera* cam = new Camera(0, 0, 0);
    int counter = 0;
    
    
    // --- Render Loop (Unchanged) ---
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_UP:    cam->position.vz -= 0.5; break;
                    case SDLK_DOWN:  cam->position.vz += 0.5; break;
                    case SDLK_LEFT:  cam->position.vx -= 0.5; break;
                    case SDLK_RIGHT: cam->position.vx += 0.5; break;
                }
            }
        }

        auto timer_start = std::chrono::high_resolution_clock::now();
        std::vector<float> pointsToDraw;
        // pointsToDraw.reserve(100 * 8 * 8 * 2);

        for (auto &vertex : obj.vertices) {
            Vec4 relative_vec = vertex + obj.pos - cam->position;
            Vec4 clip_vec = ProjectionMat() * relative_vec;

            if (clip_vec.vw > 0) {
                Vec4 perspective_vec = clip_vec * (1 / clip_vec.vw);
                float x_screen = WINDOW_WIDTH / 2.0f * perspective_vec.vx;
                float y_screen = WINDOW_HEIGHT / 2.0f * perspective_vec.vy;
                DrawPoint(pointsToDraw, x_screen, y_screen);
            }
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        if (!pointsToDraw.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, pointsToDraw.size() * sizeof(float), pointsToDraw.data(), GL_DYNAMIC_DRAW);
            glUseProgram(shaderProgram);
            Matrix ortho_proj = Ortho((float)WINDOW_WIDTH/-2, (float)WINDOW_WIDTH/2, (float)WINDOW_HEIGHT/-2, (float)WINDOW_HEIGHT/2, -1.0f, 1.0f);
            GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
            glUniformMatrix4fv(projLoc, 1, GL_FALSE, ortho_proj.value_ptr());
            glBindVertexArray(VAO);
            glEnable(GL_PROGRAM_POINT_SIZE);
            glDrawArrays(GL_POINTS, 0, pointsToDraw.size() / 2);
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        SDL_GL_SwapWindow(window);

        auto timer_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> latency = timer_end - timer_start;
        if (counter == 0) {
            std::cout << "FPS: " << 1.0 / latency.count() << std::endl;
        }
        counter = (counter + 1) % 20;
    }

    // --- Cleanup ---
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    delete cam;
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
