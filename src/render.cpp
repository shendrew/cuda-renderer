#include "common.h"
#include "std_includes.h"
#include "matrix.h"
#include "camera.h"
#include "object.h"
#include "pipeline/cpu_object_manager.h"
#ifdef USE_CUDA
  #include <cuda_gl_interop.h>
#endif

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


// --- copy shaders to GPU ---
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


void initSDL_GL() {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    // disable vsync
    SDL_GL_SetSwapInterval(0);
}

void initOpenGL(GLuint& shaderProgram, GLuint& VAO, GLuint& VBO) {
    // Create and compile shader program
    shaderProgram = createShaderProgram();
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Generate VBO and VAO
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Set VBO access pattern
    // index, # of components, type, normalized, block size, offset)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    
    // Enable access to the VBO
    glEnableVertexAttribArray(0);

    // Unbind buffer and VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    // set viewport coordinates
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // set compiled shader pipeline
    glUseProgram(shaderProgram);
    Matrix ortho_proj = Ortho(-WINDOW_WIDTH/2.0f, WINDOW_WIDTH/2.0f, -WINDOW_HEIGHT/2.0f, WINDOW_HEIGHT/2.0f, -1.0f, 1.0f);
    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projLoc,1,GL_FALSE,ortho_proj.value_ptr());
}

int main(int argc,  char* argv[]) {
    // --- Initialization ---
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        return -1;
    }
    
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
    initSDL_GL();

    // --- GLEW Initialization ---
    // This must be done *after* creating the OpenGL context.
    glewExperimental = GL_TRUE; // Needed for core profile
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    // --- OpenGL and viewport Setup ---
    GLuint shaderProgram, VAO, VBO;
    initOpenGL(shaderProgram, VAO, VBO);

    std::cout << "Init success! OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    // --- Read object from file ---
    Object obj(0, -20, -20);
    if (argc > 1) {
        std::ifstream file(argv[1]);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << argv[1] << std::endl;
            return -1;
        }
        file >> obj;
        file.close();
    } else {
        std::cerr << "Usage: " << argv[0] << " <path_to_object_file>" << std::endl;
        return -1;
    }

    // --- Render States ---
    bool running = true;
    SDL_Event event;
    Camera* cam = new Camera(0, 2, 10, CAMERA_FOV);
    int counter = 0;

    // Instantiate ObjectManager
    CPUObjectManager manager(obj.vertices, cam, VBO, VAO);

    // --- Render Loop ---
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_UP:    cam->position.vz() -= 0.5; break;
                    case SDLK_DOWN:  cam->position.vz() += 0.5; break;
                    case SDLK_LEFT:  cam->position.vx() -= 0.5; break;
                    case SDLK_RIGHT: cam->position.vx() += 0.5; break;
                }
            }
        }

        auto timer_start = std::chrono::high_resolution_clock::now();
        
        // write to point data to buffer
        manager.render();
        // update display buffer to window
        SDL_GL_SwapWindow(window);

        auto timer_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> latency = timer_end - timer_start;
        if (counter == 0) std::cout << "FPS: " << 1.0 / latency.count() << std::endl;
        counter = (counter + 1) % 100;
    }

    // --- Cleanup ---
    delete cam;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
