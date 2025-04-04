#include "../include/render.h"

// void testMatrix() {
//     Matrix a = {{1, 2, 3}, {4, 5, 6}};

//     Matrix b = {{1, 2}, {3, 4}, {5, 6}};

//     a.print();

//     Matrix c = a*b;

//     c.print();

//     Matrix* d = new Matrix();
//     std::cout << d->rows << std::endl;

//     delete d;

//     std::cout << "---------" << std::endl;

//     Vec2 v2 = {{1,2,3}, {3, 4,5 }};
//     // Vec2 v2 = {{1,2,3}, {3, 4,5 }};
//     v2.print();
// }

// int main() {
//     testMatrix();
// }

#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 800

// vertex to screen
//  PERSPECTIVE_PROJECTION * VIEW_TRANSFORM (relative to cam) * MODEL_TRANSFORM (object attr) * VERTEX
//*  then clip to screen
//*  PERSPECTIVE DIVISION (map to normalize device location)
//*  SCREEN TRANSFORM     (translate to absolute px location)

Matrix ProjectionMat() {
    // need to clip far distance as depths become asymptotic -> reduces z-fighting
    const float NEAR = 10;
    const float FAR = 20;
    const double FOV = 100 * M_PI / 180;          // in radians
    const double aspect = WINDOW_WIDTH / WINDOW_HEIGHT;

    Matrix proj = Matrix({
        {1 / (float)(aspect * tan(FOV / 2)), 0                        , 0                     , 0},
        {0                                 , 1 / (float)(tan(FOV / 2)), 0                     , 0},
        {0                                 , 0                        , -(FAR+NEAR)/(FAR-NEAR), -(2*FAR*NEAR)/(FAR-NEAR)},
        {0                                 , 0                        , -1.0                  , 0}
    });

    return proj;
}

// Function to draw a pixel
void DrawPixel(SDL_Renderer* renderer, int x, int y) {
    SDL_RenderDrawPoint(renderer, x, y);
}

// Function to draw a circle
void DrawCircle(SDL_Renderer* renderer, int centerX, int centerY, int radius) {
    int x = 0;
    int y = radius;
    int d = 1 - radius;

    while (x <= y) {
        // Draw the 8 symmetrical points
        DrawPixel(renderer, centerX + x, centerY + y);
        DrawPixel(renderer, centerX - x, centerY + y);
        DrawPixel(renderer, centerX + x, centerY - y);
        DrawPixel(renderer, centerX - x, centerY - y);
        DrawPixel(renderer, centerX + y, centerY + x);
        DrawPixel(renderer, centerX - y, centerY + x);
        DrawPixel(renderer, centerX + y, centerY - x);
        DrawPixel(renderer, centerX - y, centerY - x);

        if (d < 0) {
            d += 2 * x + 3;
        } else {
            d += 2 * (x - y) + 5;
            y--;
        }
        x++;
    }
}

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        return -1;
    }

    // Create a window
    SDL_Window* window = SDL_CreateWindow(
        "Rendering Window",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );

    if (!window) {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    // Create a renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        SDL_Log("Failed to create renderer: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Main loop
    bool running = true;
    SDL_Event event;
    Camera* cam = new Camera(0, 0, 0);

    int counter = 0;

    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }

            if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_UP:
                        cam->position.vz += 0.5;
                        break;
                    case SDLK_DOWN:
                        cam->position.vz -= 0.5;
                        break;
                    case SDLK_LEFT:
                        cam->position.vx -= 0.5;
                        break;
                    case SDLK_RIGHT:
                        cam->position.vx += 0.5;
                        break;
                }
            }
        }

        // timer
        auto timer_start = std::chrono::high_resolution_clock::now();

        // Clear the screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        //! Set the drawing color for the circle
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        //? with matrix multi
        std::vector<Vec4> cube = {
            {20, 20, -20, 1}, {20, -20, -20, 1}, {-20, -20, -20, 1}, {-20, 20, -20, 1},
            {20, 20, -40, 1}, {20, -20, -40, 1}, {-20, -20, -40, 1}, {-20, 20, -40, 1}
        };

        for (int i=1; i<100; i++) {
            for (auto &vertex : cube) {
                Vec4 relative_vec = vertex - cam->position;
                relative_vec.vx += 10*i;
                Vec4 projection_vec = ProjectionMat() * relative_vec;
                Vec4 perspective_vec = projection_vec * (1 / projection_vec.vw);

                float x_screen = WINDOW_WIDTH / 2 * perspective_vec.vx + WINDOW_WIDTH / 2;
                float y_screen = WINDOW_HEIGHT / 2 * perspective_vec.vy + WINDOW_HEIGHT / 2;

                DrawCircle(renderer, x_screen, y_screen, 5);
            }
        }


        // Present the renderer
        SDL_RenderPresent(renderer);

        auto timer_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> latency = timer_end - timer_start;

        if (counter == 0) {
            std::cout << 1/latency.count() << std::endl; 
        }
        counter = (counter+1)%20;
    }

    // Cleanup
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}