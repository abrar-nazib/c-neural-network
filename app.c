#include <stdio.h>

// Global size of the perceptron
#define WIDTH 20
#define HEIGHT 20

typedef float Layer[HEIGHT][WIDTH]; // Perceptron layer

static Layer input;   // Input pixels
static Layer weights; // Weights

/**
 * Function for feeding input to the model and getting the guessed output
 */
float feed_forward(Layer input, Layer weights)
{ // here weights array might be called a trained model.
    float output = 0.0f;
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            output += input[y][x] * weights[y][x];
        }
    }
    return output;
}

/**
 * Function for limiting an integer variable in a certain range
 */
int limit_in_range_i(int n, int min, int max)
{
    if (n < min)
        n = min;
    if (n > max)
        n = max;
    return n;
}

/**
 * Function for creating rectangle
 */
void create_rect(Layer layer, int x, int y, int w, int h, float value)
{
    // setting top left corner of the rectangle
    int x0 = limit_in_range_i(x, 0, WIDTH - 1);
    int y0 = limit_in_range_i(y, 0, HEIGHT - 1);

    // setting width and height
    w = limit_in_range_i(w, 1, (WIDTH - 1) - x0);
    h = limit_in_range_i(h, 1, (HEIGHT - 1) - y0);
}

int main()
{

    // float output = feed_forward(input, weights);
    // printf("%f", output);

    return 0;
}