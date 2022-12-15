#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

    int x1 = x0 + w;
    int y1 = y0 + h;

    // Fillup the space using the given value
    for (int ix = x0; ix <= x1; ix++)
    {
        for (int iy = y0; iy < y1; iy++)
        {
            layer[iy][ix] = value;
        }
    }
}

/**
 * Function to save a layer as ppm file
 */
void save_layer(Layer layer, const char *filepath)
{
    FILE *f = fopen(filepath, "wb"); // open file to write in binary format

    if (f == NULL) // handling failure while opening file
    {
        fprintf(stderr, "File could not be opened %s", filepath);
        exit(1);
    }
    fprintf(f, "P6\n%d %d 255\n", WIDTH, HEIGHT); // Don't quite understand why this line need to be included. Most likely codec stuff

    // write pixels to the file
    for (int iy = 0; iy < HEIGHT; iy++)
    {
        for (int ix = 0; ix < WIDTH; ix++)
        {
            float s = layer[iy][ix];
            char pixel[3] = {
                (char)floorf(s * 255), // why not only floor? floor is for double. floorf() is float specific
                0,
                0};
            fwrite(pixel, sizeof(pixel), 1, f);
            // write pixel as 3 byte size of 1 buffer in f[ile]
        }
    }
    fclose(f);
}

int main()
{

    create_rect(input, 0, 0, WIDTH / 2, HEIGHT / 2, 1.0f);
    save_layer(input, "output.ppm");
    return 0;
}