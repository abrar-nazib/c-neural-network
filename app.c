#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Global size of the perceptron
#define WIDTH 50
#define HEIGHT 50
#define PPM_SCALER 25  // scalar proportion to scale the image while showing
#define SAMPLE_SIZE 10 // samples to create or test

typedef float Layer[HEIGHT][WIDTH]; // Perceptron layer

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
    w = limit_in_range_i(w, 1, WIDTH);
    h = limit_in_range_i(h, 1, HEIGHT);

    int x1 = limit_in_range_i(x0 + w - 1, 0, WIDTH - 1);
    int y1 = limit_in_range_i(y0 + h - 1, 0, WIDTH - 1);

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
 * Function for creating circles
 */
void create_circle(Layer layer, int cx, int cy, int r, float value)
{
    r = limit_in_range_i(r, 1, WIDTH / 2);
    int x0 = limit_in_range_i(cx - r, 0, WIDTH - 1); // first point x coordinate of the circle
    int x1 = limit_in_range_i(cx + r, 0, WIDTH - 1); // first point x coordinate of the circle
    int y0 = limit_in_range_i(cy - r, 0, HEIGHT - 1);
    int y1 = limit_in_range_i(cy + r, 0, HEIGHT - 1);
    int dx, dy; // for measuring the distance from the center point
    for (int iy = y0; iy < y1; iy++)
    {
        for (int ix = x0; ix < x1; ix++)
        {
            dx = cx - ix;
            dy = cy - iy;
            if (dx * dx + dy * dy <= r * r)
            {
                layer[iy][ix] = value;
            }
        }
    }
}

/**
 * Function to save a layer as ppm file
 */
void save_layer_ppm(Layer layer, const char *filepath)
{
    FILE *f = fopen(filepath, "wb"); // open file to write in binary format

    if (f == NULL) // handling failure while opening file
    {
        fprintf(stderr, "File could not be opened %s", filepath);
        exit(1);
    }
    fprintf(f, "P6\n%d %d 255\n", WIDTH * PPM_SCALER, HEIGHT * PPM_SCALER); // Don't quite understand why this line need to be included. Most likely codec stuff

    // write pixels to the file
    for (int iy = 0; iy < HEIGHT * PPM_SCALER; iy++)
    {
        for (int ix = 0; ix < WIDTH * PPM_SCALER; ix++)
        {
            float s = layer[iy / PPM_SCALER][ix / PPM_SCALER];
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

/**
 * Function for saving the model
 */
void save_layer_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "Error in opening file %s", file_path);
        exit(1);
    }
    fwrite(layer, sizeof(Layer), 1, f);
    // layer -> print the elements of the layer.
    // Buffer size -> sizeof Layer variable
    // Elements to print -> 1
    // Which file to print -> f
    fclose(f);
}

void load_model(Layer layer, const char *file_path)
{
    return;
}

int rand_range(int low, int high)
{
    if (low > high)
    {
        exit(1);
    }
    return rand() % (high - low) + low;
}

/**
 * Function for generating random rectangles for traiining
 */
void layer_random_rect(Layer input)
{
    create_rect(input, 0, 0, WIDTH, HEIGHT, 0.0f); // filling the whole layer with white
    int x = rand_range(0, WIDTH);
    int y = rand_range(0, HEIGHT);
    int w = rand_range(1, WIDTH);
    int h = rand_range(1, HEIGHT);
    create_rect(input, x, y, w, h, 1.0f);
}

/**
 * Function for generating random circles for training
 */
void layer_random_circle(Layer input)
{
    create_rect(input, 0, 0, WIDTH, HEIGHT, 0.0f); // filling the whole layer with white
    int cx = rand_range(0, WIDTH);
    int cy = rand_range(0, HEIGHT);
    int r = rand_range(1, WIDTH);
    create_circle(input, cx, cy, r, 1.0f);
}

static Layer input;   // Input pixels
static Layer weights; // Weights

int main()
{
    char file_path[256];
    for (int i = 0; i < SAMPLE_SIZE; ++i)
    {
        // print in the buffer. Why not strcpy? most likely the format string thing
        printf("Generating %s\n", file_path);
        // layer_random_rect(input);
        layer_random_circle(input);
        snprintf(file_path, sizeof(file_path), "circle-%02d.bin", i); // snprintf for format string stuffs
        save_layer_ppm(input, file_path);
        snprintf(file_path, sizeof(file_path), "circle-%02d.ppm", i);
        save_layer_bin(input, file_path);
    }

    return 0;
}