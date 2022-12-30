#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

// Global size of the perceptron
#define WIDTH 80
#define HEIGHT 80

// #define PPM_SCALER 25   // scalar proportion to scale the image while showing
#define SAMPLE_SIZE 100 // samples to create or test

#define BIAS 10 // experimental bias value for decision making of the neuron

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
    for (int iy = y0; iy <= y1; iy++)
    {
        for (int ix = x0; ix <= x1; ix++)
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

    float min = FLT_MAX;
    float max = FLT_MIN;
    for (int iy = 0; iy < HEIGHT - 1; iy++)
    {
        for (int ix = 0; ix < WIDTH - 1; ix++)
        {
            if (layer[iy][ix] < min)
            {
                min = layer[iy][ix];
            }
            if (layer[iy][ix] > max)
            {
                max = layer[iy][ix];
            }
        }
    }

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
            float s = (layer[iy / PPM_SCALER][ix / PPM_SCALER] - min) / (max - min); // limits the value between 0-1 in pixel. Working as a scalar
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

void load_layer_bin(Layer layer, const char *file_path)
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
void layer_random_rect(Layer inputs)
{
    create_rect(inputs, 0, 0, WIDTH, HEIGHT, 0.0f); // filling the whole layer with white
    int x = rand_range(0, WIDTH);
    int y = rand_range(0, HEIGHT);

    // determining suitable width
    int w = WIDTH - x;
    if (w < 2)
        w = 2;
    w = rand_range(1, w);

    // determining suitable height
    int h = HEIGHT - y;
    if (h < 2)
        h = 2;
    h = rand_range(1, h);

    create_rect(inputs, x, y, w, h, 1.0f);
}

/**
 * Function for generating random circles for training
 */
void layer_random_circle(Layer inputs)
{
    create_rect(inputs, 0, 0, WIDTH, HEIGHT, 0.0f); // filling the whole layer with white
    int cx = rand_range(0, WIDTH);
    int cy = rand_range(0, HEIGHT);

    // applying condition to generate circles inside the window
    int r = __INT_MAX__;
    if (r > cx)
        r = cx;
    if (r > cy)
        r = cy;
    if (r > WIDTH - cx)
        r = WIDTH - cx;
    if (r > HEIGHT - cy)
        r = HEIGHT - cy;
    r = rand_range(1, r);
    create_circle(inputs, cx, cy, r, 1.0f);
}

/**
 * Function to add inputs to weights
 * Need to add to the weights if want the output neuron to be fired
 */
void excite_neuron(Layer inputs, Layer weights)
{
    for (int iy = 0; iy < HEIGHT; iy++)
    {
        for (int ix = 0; ix < WIDTH; ix++)
        {
            weights[iy][ix] += inputs[iy][ix];
        }
    }
}

/**
 * Function to subtract inputs from weights
 * If the output neuron need not to be fired, subtraction is needed
 */
void supress_neuron(Layer inputs, Layer weights)
{
    for (int iy = 0; iy < HEIGHT; iy++)
    {
        for (int ix = 0; ix < WIDTH; ix++)
        {
            weights[iy][ix] -= inputs[iy][ix];
        }
    }
}

static Layer inputs;  // Input pixels
static Layer weights; // Weights

int main()
{
    srand(69); // seeding random number generator with a constant. Not using time right now tho
    for (int i = 0; i < SAMPLE_SIZE; i++)
    {
        layer_random_rect(inputs);
        // if (feed_forward(inputs, weights) > BIAS)
        // { // neuron need not to be fired when sees rectangle
        // supress_neuron(inputs, weights);
        // }
        printf("%d\n", i);
        layer_random_circle(inputs);
        // if (feed_forward(inputs, weights) < BIAS)
        // { // neuron need to be fired when sees circle
        // excite_neuron(inputs, weights);
        // }
    }

    return 0;
}