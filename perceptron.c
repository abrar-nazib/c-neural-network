/**
 * Single layer perceptron based neural network with C
 * Author: Nazib Abrar
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>

#include <sys/stat.h>
#include <sys/types.h>

#define WIDTH 100
#define HEIGHT 100

#define BIAS 20.0

#define SAMPLE_SIZE 4000
#define TRAIN_PASSES 5000

#define PPM_SCALER 5
#define PPM_COLOR_INTENSITY 255
#define PPM_RANGE 10.0

#define DATA_FOLDER "data"
#define MODEL_FOLDER "ml_models"

#define TRAIN_SEED 63 // seed for srand() while training
#define CHECK_SEED 54 // seed for srand() while testing

typedef float Layer[HEIGHT][WIDTH];

/**
 * Function for limiting an integer variable
 */
int limit_in_range(int x, int low, int high)
{
    if (x < low)
        x = low;
    if (x > high)
        x = high;
    return x;
}

/**
 * Function that fills a layer with color in rectangular shape
 */
void layer_fill_rect(Layer layer, int x, int y, int w, int h, float value)
{
    assert(w > 0);
    assert(h > 0);
    int x0 = limit_in_range(x, 0, WIDTH - 1);
    int y0 = limit_in_range(y, 0, HEIGHT - 1);
    int x1 = limit_in_range(x0 + w - 1, 0, WIDTH - 1);
    int y1 = limit_in_range(y0 + h - 1, 0, HEIGHT - 1);
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            layer[y][x] = value;
        }
    }
}

/**
 * Function to create circle
 */
void layer_fill_circle(Layer layer, int cx, int cy, int r, float value)
{
    assert(r > 0);
    int x0 = limit_in_range(cx - r, 0, WIDTH - 1);
    int y0 = limit_in_range(cy - r, 0, HEIGHT - 1);
    int x1 = limit_in_range(cx + r, 0, WIDTH - 1);
    int y1 = limit_in_range(cy + r, 0, HEIGHT - 1);
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            int dx = x - cx;
            int dy = y - cy;
            if (dx * dx + dy * dy <= r * r)
            {
                layer[y][x] = value;
            }
        }
    }
}

/**
 * Save a layer as ppm file
 */
void layer_save_as_ppm(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "ERROR: could not open file %s: %m\n",
                file_path);
        exit(1);
    }

    fprintf(f, "P6\n%d %d 255\n", WIDTH * PPM_SCALER, HEIGHT * PPM_SCALER);

    for (int y = 0; y < HEIGHT * PPM_SCALER; ++y)
    {
        for (int x = 0; x < WIDTH * PPM_SCALER; ++x)
        {
            float s = (layer[y / PPM_SCALER][x / PPM_SCALER] + PPM_RANGE) / (2.0f * PPM_RANGE);
            // printf("%f\n", s * PPM_COLOR_INTENSITY);
            if (s * PPM_COLOR_INTENSITY < 140)
                s = 0;
            char pixel[3] = {
                // 0,
                // 0,
                // 0,
                // (char)floorf(PPM_COLOR_INTENSITY * (1.0f - s)),
                // (char)floorf(PPM_COLOR_INTENSITY * (1.0f - s)),
                (char)floorf(PPM_COLOR_INTENSITY * s),
                0,
                0,
            };

            fwrite(pixel, sizeof(pixel), 1, f);
        }
    }

    fclose(f);
}

/**
 * Save layer as binary file. Needs to be done
 */
void layer_save_as_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "ERROR: could not open file %s: %m", file_path);
        exit(1);
    }
    fwrite(layer, sizeof(Layer), 1, f);
    fclose(f);
}

/**
 * Loads a binary file.
 * Intended use case is to load pre trained model
 */
void layer_load_from_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "rb");
    size_t x = fread(layer, 1, sizeof(Layer) + 1, f);
    fclose(f);
}

/**
 * Function to feed forward the input layer
 * Returns the activation of output
 */
float feed_forward(Layer inputs, Layer weights)
{
    float output_activation = 0.0f;

    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            output_activation += inputs[y][x] * weights[y][x];
        }
    }

    return output_activation;
}

/**
 * Function to reward the model
 */
void add_inputs_from_weights(Layer inputs, Layer weights)
{
    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            weights[y][x] += inputs[y][x];
        }
    }
}

/**
 * Function to punish the model
 */
void sub_inputs_from_weights(Layer inputs, Layer weights)
{
    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            weights[y][x] -= inputs[y][x];
        }
    }
}

/**
 * Function to generate a random integer between given range
 */
int rand_range(int low, int high)
{
    assert(low < high);
    return rand() % (high - low) + low;
}

/**
 * Function to generate random sized rectangles at random places of the screen
 */
void layer_random_rect(Layer layer)
{
    layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
    int x = rand_range(0, WIDTH);
    int y = rand_range(0, HEIGHT);

    int w = WIDTH - x;
    if (w < 2)
        w = 2;
    w = rand_range(1, w);

    int h = HEIGHT - y;
    if (h < 2)
        h = 2;
    h = rand_range(1, h);

    layer_fill_rect(layer, x, y, w, h, 1.0f);
}

/**
 * Function to generate random sized circles at random places of the screen
 */
void layer_random_circle(Layer layer)
{
    layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
    int cx = rand_range(0, WIDTH);
    int cy = rand_range(0, HEIGHT);
    int r = INT_MAX;
    if (r > cx)
        r = cx;
    if (r > cy)
        r = cy;
    if (r > WIDTH - cx)
        r = WIDTH - cx;
    if (r > HEIGHT - cy)
        r = HEIGHT - cy;
    if (r < 2)
        r = 2;
    r = rand_range(1, r);
    layer_fill_circle(layer, cx, cy, r, 1.0f);
}

/**
 * Function to adjust the weights of the model.
 */
int adjust_weights(Layer inputs, Layer weights)
{
    static char file_path[256];
    static int count = 0;

    int adjusted = 0;

    for (int i = 0; i < SAMPLE_SIZE; ++i)
    {
        layer_random_rect(inputs);
        if (feed_forward(inputs, weights) > BIAS)
        {
            sub_inputs_from_weights(inputs, weights);
            // snprintf(file_path, sizeof(file_path), DATA_FOLDER "/weights-%03d.ppm", count++);
            // printf("[INFO] saving %s\n", file_path);
            // layer_save_as_ppm(weights, file_path);
            adjusted += 1;
        }

        layer_random_circle(inputs);
        if (feed_forward(inputs, weights) < BIAS)
        {
            add_inputs_from_weights(inputs, weights);
            // snprintf(file_path, sizeof(file_path), DATA_FOLDER "/weights-%03d.ppm", count++);
            // printf("[INFO] saving %s\n", file_path);
            // layer_save_as_ppm(weights, file_path);
            adjusted += 1;
        }
    }

    return adjusted;
}

/**
 * Function for testing the accuracy of the model.
 *  Returns the number of errors it made.
 * Tests the model of the 1/4th of its training data
 */
int test_model(Layer inputs, Layer weights, int div)
{
    static char file_path[256];
    int n_error = 0;

    for (int i = 0; i < SAMPLE_SIZE / div; ++i)
    {
        layer_random_rect(inputs);
        if (i % 50 == 0) // save a test rectangle
        {
            snprintf(file_path, sizeof(file_path), DATA_FOLDER "/training_data/rect-%d.ppm", i);
            // printf("%s is saved\n", file_path);
            layer_save_as_ppm(inputs, file_path);
            snprintf(file_path, sizeof(file_path), DATA_FOLDER "/bin/rect-%d.bin", i);
            // printf("%s is saved\n", file_path);
            layer_save_as_bin(inputs, file_path);
        }
        if (feed_forward(inputs, weights) > BIAS)
        {
            // neuron should not fire when rectangle is shown
            n_error += 1;
        }

        layer_random_circle(inputs);
        if (i % 50 == 0) // save a test circle
        {
            snprintf(file_path, sizeof(file_path), DATA_FOLDER "/training_data/circle-%d.ppm", i);
            // printf("%s is saved\n", file_path);
            layer_save_as_ppm(inputs, file_path);
            snprintf(file_path, sizeof(file_path), DATA_FOLDER "/bin/circle-%d.bin", i);
            // printf("%s is saved\n", file_path);
            layer_save_as_bin(inputs, file_path);
        }
        if (feed_forward(inputs, weights) < BIAS)
        {
            // neuron should not be inactive when circle is shown
            n_error += 1;
        }
    }

    return n_error;
}

void manualModelTest(Layer model, const char *file_path)
{
    float output;
    Layer inputs;
    layer_load_from_bin(inputs, file_path);
    output = feed_forward(inputs, model);
    if (output > BIAS)
    {
        printf("[GUESS] CIRCLE \t %f\n", output);
    }
    else
    {
        printf("[GUESS] RECTANGLE \t %f\n", output);
    }
}

/**
 * Train the model with on-the-fly generated images
 */
void train_model(Layer inputs, Layer weights)
{
    static char file_path[256];
    for (int i = 0; i < TRAIN_PASSES; ++i)
    {
        srand(TRAIN_SEED);
        int adj = adjust_weights(inputs, weights);
        printf("[TRAINING MODEL] Pass %d -- Error Adjusted %d Times\n", i, adj);
        if (i % 25 == 0)
        {
            snprintf(file_path, sizeof(file_path), MODEL_FOLDER "/visuals/model-w%d-h%d-r%d-s%d-%d.ppm", WIDTH, HEIGHT, TRAIN_SEED, SAMPLE_SIZE, i);
            layer_save_as_ppm(weights, file_path);
            printf("[SAVE STATE] Saving %s\n", file_path);
        }
        if (adj <= 0)
        {
            printf("Model Training Done!\n");
            break;
        }
    }
}

static Layer inputs;
static Layer weights;

// int main(void)
// {
//     char interruptor = 'a';

//     static char file_path[256]; // for saving the trained model

//     // check the accuracy of the untrained model
//     srand(CHECK_SEED); // set seed for for checking
//     int adj = test_model(inputs, weights, 1);
//     printf("Accuracy of untrained model is %f%%\n", (1 - adj / (SAMPLE_SIZE * 2.0)) * 100);

//     scanf("%c", &interruptor); // interruption for

//     // Save the untrained model as ppm
//     snprintf(file_path, sizeof(file_path), MODEL_FOLDER "/visuals/model-w%d-h%d-r%d-s%d-untrained.ppm", WIDTH, HEIGHT, TRAIN_SEED, SAMPLE_SIZE);
//     layer_save_as_ppm(weights, file_path);

//     // Train model
//     train_model(inputs, weights);

//     // test after training with training data
//     srand(TRAIN_SEED);
//     adj = test_model(inputs, weights, 1);
//     printf("Accuracy of trained model in training data is %f%%\n", (1 - adj / (SAMPLE_SIZE * 2.0)) * 100);

//     // test after training with new data
//     srand(CHECK_SEED);
//     adj = test_model(inputs, weights, 3);
//     printf("Accuracy of trained model in new data is %f%%\n", (1 - adj / (SAMPLE_SIZE * 2.0)) * 100);

//     // save the model
//     snprintf(file_path, sizeof(file_path), MODEL_FOLDER "/model-w%d-h%d-r%d-s%d.bin", WIDTH, HEIGHT, TRAIN_SEED, SAMPLE_SIZE);
//     layer_save_as_bin(weights, file_path);
//     snprintf(file_path, sizeof(file_path), MODEL_FOLDER "/model-w%d-h%d-r%d-s%d.ppm", WIDTH, HEIGHT, TRAIN_SEED, SAMPLE_SIZE);
//     layer_save_as_ppm(weights, file_path);
//     return 0;
// }

int main()
{
    char filename[100];
    char filename2[100];
    Layer model;
    layer_load_from_bin(model, "ml_models/model-w100-h100-r63-s4000.bin");
    printf("Loaded model model-w100-h100-r63-s4000.bin\nTo test the model, input filename\n");
    while (true)
    {
        printf("Enter Filename: ");
        scanf("%s", filename2);
        snprintf(filename, sizeof(filename), DATA_FOLDER "/bin/%s", filename2);
        manualModelTest(model, filename);
    }
}