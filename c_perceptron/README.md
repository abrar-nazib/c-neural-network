# Perceptron Implementation in C Programming Language

So far I just understood the working procedure of a perceptron and ways of training it. `perceptron.c` has the code\

Make sure `data/training_data` and `data/bin` folder exists in the root folder (where perceptron.c is)

```
gcc perceptron.c -lm -o perceptron ; ./perceptron
```

Execute above command for getting started.

- It will train the model with randomly generated 8000 images. (4000 rectangles and 4000 circles)
- Then will test 2000 randomly generated new images to test the accuracy. It's roughly 86%.
