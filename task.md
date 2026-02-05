1. Use the correct answer samples from the MNIST dataset
2. load a simple number image as test sample
3. For each pixel, calculate the score simulating its likelyhood of a number, so it should be {
    0: the average pixel value of this pixel in all "0" images, 
    ...
}
4. for the sample image add up the pixel scores
5. show the final prediction and confidence of the sample image, you should also show the sample image