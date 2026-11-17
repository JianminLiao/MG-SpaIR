# Baboon: blur sigma 1, x2 downsampling, noise 5/255, 20% missing pixels. SIREN backbone.
IMAGE = "baboon.png"
BLUR_SIGMA = 1.0
SCALE = 2
NOISE_STD = 5 / 255
MISSING_PROB = 0.2

BACKBONE = "siren"
NUM_GRADES = 3
LAYERS_PER_GRADE = 1
HIDDEN_DIM = 256

EPOCHS = 9000
LR = 4e-3

THRESHOLD = 10 / 255
LAMBDA = 0.001 * THRESHOLD
