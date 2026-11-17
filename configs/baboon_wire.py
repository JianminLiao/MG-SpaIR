# Baboon: blur sigma 1, x2 downsampling, noise 5/255, 20% missing pixels. WIRE backbone.
IMAGE = "baboon.png"
BLUR_SIGMA = 1.0
SCALE = 2
NOISE_STD = 5 / 255
MISSING_PROB = 0.2

BACKBONE = "wire"
NUM_GRADES = 3
LAYERS_PER_GRADE = 1
HIDDEN_DIM = 256

EPOCHS = 4000
LR = [2e-3, 2e-3, 1e-3]
LR_DECAY = 0.2

THRESHOLD = 10 / 255
LAMBDA = 0.0011 * THRESHOLD
