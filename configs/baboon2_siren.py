# Baboon: blur sigma 0.5, x2 downsampling, noise 10/255, 10% missing pixels. SIREN backbone.
IMAGE = "baboon.png"
BLUR_SIGMA = 0.5
SCALE = 2
NOISE_STD = 10 / 255
MISSING_PROB = 0.1

BACKBONE = "siren"
NUM_GRADES = 3
LAYERS_PER_GRADE = 1
HIDDEN_DIM = 256

EPOCHS = 8000
LR = 4e-3
RESTORE_BEST = True

THRESHOLD = 10 / 255
LAMBDA = 0.00295 * THRESHOLD
