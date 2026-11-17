# Bee: blur sigma 0.5, x2 downsampling, noise 5/255, 20% missing pixels. SIREN backbone.
IMAGE = "bee.png"
BLUR_SIGMA = 0.5
SCALE = 2
NOISE_STD = 5 / 255
MISSING_PROB = 0.2

BACKBONE = "siren"
NUM_GRADES = 3
LAYERS_PER_GRADE = 1
HIDDEN_DIM = 512

EPOCHS = 15000
LR = 2e-3

THRESHOLD = 10 / 255
LAMBDA = 0.002 * THRESHOLD
