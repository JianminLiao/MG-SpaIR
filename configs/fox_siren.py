# Fox: blur sigma 1, x2 downsampling, noise 5/255, 20% missing pixels. SIREN backbone.
IMAGE = "fox.jpg"
BLUR_SIGMA = 1.0
SCALE = 2
NOISE_STD = 5 / 255
MISSING_PROB = 0.2

BACKBONE = "siren"
NUM_GRADES = 3
LAYERS_PER_GRADE = 1
HIDDEN_DIM = 512

EPOCHS = 7000
LR = 4e-3

THRESHOLD = 1 / 255
LAMBDA = 0.01 * THRESHOLD
