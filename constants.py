"""
Shared constants for the biometric age detection system
"""

# Age class ranges for classification
# Format: (min_age, max_age) where max_age=None means "and above"
AGE_CLASSES = [
    (0, 9),
    (10, 17),
    (18, 25),
    (26, 35),
    (36, 45),
    (46, 60),
    (61, 74),
    (75, None)
]

# Image processing settings
IMAGE_SIZE = (224, 224)
