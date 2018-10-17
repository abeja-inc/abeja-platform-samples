def preprocess_input(img):
    img *= (2.0 / 255.0)
    img -= 1
    return img