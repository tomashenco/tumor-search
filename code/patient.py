from settings import patient_contours, patient_pngs


class Patient(object):
    def __init__(self, path, contours, images):
        self.main_path = path
        self.contours = contours
        self.images = images
