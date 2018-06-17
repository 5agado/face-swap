class Face:
    def __init__(self, img=None, face_img=None, rect=None):
        """
        Utility class for a face
        :param img: image containing the face
        :param face_img: image bounded to target face (can be the same as img)
        :param rect: dlib face rectangle
        """
        self.img = img
        self.rect = rect
        self.face_img = face_img
        self.landmarks = None

    def get_face_center(self, absolute=True):
        """
        Return center coordinates of the face. Coordinates are rounded to closest int
        :param absolute: if True, center is absolute to whole image, otherwise is relative to face img
        :return: (x, y)
        """
        if self.rect:
            top, right, bottom, left = (self.rect.top(), self.rect.right(), self.rect.bottom(), self.rect.left())
            x = (right - left)//2
            y = (bottom - top)//2
            if absolute:
                x += left
                y += top
            return x, y

    def get_face_size(self):
        """
        Return size face as (width, height)
        :return: (w, h)
        """
        top, right, bottom, left = (self.rect.top(), self.rect.right(), self.rect.bottom(), self.rect.left())
        w = right - left
        h = bottom - top
        return w, h
