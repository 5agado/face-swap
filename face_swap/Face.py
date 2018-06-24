import dlib


class Face:
    # TODO option of major refactoring around having custom rect instead of dlib one
    def __init__(self, img=None, rect=None):
        """
        Utility class for a face
        :param img: image containing the face
        :param rect: dlib face rectangle
        """
        self.img = img
        self.rect = rect
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

    def get_face_img(self):
        """
        Return image bounded to target face (boundary is defined by rect attribute)
        :return:
        """
        top, right, bottom, left = (self.rect.top(), self.rect.right(), self.rect.bottom(), self.rect.left())
        face_img = self.img[top:bottom, left:right]
        return face_img

    def get_face_size(self):
        """
        Return size of face as (width, height)
        :return: (w, h)
        """
        top, right, bottom, left = (self.rect.top(), self.rect.right(), self.rect.bottom(), self.rect.left())
        w = right - left
        h = bottom - top
        return w, h

    def expand_face_boundary(self, border_expand: tuple):
        face_size = self.get_face_size()
        img_size = self.img.shape
        # if float given, consider as expansion ratio and obtain equivalent int values
        if type(border_expand[0]) == float:
            border_expand = (int(border_expand[0] * face_size[0]),
                             int(border_expand[1] * face_size[1]))

        border_expand = (border_expand[0]//2, border_expand[1]//2)

        top, right, bottom, left = (self.rect.top(), self.rect.right(), self.rect.bottom(), self.rect.left())
        x, y = left, top
        w = right - left
        h = bottom - top
        new_top = max(0, y - border_expand[1])
        new_bottom = min(img_size[0], y + h + border_expand[1])
        new_left = max(0, x - border_expand[0])
        new_right = min(img_size[1], x + w + border_expand[0])

        self.rect = dlib.rectangle(left=new_left, top=new_top,
                                   right=new_right, bottom=new_bottom)

