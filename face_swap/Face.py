class Face:
    def __init__(self, img, rect=None):
        """
        Utility class for a face
        :param img: image containing the face
        :param rect: face rectangle
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
            x, y = self.rect.get_center()
            if absolute:
                x += self.rect.left
                y += self.rect.top
            return x, y

    def get_face_img(self):
        """
        Return image bounded to target face (boundary is defined by rect attribute)
        :return:
        """
        top, right, bottom, left = self.rect.get_coords()
        face_img = self.img[top:bottom, left:right]
        return face_img

    def get_face_size(self):
        """
        Return size of face as (width, height)
        :return: (w, h)
        """
        w, h = self.rect.get_size()
        return w, h

    def expand_face_boundary(self, border_expand: tuple):
        face_size = self.get_face_size()
        img_size = self.img.shape
        # if float given, consider as expansion ratio and obtain equivalent int values
        if type(border_expand[0]) == float:
            border_expand = (int(border_expand[0] * face_size[0]),
                             int(border_expand[1] * face_size[1]))

        border_expand = (border_expand[0]//2, border_expand[1]//2)

        top, right, bottom, left = self.rect.get_coords()
        x, y = left, top
        w = right - left
        h = bottom - top
        new_top = max(0, y - border_expand[1])
        new_bottom = min(img_size[0], y + h + border_expand[1])
        new_left = max(0, x - border_expand[0])
        new_right = min(img_size[1], x + w + border_expand[0])

        self.rect = Face.Rectangle(top=new_top, right=new_right, left=new_left, bottom=new_bottom)

    class Rectangle:
        def __init__(self, top, right, bottom, left):
            """
            Utility class to hold information about face position/boundaries in an image
            :param top:
            :param right:
            :param bottom:
            :param left:
            """
            self.top = top
            self.right = right
            self.bottom = bottom
            self.left = left

        def get_coords(self):
            return self.top, self.right, self.bottom, self.left

        def get_center(self):
            x = (self.right - self.left)//2
            y = (self.bottom - self.top)//2
            return x, y

        def get_size(self):
            w = self.right - self.left
            h = self.bottom - self.top
            return w, h

