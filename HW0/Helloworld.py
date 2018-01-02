import cv2


class Imageprocessing:

    def __init__(self, path):
        self.img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def imgDisplay(self):
        winName = 'Image Display'
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(winName, self.img)
        cv2.waitKey(6000)
        cv2.destroyWindow(winName)

    def imgOperations(self, operator, val):

        if operator == '+':
            winName = 'Image Add Operation'
            img1 = self.img + val

        elif operator == '-':
            winName = 'Image Subtract Operation'
            img1 = self.img - val

        elif operator == '*':
            winName = 'Image Multiply Operation'
            img1 = self.img * val

        elif operator == '/':
            winName = 'Image Division Operation'
            img1 = self.img // val

        print(self.img)
        print('After operation')
        print(img1)
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(winName, img1)
        cv2.waitKey(6000)
        cv2.destroyWindow(winName)

    def imgResize(self, val):
        winName = 'Image Resize Operation'
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        res = cv2.resize(
            self.img, None, fx=val, fy=val, interpolation=cv2.INTER_LINEAR)
        print(self.img)
        print(self.img.shape)
        cv2.imshow(winName, res)
        print(res.shape)
        cv2.waitKey(6000)
        cv2.destroyWindow(winName)


if __name__ == "__main__":

    disp = Imageprocessing("Dog.jpg")
    disp.imgDisplay()
    disp.imgOperations('+', 2)
    disp.imgOperations('-', 4)
    disp.imgOperations('*', 4)
    disp.imgOperations('/', 4)
    disp.imgResize(0.5)