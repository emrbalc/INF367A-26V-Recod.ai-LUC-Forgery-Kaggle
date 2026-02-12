import numpy as np
from scipy import ndimage
from matplotlib import pyplot

class PixelMapUtil:
    def __init__(
            self,
            gaussian_sigma: float = 0.5,
    ):
        self.gaussian_sigma = gaussian_sigma
    
    def get_gaussian_sigma(self) -> float:
        return self.gaussian_sigma
    
    def set_gaussian_sigma(self, sigma: float) -> None:
        self.gaussian_sigma = sigma

    def _get_edges(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        sobel_x = ndimage.sobel(img, 0)
        sobel_y = ndimage.sobel(img, 1)
        mag = np.hypot(sobel_x, sobel_y)
        return mag
    
    def get_edges(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return self._get_edges(img)
    
    def _gaussian_blur(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        gaussian = ndimage.gaussian_filter(img, self.gaussian_sigma)
        return gaussian
    
    def gaussian_blur(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return self._gaussian_blur(img)
    
    def opening(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_opening(img, structure=np.ones((3,3)))

    def closing(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_closing(img, structure=np.ones((7,7)))

    def fill_components(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_fill_holes(img) # type: ignore
    
    def post_process_img(self, img: np.typing.NDArray) -> np.typing.NDArray:
    
        
        #Get the edges using sobel gradient
        img = img.copy()

        edges = self.gaussian_blur(img)

        edges = self.get_edges(img.copy())
        edges = self.gaussian_blur(edges)
        edges = edges > 0.5
        edges = self.fill_components(edges)
        img += edges - 1 

        img = 1-img
        
        img = self.closing(img)

        img = self.opening(img)

        img = 1-img

        
        
        return img
