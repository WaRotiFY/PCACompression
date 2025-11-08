import numpy as np
from PIL import Image
from binzp import *

class WBImage:
    def __init__(self, image_path):
        self.img = Image.open(image_path)
        self.wb_matrix = np.array(self.img.convert('L'), dtype=np.float64)
        self.wb_img = self.img.convert('L')
    def gen_wb_matrix(self, img):
        img_gray = img.convert('L')
        gray_matrix = np.array(img_gray)
        return gray_matrix
    def show_wb_img(self):
        self.wb_img.show()
    def save_wb_img(self):
        self.wb_img.save("wb_img.jpg")

class ConImage(WBImage):
    def __init__(self, image_path, ind):
        super().__init__(image_path)
        self.cur_dim = self.img.size
        self.next_dim = (int(self.img.width * ind), int(self.img.height * ind))
        self.mean_matrix = self.make_mean_matrix()
        self.centred_matrix = self.wb_matrix - self.mean_matrix
        self.codek_matrix = self.make_codek_matrix(self.next_dim)
        self.reduced_wb_matrix = self.centred_matrix @ self.codek_matrix
        self.reduced_wb_img = Image.fromarray(self.reduced_wb_matrix, mode = 'L')

    def make_bin_rdc(self, file_name):
        r_flat = self.reduced_wb_matrix.flatten()
        c_flat = self.codek_matrix.flatten()
        m_flat = self.mean_matrix.flatten()
        com = np.concatenate([
            np.array(self.reduced_wb_matrix.shape, dtype=np.float32),
            r_flat.astype(np.float32),
            np.array(self.codek_matrix.shape, dtype=np.float32),
            c_flat.astype(np.float32),
            np.array(self.mean_matrix.shape, dtype=np.float32),
            m_flat.astype(np.float32)
        ])
        com.tofile(file_name)
        compress_binary_gzip(file_name, file_name)

    def load_bin_rdc(self, file_name):
        decompress_binary_gzip(file_name, file_name)
        all_data = np.fromfile(file_name, dtype=np.float32)

        idx = 0
        dim_r = all_data[:2].astype(np.int32)
        idx += 2

        size_r = dim_r[0] * dim_r[1]
        r_flat = all_data[idx:idx + size_r]
        reduced_wb_matrix = r_flat.reshape(dim_r)
        idx += size_r

        dim_c = all_data[idx:idx + 2].astype(np.int32)
        idx += 2

        size_c = dim_c[0] * dim_c[1]
        c_flat = all_data[idx:idx + size_c]
        codek_matrix = c_flat.reshape(dim_c)
        idx += size_c

        dim_m = all_data[idx:idx + 2].astype(np.int32)
        idx += 2

        size_m = dim_m[0] * dim_m[1]
        m_flat = all_data[idx:idx + size_m]
        mean_matrix = m_flat.reshape(dim_m)

        return reduced_wb_matrix, codek_matrix, mean_matrix
    def make_mean_matrix(self):
        return np.mean(self.wb_matrix, axis=0)
    def make_codek_matrix(self, dim):
        target_matrix = np.dot(self.centred_matrix.T, self.centred_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(target_matrix)
        sorted_ind = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_ind]
        Q, R = np.linalg.qr(eigenvectors)
        Q_reduced = Q[:, :dim[1]].real
        return Q_reduced
    def show_reduced_wb_img(self):
        self.reduced_wb_img.show()
    def save_reduced__wb_img(self):
        self.reduced_wb_img.save("reduced_wb_img.jpg")
    def reverse_codek_img(self, output_name, file_path=-1):
        if file_path != -1:
            data = self.load_bin_rdc(file_path)
            self.reduced_wb_matrix = data[0]
            self.codek_matrix = data[1]
            self.mean_matrix = data[2]
        matrix = np.clip(
            self.reduced_wb_matrix @ self.codek_matrix.T + self.mean_matrix,
            0, 255
        ).astype(np.uint8)
        img = Image.fromarray(matrix, mode = 'L')
        img.show()
        img.save(output_name)