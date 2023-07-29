import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
import keras

# Load mô hình từ file mnist.h5
model = keras.models.load_model('mnist.h5')

class PaintApp:
    def __init__(self, root, width, height):
        self.root = root
        self.root.title("Handwritten Digit Recognition")  # Sửa tên ứng dụng

        # Tạo canvas để vẽ
        self.canvas = tk.Canvas(root, width=width, height=height, bg='black')
        self.canvas.pack()

        # Cấu hình bút vẽ mặc định
        self.pen_color = 'white'
        self.pen_size = 10

        # Thiết lập các nút điều khiển
        self.setup_buttons()

        # Liên kết sự kiện chuột kéo để vẽ trên canvas
        self.canvas.bind("<B1-Motion>", self.paint)

    def setup_buttons(self):
        # Nút dự đoán chữ số
        self.predict_button = ttk.Button(root, text="Nhận diện chữ số", command=self.predict_digit)
        self.predict_button.pack(pady=10)

        # Nút xóa canvas
        self.clear_button = ttk.Button(root, text="Xóa", command=self.clear_canvas)
        self.clear_button.pack(pady=5)

    def paint(self, event):
        # Vẽ một điểm tròn tại vị trí chuột di chuyển
        x1, y1 = (event.x - self.pen_size), (event.y - self.pen_size)
        x2, y2 = (event.x + self.pen_size), (event.y + self.pen_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_color, outline=self.pen_color)

    def predict_digit(self):
        # Lấy ảnh từ canvas và tiền xử lý để phù hợp với mô hình
        image = self.get_canvas_image()
        image = image.convert('L')
        image = image.resize((28, 28))
        image = np.array(image)
        image = image.reshape(1, 28, 28, 1)
        image = image.astype('float32')
        image /= 255

        # Dự đoán chữ số từ ảnh
        result = model.predict(image)
        digit = np.argmax(result[0])

        # Hiển thị kết quả dự đoán và làm nổi bật kết quả
        result_label.config(text="Kết quả dự đoán: " + str(digit), font=("Arial", 16, "bold"), fg="blue")

    def get_canvas_image(self):
        # Tạo ảnh từ nội dung của canvas
        image = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()), "black")
        draw = ImageDraw.Draw(image)

        # Tạo hình chữ nhật bao quanh canvas với màu đen
        draw.rectangle([(0, 0), (self.canvas.winfo_width(), self.canvas.winfo_height())], fill="black")

        # Lấy tất cả các item trên canvas và vẽ chúng trên ảnh
        items = self.canvas.find_all()
        for item in items:
            x1, y1, x2, y2 = self.canvas.coords(item)
            color = self.canvas.itemcget(item, "fill")
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=color)

        return image

    def clear_canvas(self):
        # Xóa tất cả các item trên canvas
        self.canvas.delete("all")

if __name__ == "__main__":
    width, height = 400, 400
    root = tk.Tk()

    # Tạo ứng dụng Paint
    app = PaintApp(root, width, height)

    # Hiển thị kết quả dự đoán và làm nổi bật kết quả
    result_label = tk.Label(root, text="Kết quả dự đoán: ", font=("Arial", 16, "bold"), fg="blue")
    result_label.pack(pady=10)

    # Bắt đầu vòng lặp chính của ứng dụng
    root.mainloop()
