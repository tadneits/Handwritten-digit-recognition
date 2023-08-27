# Nhận dạng chữ số từ dữ liệu MNIST

Đây là một chương trình sử dụng thư viện Keras để xây dựng và huấn luyện mô hình nhận dạng chữ số từ bộ dữ liệu MNIST. Mô hình được huấn luyện với mạng neural tích chập (CNN) và sau đó được lưu lại để có thể sử dụng lại sau này. Hàm `save_log` được sử dụng để lưu nhật ký huấn luyện vào một file CSV để có thể theo dõi quá trình huấn luyện và đánh giá mô hình.

## Các bước chính trong chương trình:

1. **Tiền xử lý dữ liệu**: Dữ liệu MNIST gồm các ảnh kích thước 28x28 pixel. Đầu tiên, dữ liệu được tiền xử lý bằng cách reshape để phù hợp với đầu vào của mạng CNN và chuẩn hóa giá trị pixel từ [0, 255] về [0, 1].

2. **Xây dựng mô hình**: Mô hình neural network được xây dựng bao gồm các lớp Conv2D (tích chập 2D), MaxPooling2D (tối đa hóa), Flatten (để chuyển từ tensor sang vector), và các lớp Dense (kết nối đầy đủ) với các hàm kích hoạt relu và softmax.

3. **Biên dịch mô hình**: Trước khi huấn luyện, mô hình được biên dịch với hàm mất mát là categorical_crossentropy (sử dụng với các bài toán phân loại nhiều lớp), tối ưu hóa Adam với learning rate là 0.001 và độ đo là accuracy.

4. **Huấn luyện mô hình**: Sử dụng dữ liệu huấn luyện từ MNIST, mô hình được huấn luyện trong 100 epochs (vòng lặp huấn luyện) với kích thước batch là 128. Dữ liệu kiểm tra được sử dụng để đánh giá mô hình sau mỗi epoch.

5. **Lưu mô hình và nhật ký**: Sau khi huấn luyện, mô hình được lưu vào file 'mnist.h5' để có thể sử dụng lại sau này. Nhật ký huấn luyện, bao gồm thông tin về số epoch, loss và accuracy trên tập huấn luyện và tập kiểm tra, được lưu vào file 'training_log.csv'.

## Yêu cầu hệ thống:

- Python 3.x
- Thư viện Keras, numpy, csv, tensorflow, PIL (có thể cài đặt thông qua pip)

## Cách sử dụng:

1. Đảm bảo đã cài đặt các thư viện cần thiết.
2. Chạy chương trình và nó sẽ tự động tải dữ liệu MNIST, tiền xử lý và huấn luyện mô hình.
3. Sau khi huấn luyện xong, mô hình sẽ được lưu vào file 'mnist.h5'.
4. Nhật ký huấn luyện sẽ được lưu vào file 'training_log.csv', chứa thông tin về loss và accuracy của từng epoch trong quá trình huấn luyện.

Chương trình này sẽ tạo ra một mô hình có thể nhận dạng chữ số từ ảnh kích thước 28x28 pixel và có khả năng phân loại thành 10 lớp từ 0 đến 9.
