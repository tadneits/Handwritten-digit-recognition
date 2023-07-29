import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import csv

# Hàm để lưu nhật ký huấn luyện vào file CSV
def save_log(history, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Lặp qua từng epoch trong lịch sử huấn luyện (history) và ghi thông tin vào file CSV
        for epoch in range(1, len(history.history['accuracy']) + 1):
            row = {
                'epoch': epoch,  # Số epoch hiện tại
                'loss': history.history['loss'][epoch - 1],  # Giá trị loss của epoch hiện tại
                'accuracy': history.history['accuracy'][epoch - 1],  # Độ chính xác của epoch hiện tại trên tập huấn luyện
                'val_loss': history.history['val_loss'][epoch - 1],  # Giá trị loss của epoch hiện tại trên tập kiểm tra
                'val_accuracy': history.history['val_accuracy'][epoch - 1]  # Độ chính xác của epoch hiện tại trên tập kiểm tra
            }
            writer.writerow(row)

# Load dữ liệu MNIST từ keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Tiền xử lý dữ liệu
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

batch_size = 128
num_classes = 10
epochs = 1

# Xây dựng mô hình neural network
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=0.001)

# Biên dịch mô hình
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

# Huấn luyện mô hình
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
print("The model has successfully trained")

# Đánh giá mô hình trên tập kiểm tra
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Lưu mô hình vào file mnist.h5
model.save('mnist.h5')
print("Saving the model as mnist.h5")

# Lưu nhật ký huấn luyện vào file CSV
save_log(hist, 'training_log.csv')
print("Training log has been saved to training_log.csv")
