import cv2 as cv
import numpy as np

# 1. 读取图像与灰度图转换
def read_and_convert_to_gray(image_path):
    # 读取图像
    image = cv.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径！")
        return None
    # 转换为灰度图
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image, gray_image

# 2. 读取摄像头与视频显示
def read_camera():
    # 打开默认摄像头
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧！")
            break

        # 显示摄像头帧
        cv.imshow("Camera", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# 3. 归一化与显示
def normalize_and_display(image):
    # 归一化到 [0, 1]
    normalized_image = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    # 转换为 8 位图像以便显示
    normalized_image = (normalized_image * 255).astype(np.uint8)
    cv.imshow("Normalized Image", normalized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 4. 创建空白图像
def create_blank_image(width, height, color=(0, 0, 0)):
    # 创建一个指定大小和颜色的空白图像
    blank_image = np.full((height, width, 3), color, dtype=np.uint8)
    return blank_image

# 5. 提取 ROI 与分离、合并通道
def extract_roi_and_split_merge_channels(image, roi_start_point, roi_end_point):
    # 提取 ROI
    roi = image[roi_start_point[1]:roi_end_point[1], roi_start_point[0]:roi_end_point[0]]
    cv.imshow("ROI", roi)

    # 分离通道
    b, g, r = cv.split(image)
    cv.imshow("Blue Channel", b)
    cv.imshow("Green Channel", g)
    cv.imshow("Red Channel", r)

    # 合并通道
    merged_image = cv.merge([b, g, r])
    cv.imshow("Merged Image", merged_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

# 主函数
if __name__ == "__main__":
    # 读取图像并转换为灰度图
    image_path = "C:\\Users\\Lenovo\\Pictures\\Screenshots\\屏幕截图 2024-10-23 190902.png"  # 替换为你的图片路径
    image, gray_image = read_and_convert_to_gray(image_path)
    if image is not None:
        cv.imshow("Original Image", image)
        cv.imshow("Gray Image", gray_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # 读取摄像头
    # read_camera()

    # 归一化并显示
    # normalize_and_display(gray_image)

    # 创建空白图像
    blank_image = create_blank_image(300, 300, color=(255, 0, 0))  # 创建一个红色的空白图像
    cv.imshow("Blank Image", blank_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 提取 ROI 并分离/合并通道
    extract_roi_and_split_merge_channels(image, (50, 50), (200, 200))

import tensorflow as tf
import numpy as np

# 将numpy数组转换为Tensor
X_tensor = tf.constant(data_train_x, dtype=tf.float32)
Y_tensor = tf.constant(data_train.y, dtype=tf.float32)

# 初始化权重
W = tf.Variable(tf.zeros((data_train_x.shape[1], 1)), dtype=tf.float32)


# 定义损失函数（均方误差）
def compute_loss(W, X, Y):
    predictions = tf.matmul(X, W)
    errors = Y - predictions
    loss = tf.reduce_mean(tf.square(errors)) / 2
    return loss


# 定义训练步骤
optimizer = tf.optimizers.SGD(learning_rate=0.01)


def train_step(X, Y, W):
    with tf.GradientTape() as tape:
        loss = compute_loss(W, X, Y)
    gradients = tape.gradient(loss, [W])
    optimizer.apply_gradients(zip(gradients, [W]))


# 训练模型
iterations = 1000
for i in range(iterations):
    train_step(X_tensor, Y_tensor, W)
    if i % 100 == 0:
        current_loss = compute_loss(W, X_tensor, Y_tensor).numpy()
        print(f"Iteration {i}: Loss={current_loss}")

print("Optimal Weights:")
print(W.numpy())

# 定义损失函数
def fun(W, X, Y):
    predictions = np.dot(X, W)
    errors = Y - predictions
    loss = np.sum(errors ** 2) / 2
    return loss

# 风险最小化R(w)对w的偏导
def grad(W, X, Y):
    predictions = np.dot(X, W)
    errors = Y - predictions
    gradient = -np.dot(X.T, errors)
    return gradient

# 梯度下降优化函数
def gradient_descent(X, Y, learning_rate=0.01, iterations=1000):
    W = np.zeros((X.shape[1],1))  # 初始化权重向量，考虑了偏置项
    for i in range(iterations):
        gradient = grad(W, X, Y)
        W -= learning_rate * gradient  # 更新权重

        if i % 100 == 0:  # 打印每100次迭代后的损失值
            loss = fun(W, X, Y)
            print(f"Iteration {i}: Loss={loss}")
    print(W)
    return W

# 应用梯度下降法
W_optimal = gradient_descent(data_train_x, pd.DataFrame(data_train.y), learning_rate=0.01, iterations=1000)