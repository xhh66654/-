from predict import predict_image

# 输入要预测的图片路径
image_path = r"E:\pycharm\中草药智能识别\代码\test_picture\img_3.png"

result = predict_image(image_path)
print(f"预测结果是: {result}")
