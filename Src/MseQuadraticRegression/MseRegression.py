import numpy as np

X = [14.359, 17.9487, 12.8205, 21.5385, 18.9744, 25.641, 26.1538, 32.8205, 33.0769, 42.0513, 42.8205, 50.7692, 51.7949, 55.3846, 60.2564, 64.359, 70.5128, 78.2051, 77.4359, 84.359, 83.5897, 89.7436, 83.3333, 94.1026, 88.4615, 88.4615, 91.5385, 82.5641, 81.0256, 76.1538, 84.1026, 93.0769, 70.0, 62.8205, 49.4872, 35.8974, 40.5128, 34.359, 32.0513]
Y = [5.0096, 4.2404, 19.625, 15.3942, 30.7788, 30.0096, 41.5481, 44.2404, 58.0865, 56.5481, 67.3173, 66.1635, 60.3942, 73.4712, 65.3942, 76.1635, 64.2404, 65.7788, 56.9327, 53.4712, 44.2404, 43.0865, 32.3173, 27.3173, 21.5481, 32.3173, 15.0096, 15.7788, 29.2404, 41.5481, 8.4712, 7.3173, 52.7019, 57.7019, 44.625, 42.7019, 24.625, 27.7019, 10.3942]

def normalize(X, Y):
    normalized_X = []
    normalized_Y = []
    for i in range(len(X)):
        temp_x = (X[i] - min(X)) / (max(X) - min(X))
        temp_y = (Y[i] - min(Y)) / (max(Y) - min(Y))
        normalized_X.append(temp_x)
        normalized_Y.append(temp_y)
    print("Normalized X: ", normalized_X)
    print("Normalized Y: ", normalized_Y)

    return normalized_X, normalized_Y


def update_parameters(X, Y, w1, w2, b, alpha):
    gradient_sum_w1 = 0.0
    gradient_sum_w2 = 0.0
    gradient_sum_b = 0.0
    N = len(X)

    for i in range(N):

        gradient_sum_w1 += -2*(X[i]**2)*(Y[i]-(w1*(X[i]**2)+w2*X[i]+b))
        gradient_sum_w2 += -2*X[i]*(Y[i]-(w1*(X[i]**2)+w2*X[i]+b))
        gradient_sum_b += -2*(Y[i]-(w1*(X[i]**2)+w2*X[i]+b))

    w1 = w1 - (1/float(N))*gradient_sum_w1*alpha
    w2 = w2 - (1/float(N))*gradient_sum_w2*alpha
    b = b - (1/float(N))*gradient_sum_b*alpha

    return w1, w2, b

def train(X, Y, w1, w2, b, epoch, alpha):
    for i in range(epoch):
        w1, w2, b = update_parameters(X, Y, w1, w2, b, alpha)

        if i % 400 == 0:
            print("Epoch: ", i, "Loss: ", avg_loss(X, Y, w1, w2, b))
    print("Finished Training", "W1: ", w1, "W2: ", w2, "B: ", b)
    return w1, w2, b

def avg_loss(X, Y, w1, w2, b):
    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (Y[i] - (w1*(X[i]**2)+w2*X[i]+b))**2
    return total_error/float(N)

def predict(X_new, final_w1, final_w2, final_b):
    return (final_w1*(X_new**2))+(final_w2*X_new)+final_b

norm_X, norm_Y = normalize(X, Y)
w1, w2, w3 = train(norm_X, norm_Y, 0.0, 0.0, 0.0, 30000, 0.01)
X_new = 0.9
Y_new = predict(X_new, w1, w2, w3)
print("Prediction: ",Y_new)

