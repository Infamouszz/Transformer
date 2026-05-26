import math
from math import sqrt
import Visualisation

X_linear = [110.15,98.71,94.91,116.35,108.75,113.32,118.63,101.81,108.73,121.82,148.57,136.29,153.19,164.73,151.76,203,178.5,169.28,194.63,195.38,192.3,191.52,207.64,227.57,226.79,234.46,242.98,216.21,214.58,228.45,222.25,238.35,242.17,252.88,260.58,257.56,284.33,285.12,283.64,298.96,315.07,311.99,322.68,333.47,338.06,350.35,364.91,372.57,387.91,385.61,368.68,355.68,382.49,386.34,398.58,410.82,416.97,409.36,395.6,398.63,410.12,420.87,439.97,442.26,441.55,443.11,458.41,460.73,471.45,479.89,492.91,496.77,528.15,528.18,510.62,535.89,534.39,548.94,554.31,571.93,570.46,598.01,598,599.59,607.25,627.15,626.45,654.01,657.88,660.95,673.97,660.86,650.89,637.82,621.72,607.13,584.9,561.89,562.61,600.18,630.09,653.87,666.19,663.95,540.36,512.01,485.18,455.28,447.58,409.24,382.4,337.2,307.28,296.54,278.88,253.56,236.65,214.4,176.08,152.29,130.05,116.25,141.59,160.01,195.31,253.59]
Y_linear = [545.8,538.17,532.06,523.66,513.74,517.56,528.24,527.48,516.79,493.89,496.18,506.87,487.02,467.18,460.31,455.73,466.41,477.1,446.56,449.62,455.73,459.54,443.51,428.24,432.82,425.95,397.71,402.29,426.72,398.47,420.61,409.92,410.69,409.16,393.89,383.97,380.15,374.81,362.6,352.67,338.17,345.8,348.09,326.72,325.19,309.92,300,296.18,284.73,285.5,315.27,316.79,301.53,295.42,292.37,290.84,280.92,273.28,272.52,280.92,273.28,261.83,265.65,266.41,252.67,245.04,242.75,235.11,230.53,222.14,217.56,206.11,196.18,190.08,183.97,172.52,164.89,158.78,154.96,145.8,129.77,121.37,125.95,109.92,103.82,98.47,82.44,73.28,59.54,56.49,51.91,81.68,89.31,108.4,120.61,135.11,151.91,170.23,182.44,155.73,129.77,109.92,86.26,70.99,202.29,220.61,237.4,260.31,275.57,305.34,325.19,348.85,377.86,387.79,406.87,429.01,454.96,473.28,499.24,521.37,539.69,551.15,522.14,503.05,469.47,422.9]

X_quadratic = [76.92,92.97,96.77,94.43,115.03,115.73,117.2,133.96,139.27,140.71,162.81,160.41,174.13,185.54,195.39,220.56,228.1,256.36,292.93,304.41,314.36,341.08,369.43,416.74,410.7,438.23,453.53,455.11,475.04,528.57,507.25,501.15,509.52,513.27,510.34,534.85,551.63,537.23,568.61,564.06,580.95,576.43,612.47,598.78,586.44,609.29,610.89,613.31,625.58,629.44,650.09,644.82,663.8,635.42,629.35,647.02,660.07,663.21,659.46,670.21,671.76,461.35,447.48,418.36,394.61,374.72,367.91,404.67,364.18,353.35,360.21,328.95,332.06,328.99,298.39,288.41,306.89,273.22,244.12,278.65,258.83,231.98,218.24,235.9,187.03,197.79,187.94,208.58,209.35,199.25,219.9,207.61,224.51,216.23,201.71,183.42,188.08,152.85,161.28,185.05,158.34,157.61,148.57,150.91,102.67,118.65,142.47,149.35,182.88,137.85,124.97,131.15,99.79,89.84,115.92,104.34,109,123.52,127.39,83.07,75.47,101.49,89.31,68.71,90.15,83.31,65.79,69.65,51.36,49.91,41.53,30.1,36.29,24.88,34.1,31.07,7.44,18.16,5.99]
Y_quadratic = [238.17,240.46,246.56,258.02,269.47,285.5,300.76,316.79,329.77,351.91,370.23,397.71,408.4,425.19,446.56,462.6,490.84,498.47,530.53,526.72,523.66,534.35,519.85,541.22,521.37,521.37,518.32,503.82,490.84,487.02,464.89,461.07,470.99,490.08,457.25,443.51,455.73,421.37,412.21,402.29,384.73,367.18,340.46,320.61,348.85,371.76,351.91,320.61,310.69,299.24,297.71,277.1,311.45,336.64,324.43,302.29,286.26,264.89,248.09,233.59,229.77,473.28,500,519.08,532.06,535.11,517.56,499.24,493.13,524.43,531.3,509.16,494.66,496.95,503.82,512.21,480.15,486.26,498.47,467.94,451.15,475.57,468.7,449.62,434.35,419.85,395.42,396.95,395.42,436.64,436.64,449.62,429.01,396.95,393.89,377.86,360.31,374.81,371.76,351.15,340.46,329.77,296.18,283.97,303.05,322.14,292.37,291.6,319.08,300.76,269.47,253.44,255.73,259.54,237.4,266.41,246.56,249.62,235.11,230.53,218.32,211.45,199.24,186.26,178.63,166.41,150.38,141.98,126.72,103.82,96.95,86.26,68.7,53.44,41.98,34.35,14.5,10.69,-4.58]

def z_score(val):
    z_scored_val = []

    N = len(val)
    sum_val = 0.0
    sum_val_squared = 0.0
    for i in range(N):
        sum_val += val[i]
        sum_val_squared += (val[i] ** 2)
    mean_val = sum_val / float(N)
    mean_val_squared = sum_val_squared / float(N)

    variance = mean_val_squared - (mean_val ** 2)
    standard_deviation = math.sqrt(variance)

    for i in range(N):
        z_scored_val.append((val[i] - mean_val) / standard_deviation)
    print("Z_SCORE_X = ", z_scored_val)
    return z_scored_val


def update_parameters(X, Y, w1, w2, w3, w4, b, alpha_w4,alpha_w3, alpha_w2, alpha_w1, lasso_lambda):
    gradient_sum_w4 = 0
    gradient_sum_w3= 0.0
    gradient_sum_w2 = 0.0
    gradient_sum_w1 = 0.0
    gradient_sum_b = 0.0
    N = len(X)

    for i in range(N):
        error_function = Y[i] - (w4 * math.exp(X[i]) + w3 * math.log(abs(X[i]) + 1e-8) + w2 * (X[i] ** 2) + w1 * X[i] + b)

        gradient_sum_w4 += -2 * math.exp(X[i]) * error_function
        gradient_sum_w3 += -2 * math.log(abs(X[i]) + 1e-8) * error_function
        gradient_sum_w2 += -2 * (X[i] ** 2) * error_function
        gradient_sum_w1 += -2 * X[i] * error_function
        gradient_sum_b += -2 * error_function

    w4 = w4 - ((1/float(N)) * gradient_sum_w4 + apply_lasso(w4, lasso_lambda)) * alpha_w4
    w3 = w3 - ((1 / float(N)) * gradient_sum_w3 + apply_lasso(w3, lasso_lambda)) * alpha_w3
    w2 = w2 - ((1 / float(N)) * gradient_sum_w2 + apply_lasso(w2, lasso_lambda)) * alpha_w2
    w1 = w1 - ((1 / float(N)) * gradient_sum_w1 + apply_lasso(w1, lasso_lambda)) * alpha_w1
    b = b - (1 / float(N)) * gradient_sum_b * alpha_w2

    return w4, w3, w2, w1, b

def apply_lasso(weight, lasso_lambda):
    sign = 0
    if weight > 0:
        sign = 1
    if weight < 0:
        sign = -1

    return lasso_lambda * sign


def train(X, Y, w1, w2, w3,w4, b, epoch, alpha_w4, alpha_w3, alpha_w2, alpha_w1, L1_Lambda):

    for i in range(epoch):
        w4, w3, w2, w1, b = update_parameters(X, Y, w1, w2, w3, w4, b, alpha_w4,alpha_w3, alpha_w2, alpha_w1, L1_Lambda)
        if 0.00001 <= abs(w4) <= 0.01:
            w4 = 0.0

        if 0.00001 <= abs(w3) <= 0.01:
            w3 = 0.0

        if i % 400 == 0 or i == epoch-1:
            avg_root_mse = sqrt(avg_loss(X, Y, w4, w3, w2, w1,b))
            real_root_loss = real_root_mse(avg_root_mse, Y)
            print("Epoch: ", i, "Avg Root Loss: ", avg_root_mse, "Real Root Loss: ", real_root_loss)
            print("W4: ",w4, "W3: ",w3, "W2: ",w2, "W1: ",w1, "B: ", b)

    weight_sum = abs(w4) + abs(w3) + abs(w2) + abs(w1)

    w1 = weight_power_denier(w1, weight_sum, 5)
    w2 = weight_power_denier(w2, weight_sum, 5)
    w3 = weight_power_denier(w3, weight_sum, 5)
    w4 = weight_power_denier(w4, weight_sum, 5)

    print("Finished Training", "W4: ", w4, "W3: ", w3,"W2: ", w2, "W1: ",w1, "B: ", b)
    return w4, w3, w2, w1, b, real_root_loss

def weight_power_denier(weight, weight_sum, power_percentage):
    weight_power_percentage = 0.0
    if weight != 0:
        weight_power_percentage = (abs(weight) / weight_sum) * 100
    if abs(weight_power_percentage) < power_percentage:
        return 0.0
    else:
        return weight

def avg_loss(X, Y, w4, w3, w2, w1, b):
    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (Y[i] - (w4 * math.exp(X[i]) + w3 * math.log(abs(X[i])) + w2 * (X[i] ** 2) + w1 * X[i] + b))**2
    avg_loss = total_error/float(N)
    return avg_loss

def real_root_mse(root_mse, Y):
    max_y, min_y = max(Y), min(Y)
    normalized_root_mse = (root_mse/(max_y-min_y)) * 100
    return normalized_root_mse

def predict(X_new, final_w4, final_w3, final_w2, final_w1, final_b):
    return (final_w4 * math.exp(X_new))+(final_w3 * math.log(abs(X_new) + 1e-8))+(final_w2*(X_new**2))+(final_w1*X_new)+final_b

z_score_X = z_score(X_quadratic)
z_score_Y = z_score(Y_quadratic)
w4, w3, w2, w1, b, avg_loss_final = train(z_score_X, Y_quadratic, 0.001, 0.001, 0.001, 0.001, 0.001, 30000, 0.001, 0.001,0.01, 0.1,1)
X_new = 0.5
Y_new = predict(X_new, w4, w3, w2, w1, b)
normalized_loss = real_root_mse(sqrt(avg_loss(z_score_X, Y_quadratic, w4, w3, w2, w1, b)), Y_quadratic)
print("Normalized loss: ",normalized_loss)
print("Predicted Value: ", Y_new)

Visualisation.plotar(z_score_X, Y_quadratic, w4, w3, w2, w1, b)


