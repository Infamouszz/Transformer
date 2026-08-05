import MseRegression
import math

def tuning(alpha_range, lasso_range, X, Y):
    total_alpha_possibilites = math.log((alpha_range / 0.001),10) + 1
    total_lasso_possibilites = math.log((lasso_range / 0.001),10) + 1

    winner_model = [{"winner_alpha" : 0.0},{"winner_lasso" : 0.0},{"avg_loss_winner" : 0.0}]
    candidate_alpha = 0.001
    candidate_lasso = 0.001
    avg_loss_winner = math.inf
    worst_loss = 0.0

    for i in range(round(total_alpha_possibilites)):
        candidate_lasso = 0.001
        for j in range(round(total_lasso_possibilites)):
            w4, w3, w2, w1, b, avg_loss = MseRegression.train(X,Y, 0.001, 0.001, 0.001, 0.001, 0.001, 10000, 0.01, 0.01, candidate_alpha, candidate_alpha, candidate_lasso)
            if avg_loss < avg_loss_winner:
                avg_loss_winner = avg_loss
                winner_model = [{"winner_alpha" : candidate_alpha, "winner_lasso" : candidate_lasso}, {"avg_loss_winner" : avg_loss_winner}]
            candidate_lasso *= 10
            if avg_loss > worst_loss:
                worst_loss = avg_loss
        candidate_alpha *= 10
    return winner_model, worst_loss

X,Y = MseRegression.z_score_X, MseRegression.Y_quadratic
best_model, worst_loss = tuning(0.1, 100, X, Y)
print("Melhor modelo: ", best_model)
print("Pior modelo: ", worst_loss)
