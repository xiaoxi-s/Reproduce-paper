import numpy as np
import matplotlib.pyplot as plt
import itlm


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

# for SGD, randomly choose a batch of (X, y) pair
def random_choose(X_train, y_train, batch_size = 256):
    if (batch_size >= len(X_train)):
        return X_train, y_train
    indices = np.random.choice(len(X_train), batch_size)
    X_selected = np.resize(X_train[indices], new_shape = (batch_size, len(X_train[0])))
    y_selected = np.resize(y_train[indices], new_shape=(batch_size, 1))
    return X_selected, y_selected


# return a vector containing noise
def generate_noise(n, sigmoid=1):
    return np.resize([np.random.normal(0, sigmoid) for i in range(n)], new_shape=(n, 1))


# return a [good_n, 100] feature matrix that contains good_n samples
def generate_feature_matrix(n, alpha):
    good_n = int(n * (alpha))
    if good_n <= 0:
        return None

    cov = np.zeros((good_n, good_n))
    np.fill_diagonal(cov, 1)
    a = np.random.default_rng()
    X = a.multivariate_normal(mean=np.array(
        [0 for i in range(good_n)]), cov=cov, size=(100))

    return X.transpose()


# generate both good and bad samples
def generate_samples(n, alpha, optimum_coef, num_of_features=100):
    X_good = generate_feature_matrix(n, alpha)
    X_bad = generate_feature_matrix(n, 1 - alpha)

    good_n = int(n * (alpha))
    # generate label output accordingly
    y_good = None
    if good_n > 0:
        y_good = np.matmul(X_good, optimum_coef) + generate_noise(len(X_good))

    bad_n = n - good_n
    y_bad = None
    if bad_n > 0:
        y_bad = np.resize([np.random.normal(0, 1) for i in range(
            len(X_bad))], new_shape=(len(X_bad), 1)) + generate_noise(len(X_bad))

    return X_good, y_good, X_bad, y_bad


# generate both the model to be train and the ground-truth coef
def get_optimum_model(num_of_features=100):
    # get optimum coefficient
    optimum_coef = np.resize([np.random.uniform()
                              for i in range(100)], new_shape=(num_of_features, 1))
    optimum_coef = normalize(optimum_coef, axis=0)


    return optimum_coef


def select_subset(model, X, y, num):
    loss = np.square(model.predict(X) - y)
    indices = np.argsort(loss, axis=0)
    indices = indices[0:num, 0].tolist()

    X_selected = np.array([X[i] for i in indices])
    y_selected = np.array([y[i] for i in indices])

    return X_selected, y_selected


def update_model(model, X, y, fit_full = True, batch_size = 256, learning_rate = 0.1):
    
    if fit_full == True : # for ITLM - 1. Assume full update. So I use fit() method.
        model.fit(X, y)
        return model
    else: # for ITML - 2; manually compute gradient and update parameters
        X_batch, y_batch = random_choose(X, y, batch_size=batch_size)
        gradient = np.multiply(X_batch, (model.predict(X_batch) - y_batch))
        gradient = np.sum(gradient, axis=0)/len(X_batch)
        model.coef_ = model.coef_ - learning_rate * gradient
    return model


def get_models_results(num_of_train_samples, alpha, num_of_rounds):
    optimum_coef = get_optimum_model()
    X_good, y_good, X_bad, y_bad = generate_samples(
        num_of_train_samples, alpha, optimum_coef)

    # have mixed dataset
    X = np.concatenate((X_good, X_bad))
    y = np.concatenate((y_good, y_bad))

    model_itlm_2 = LinearRegression(fit_intercept=False) # init 
    model_itlm_2.fit(X, y)
    model_itlm_2 = itlm.itlm_2(X, y, alpha, model_itlm_2,
                               select_subset, update_model)

    # Two models to be tested
    model_itlm_1 = LinearRegression(fit_intercept=False) # init
    model_itlm_1.fit(X, y)
    model_itlm_1 = itlm.itlm_1(X, y, num_of_rounds, alpha, model_itlm_1,
                               select_subset, update_model)

    oracle_model = LinearRegression(fit_intercept=False)
    oracle_model.fit(X_good, y_good)

    # ITLM-1 trained model
    model_itlm_1_loss = np.sqrt((np.square(model_itlm_1.coef_ - optimum_coef)).sum()/len(optimum_coef))
    # ITLM-2 trained model
    model_itlm_2_loss = np.sqrt((np.square(model_itlm_2.coef_ - optimum_coef)).sum()/len(optimum_coef))
    # oracle model
    oracle_model_loss = np.sqrt((np.square(oracle_model.coef_ - optimum_coef)).sum()/len(optimum_coef))

    return np.array([model_itlm_1_loss, model_itlm_2_loss, oracle_model_loss])


def test_sample_size(times, sample_sizes = None):
    if sample_sizes is None:
        sample_sizes = [200, 400, 600, 1000, 2000]

    avg_loss = np.zeros((5, 3))

    names = ["ITML-1", "ITML-2", "Oracle"]
    colors = ["green", 'red', 'blue']

    # conduct multiple times
    for i in range(times):
        loss = np.zeros((5, 3))
        for j in range(len(sample_sizes)):
            # small noise
            # 100 is the hyper parameter in ITML
            temp = get_models_results(sample_sizes[j], 0.75, 100)
            print(temp)
            loss[j] = temp

        avg_loss += avg_loss + loss/times

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot()
    ax.set(xlabel='sample size', ylabel='loss',
           title='Performance with 25% error ({} times)'.format(times))

    avg_loss = avg_loss.transpose()
    print(avg_loss)

    for i in range(len(avg_loss)):
        plt.plot(sample_sizes, avg_loss[i], color = colors[i], label=names[i])

    plt.legend()
    plt.show()


def test_alpha(times, alphas = None):
    if alphas is None:
        alphas = [i/50 + 0.5 for i in range(20)]

    names = ["ITML-1", "ITML-2", "Oracle"]
    colors = ["green", 'red', 'blue']
    avg_loss = np.zeros((len(alphas), 3))

    # conduct multiple times
    for i in range(times):
        loss = np.zeros((len(alphas), 3))
        for j in range(len(alphas)):
            # small noise
            temp = get_models_results(1000, alphas[j], 100)
            loss[j] = temp

        avg_loss += avg_loss + loss/times

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot()
    ax.set(xlabel='fraction of correct samples', ylabel='loss',
           title='Performance with 1000 samples ({} times)'.format(times))

    avg_loss = avg_loss.transpose()
    print(avg_loss)
    for i in range(len(avg_loss)):
        plt.plot(alphas, avg_loss[i], color = colors[i], label=names[i])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # modify the first parameter to determine test times for each variable
    # modify the second parameter, which denotes different sample sizes to be tested
    # Default tested sizes are: [200, 400, 600, 1000, 2000]
    #test_sample_size(10, None)

    # modify the first parameter to determine test times for each variable
    # modify the second parameter, which denotes different alphas to be tested
    # Default tested alphas are: [i/50 + 0.5 for i in range(20)] (0.5, 0.52, 0.54,..., 0.9)
    test_alpha(10, None)
