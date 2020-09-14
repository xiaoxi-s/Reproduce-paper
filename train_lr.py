import numpy as np
import matplotlib.pyplot as plt
import itlm


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale


model_num = 4
names = ["baseline", "ITML-1", "ITML-2", "Oracle"]
colors = ['skyblue', "green", 'red', 'blue']
draw_figure = [False, True, True, True]

noise_std_dev = 1
cov_var = 9

file_nums = 0

feature_vector = '3 + np.random.rand()'

# for SGD, randomly choose a batch of (X, y) pairs
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


# return a [good_n, 100] feature matrix that contains samples
def generate_feature_matrix(n, alpha):
    good_n = int(n * (alpha))
    if good_n <= 0:
        return None

    cov = np.zeros((100, 100))
    np.fill_diagonal(cov, cov_var)
    a = np.random.default_rng()
    X = a.multivariate_normal(mean=np.array(
        [2 + np.random.rand() for i in range(100)]), cov=cov, size=(good_n))

    return X


# generate both good and bad samples
def generate_samples(n, alpha, optimum_coef, num_of_features=100, noise_std_deviation=1):
    X_good = generate_feature_matrix(n, alpha)
    X_bad = generate_feature_matrix(n, 1 - alpha)

    good_n = int(n * (alpha))
    # generate label output accordingly
    y_good = None
    if good_n > 0:
        y_good = np.matmul(X_good, optimum_coef) + generate_noise(len(X_good), noise_std_deviation)

    bad_n = n - good_n
    y_bad = None
    if bad_n > 0:
        y_bad = np.resize([np.random.normal(0, 1) for i in range(
            len(X_bad))], new_shape=(len(X_bad), 1)) + generate_noise(len(X_bad), noise_std_deviation)

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


def update_model(model, X, y, M=80, fit_full = True, batch_size = 256, learning_rate = 0.0001):
    if fit_full == True : # for ITLM - 1. Use SGD for M times. 
        model.fit(X, y)
        '''
        for i in range(M):
            X_batch, y_batch = random_choose(X, y, batch_size=batch_size)
            gradient = np.multiply(X_batch, (model.predict(X_batch) - y_batch))
            gradient = np.resize(np.sum(gradient, axis=0)/len(X_batch), new_shape=(1, 100))
            model.coef_ = model.coef_ - learning_rate * gradient
        '''
        return model
    else: # for ITML - 2; manually compute gradient and update parameters
        model.fit(X, y)
        '''
        X_batch, y_batch = random_choose(X, y, batch_size=batch_size)
        gradient = np.multiply(X_batch, (model.predict(X_batch) - y_batch))
        gradient = np.resize(np.sum(gradient, axis=0)/len(X_batch),(1, 100))
        model.coef_ = model.coef_ - learning_rate * gradient
        '''
    return model


def get_models_results(num_of_train_samples, alpha, num_of_rounds):
    global noise_std_dev

    optimum_coef = get_optimum_model()
    X_good, y_good, X_bad, y_bad = generate_samples(
        num_of_train_samples, alpha, optimum_coef, noise_std_deviation=noise_std_dev)

    # have mixed dataset
    X = np.concatenate((X_good, X_bad))
    y = np.concatenate((y_good, y_bad))

    baseline_model_loss = 0
    if draw_figure[0]:
        # Base line model
        
        baseline_model = LinearRegression(fit_intercept=False)
        baseline_model = baseline_model.fit(X, y)

        baseline_model_loss = np.sqrt((np.square(baseline_model.coef_ - optimum_coef)).sum()/len(optimum_coef))

    model_itlm_1_loss = 0
    if draw_figure[1]:
        # Two models to be tested
        model_itlm_1 = LinearRegression(fit_intercept=False) # init
        model_itlm_1.fit(X_bad, y_bad) # just for validating the model
        model_itlm_1.coef_ = np.random.rand(1, 100)

        model_itlm_1 = itlm.itlm_1(X, y, num_of_rounds, alpha, model_itlm_1,
                                select_subset, update_model)

        model_itlm_1_loss = np.sqrt((np.square(model_itlm_1.coef_ - optimum_coef)).sum()/len(optimum_coef))

    model_itlm_2_loss = 0
    if draw_figure[2]:
        model_itlm_2 = LinearRegression(fit_intercept=False) # init 
        model_itlm_2.fit(X_bad, y_bad) # just for validating the model
        model_itlm_2.coef_ = np.random.rand(1, 100)
        model_itlm_2 = itlm.itlm_2(X, y, num_of_rounds * 30, alpha, model_itlm_2,
                                select_subset, update_model)

        model_itlm_2_loss = np.sqrt((np.square(model_itlm_2.coef_ - optimum_coef)).sum()/len(optimum_coef))

    oracle_model_loss = 0
    if draw_figure[3]:
        # oracle 
        oracle_model = LinearRegression(fit_intercept=False)
        oracle_model = oracle_model.fit(X_good, y_good)

        oracle_model_loss = np.sqrt((np.square(oracle_model.coef_ - optimum_coef)).sum()/len(optimum_coef))

    return np.array([baseline_model_loss, model_itlm_1_loss, model_itlm_2_loss, oracle_model_loss])


def test_sample_size(times, sample_sizes = None, large_noise = False):
    if sample_sizes is None:
        sample_sizes = [200, 300, 400, 1000, 2000, 4000, 10000]
    len(sample_sizes)
    avg_loss = np.zeros((len(sample_sizes), model_num))
    global file_nums
    # conduct multiple times
    for i in range(times):
        loss = np.zeros((len(sample_sizes), model_num))
        print("Test sample size times {}".format(i + 1))
        for j in range(len(sample_sizes)):
            # small noise
            # 100 is the hyper parameter in ITML
            temp = get_models_results(sample_sizes[j], 0.7, 100)
            loss[j] = temp

        avg_loss += loss

    avg_loss = np.divide(avg_loss, times)

    fig = plt.figure()
    ax = fig.add_subplot()
    
    title = 'sample size vs. loss'

    explanation = "alpha: {}\niterations:  {}\nnoise std: {}\ncov var: {}\nmean of $x_i$: \n  {}".format(0.7, times, noise_std_dev, cov_var, feature_vector)

    ax.set(xlabel='sample size', ylabel='loss',
           title=title)

    avg_loss = avg_loss.transpose()

    for i in range(len(avg_loss)):
        if draw_figure[i]:
            ax.plot(sample_sizes, avg_loss[i], color = colors[i], label=names[i])

    ax.text(0.70, 0.60, explanation, verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes)
    ax.legend()
    plt.savefig('./size/Figure 2 (b)' + str(file_nums), dpi=600)
    file_nums += 1


def test_alpha(times, alphas = None, title = None):
    if alphas is None:
        alphas = [i*0.03 + 0.3 for i in range(20)]
    global file_nums
    avg_loss = np.zeros((len(alphas), model_num))

    #iteration_times = [700] * int(len(alphas)/2) + [400] * (len(alphas) - int(len(alphas)/2))
    # conduct multiple times
    for i in range(times):
        print("Test alpha for {} times".format(i))
        loss = np.zeros((len(alphas), model_num))
        for j in range(len(alphas)):
            # small noise
            temp = get_models_results(1000, alphas[j], 100)
            loss[j] = temp

        avg_loss += loss/times

    fig = plt.figure()
    ax = fig.add_subplot()

    explanation = "Size: {}\niterations:  {}\nnoise std: {}\ncov var: {}\nmean of $x_i$: \n  {}".format(1000, times, noise_std_dev, cov_var, feature_vector)

    ax.set(xlabel='alpha', ylabel='loss',
           title='alpha vs. loss')

    avg_loss = avg_loss.transpose()

    for i in range(len(avg_loss)):
        plt.plot(alphas, avg_loss[i], color = colors[i], label=names[i])

    ax.text(0.70, 0.60, explanation, verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes)
    ax.legend()
    plt.savefig('./alpha/Figure 2 (c)' + str(file_nums), dpi=600)
    file_nums += 1


'''Helper functions below. Not important'''

def create_diff_oracle_baseline(num_of_train_samples, alpha):
    optimum_coef = get_optimum_model()
    X_good, y_good, X_bad, y_bad = generate_samples(num_of_train_samples, alpha, optimum_coef, noise_std_deviation=noise_std_dev)

    # have mixed dataset
    X = np.concatenate((X_good, X_bad))
    y = np.concatenate((y_good, y_bad))

    # Base line model
    baseline_model = LinearRegression(fit_intercept=False)
    baseline_model.fit(X_bad, y_bad) # just for validating the model
    baseline_model.coef_ = np.random.rand(1, 100)
    baseline_model = baseline_model.fit(X, y)
    #baseline_model = update_model(baseline_model, X, y)

    # oracle 
    oracle_model = LinearRegression(fit_intercept=False)
    oracle_model.fit(X_bad, y_bad) # just for validating the model
    oracle_model.coef_ = np.random.rand(1, 100)
    oracle_model = oracle_model.fit(X_good, y_good)
    #oracle_model = update_model(oracle_model, X_good, y_good)

    baseline_model_loss = np.sqrt((np.square(baseline_model.coef_ - optimum_coef)).sum()/len(optimum_coef))
    oracle_model_loss = np.sqrt((np.square(oracle_model.coef_ - optimum_coef)).sum()/len(optimum_coef))

    return np.array([baseline_model_loss, oracle_model_loss])

def test_oracle_baseline(times, sample_sizes = None):
    if sample_sizes is None:
        sample_sizes = [400, 600, 1000, 2000, 3000, 4000]
    len(sample_sizes)
    avg_loss = np.zeros((len(sample_sizes), 2))

    for i in range(times):
        print('Times {}'.format(i))
        loss = np.zeros((len(sample_sizes), 2))
        for j in range(len(sample_sizes)):
            # small noise
            # 100 is the hyper parameter in ITML
            temp = create_diff_oracle_baseline(sample_sizes[j], 0.7)
            loss[j] = temp

        avg_loss += loss

    avg_loss = np.divide(avg_loss, times)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(xlabel='sample size', ylabel='loss',
           title='Oracle & Baseline ({} times, alpha: {}, iterations:  {}, noise std: {}, cov var: {})'.format(times, 0.7, times, noise_std_dev, cov_var))

    avg_loss = avg_loss.transpose()
    c = ['red','blue']
    l = ['baseline', 'oracle']

    for i in range(2):
        plt.plot(sample_sizes, avg_loss[i], color = c[i], label=l[i])

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # just for create different baseline model and oracle model
    #noise_std_dev = 0.01
    #test_oracle_baseline(5, None)
    
    # Figure 2 (a)
    # modify the first parameter to determine test times for each variable
    # modify the second parameter, which denotes different sample sizes to be tested
    #test_sample_size(5, None)
    # Figure 2 (b)
    #noise_std_dev = 1 # high noise

    
    noise_std_dev = 2
    cov_var = 8
    test_sample_size(5, None)

    draw_figure[0] = True
    noise_std_dev = 2
    cov_var = 8
    test_alpha(5, None)

    '''
    many_noise_std = [1, 4]
    many_cov_var = [1, 4]

    for i in range(len(many_noise_std)):
        noise_std_dev = many_noise_std[i]
        for j in range(len(many_cov_var)):
            cov_var = many_cov_var[j]
            test_sample_size(5, None, True)

    file_nums = 0
    draw_figure[0] = True
    for i in range(len(many_noise_std)):
        noise_std_dev = many_noise_std[i]
        for j in range(len(many_cov_var)):
            cov_var = many_cov_var[j]
            test_alpha(5, None)
'''
    # Figure 2 (c)
    # modify the first parameter to determine test times for each variable
    # modify the second parameter, which denotes different alphas to be tested
    # Default tested alphas are: [i/50 + 0.5 for i in range(20)] (0.5, 0.52, 0.54,..., 0.9)

    #noise_std_dev = 1 # small noise

    #test_alpha(5, None)



