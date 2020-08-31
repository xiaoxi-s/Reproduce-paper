
def itlm_1(X, y, num_of_rounds, alpha, model, select_subset, update_model, **arg):
    n = len(X)
    good_n = int(alpha * n)
    bad_n = n - good_n

    # the size of subset
    size_of_subset = int((alpha - 0.05) * n)

    for i in range(num_of_rounds):

        # select subset with good_n samples
        X_selected, y_selected = select_subset(model, X, y, size_of_subset)
        model = update_model(model, X_selected, y_selected)

    return model


def itlm_2(X, y, alpha, model, select_subset, update_model, **arg):
    n = len(X)
    good_n = int(alpha * n)
    bad_n = n - good_n

    # the size of subset
    size_of_subset = int((alpha - 0.05) * n)
    # select subset with good_n samples
    X_selected, y_selected = select_subset(model, X, y, size_of_subset)
    model = update_model(model, X_selected, y_selected, False)

    return model