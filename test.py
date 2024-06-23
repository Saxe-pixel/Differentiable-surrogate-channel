def count_combined_nn_parameters(filter_length, num_filters, is_complex=False):
    if is_complex:
        return 2 * (num_filters * filter_length + num_filters**2 * filter_length + num_filters * filter_length + num_filters + 2 * num_filters + 3)
    else:
        return 2 * (num_filters * filter_length + num_filters * filter_length + 3)

# Combined WHChannelNet
num_filters_range_combined = range(1, 10)
filter_length_combined = 53
nn_combined_params = [(num_filters, count_combined_nn_parameters(filter_length_combined, num_filters)) for num_filters in num_filters_range_combined]

# Combined WHChannelNetComplex
num_filters_range_complex_combined = range(1, 9)
filter_length_combined_complex = 13
nn_combined_complex_params = [(num_filters, count_combined_nn_parameters(filter_length_combined_complex, num_filters, True)) for num_filters in num_filters_range_complex_combined]


print("NN Combined Parameters:", nn_combined_params)
print("NN Combined (Complex) Parameters:", nn_combined_complex_params)
