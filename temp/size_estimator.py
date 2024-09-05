from engineering_notation import EngNumber as eng


def estimate_size(input_size, layer_sizes, output_size, layer_norm):
    total_size = 0
    total_size += input_size * layer_sizes[0]  # weights
    total_size += layer_sizes[0]  # biases
    if layer_norm:
        total_size += layer_sizes[0]
        total_size += layer_sizes[0]
    total_size += layer_sizes[0] * layer_sizes[1]  # weights
    total_size += layer_sizes[1]  # biases
    if layer_norm:
        total_size += layer_sizes[1]
        total_size += layer_sizes[1]
    total_size += layer_sizes[1] * output_size  # weights
    total_size += output_size  # biases

    byte_size = total_size * 4  # if float32
    print(f"total_size={eng(total_size)}, byte_size={eng(byte_size)}")


if __name__ == "__main__":
    n_extra_obs = 4
    input_size = 10 - n_extra_obs
    layer_sizes = [110, 80]
    output_size = 1
    layer_norm = True

    estimate_size(input_size, layer_sizes, output_size, layer_norm)
