import numpy as np

def fairRR(arr, eps, num_int, num_bit, mode='relax'):
    r = arr.shape[1]
    num_pt = arr.shape[0]

    def float_to_binary(x, m=num_int, n=num_bit - num_int):
        x_abs = np.abs(x)
        x_scaled = round(x_abs * 2 ** n)
        res = '{:0{}b}'.format(x_scaled, m + n)
        # if x >= 0:
        #     res = '0' + res
        # else:
        #     res = '1' + res
        return res

    # binary to float
    def binary_to_float(bstr, m=num_int, n=num_bit - num_int):
        # sign = bstr[0]
        bs = bstr
        res = int(bs, 2) / 2 ** n
        # if int(sign) == 1:
        #     res = -1 * res
        return res

    def string_to_int(a):
        bit_str = "".join(x for x in a)
        return np.array(list(bit_str)).astype(int)

    def join_string(a, num_bit=num_bit, num_feat=r):
        res = np.empty(num_feat, dtype="S10")
        # res = []
        for i in range(num_feat):
            # res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
            res[i] = "".join(str(x) for x in a[i * num_bit:(i + 1) * num_bit])
        return res

    def alpha_tr1(r, eps, l):
        return np.exp((eps - r * eps * (l - 1)) / (2 * r * l))

    def alpha(r, eps, l):
        nu = 2 * (np.sqrt(6 * np.log(10) / (2 * r)))
        sum_ = 0
        for k in range(l):
            sum_ += np.exp(2 * eps * k / l)
        return np.sqrt(((1 - nu) * eps + r * l) / (2 * r * sum_))

    max_val = sum([2 ** i for i in range(num_int)]) + sum(
        [2 ** (-1 * i) for i in range(1, num_bit - num_int)])
    min_val = 2 ** (-1 * (num_bit - num_int))

    max_ = np.max(arr)
    min_ = np.min(arr)
    arr = (arr - min_) / (max_ - min_) * (max_val - min_val) + min_val

    # max_ = np.max(arr)
    # min_ = np.min(arr)
    # arr = (arr - min_) / (max_ - min_) * (2 ** num_int - 1)

    alpha_ = alpha_tr1(r=r, eps=eps, l=num_bit) if mode == 'dp' else alpha(r=r, eps=eps, l=num_bit)

    float_to_binary_vec = np.vectorize(float_to_binary)
    binary_to_float_vec = np.vectorize(binary_to_float)

    feat_tmp = float_to_binary_vec(arr)
    feat = np.apply_along_axis(string_to_int, 1, feat_tmp)
    print(np.max(arr), np.min(arr), arr.shape, feat.shape)
    index_matrix = np.array(range(num_bit))
    index_matrix = np.tile(index_matrix, (num_pt, r))
    p = 1 / (1 + alpha_ * np.exp(index_matrix * eps / num_bit))
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    print(feat[0][:10])
    print(perturb[0][:10])
    perturb_feat = (perturb + feat) % 2
    print(perturb_feat[0][:10])
    perturb_feat = np.apply_along_axis(join_string, 1, perturb_feat)
    perturb_feat = binary_to_float_vec(perturb_feat)
    print(arr[0][:2])
    print(perturb_feat[0][:2])
    perturb_feat = (perturb_feat - min_val) / (max_val - min_val) * (max_ - min_) + min_
    return perturb_feat