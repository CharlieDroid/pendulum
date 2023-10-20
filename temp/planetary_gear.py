from operator import itemgetter
def get_ring(s, p):
    return (2 * p) + s


def get_ratio(s, r):
    return 1 + (r / s)


def get_sr_np_ratio(s, r, num_p):
    return (s + r) / num_p


def coprime(s, p, r):
    def factors(a):
        return [i for i in range(2, a + 1) if a % i == 0]
    for sf in factors(s):
        for pf in factors(p):
            for rf in factors(r):
                if (sf == pf) or (sf == rf) or (pf == rf):
                    return False
    return True


def get_error(meas, theo):
    return (abs(meas - theo)/theo)*100


if __name__ == "__main__":
    # R = 2 * P + S -
    # (S + R) / Np = k -
    # coprime(s, r, p)
    # i:1, i = 1 + R / S -
    # module > 1.5 mm

    np_range = (3, 10 + 1)
    s_range = (8, 25 + 1)
    r_range = (24, 60 + 1)
    p_range = (10, 30 + 1)
    desired_ratio = 2.4
    found_combinations = []
    for np in range(*np_range):
        for s in range(*s_range):
            for r in range(*r_range):
                for p in range(*p_range):
                    # first condition
                    if get_ring(s, p) == r:
                        # second condition
                        if get_sr_np_ratio(s, r, np).is_integer():
                            # third condition
                            if coprime(s, r, p):
                                error = get_error(get_ratio(s, r), desired_ratio)
                                found_combinations.append((error, s, r, p, np))

    if found_combinations:
        combinations = sorted(found_combinations, key=itemgetter(0))
        print(combinations[:10])
    else:
        print("None Found")
