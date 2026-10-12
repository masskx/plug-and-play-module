DEFAULT_DIVISOR = 8


def make_divisible(value, divisor=DEFAULT_DIVISOR):
    return max(divisor, int(value + divisor / 2) // divisor * divisor)
