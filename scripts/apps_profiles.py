def integer(low=-3, high=5):
    return {"type": "int", "min": low, "max": high}


def sequence(alphabet=(-2, 0, 1, 3), minimum=0, maximum=4):
    return {"type": "list", "alphabet": list(alphabet), "min_length": minimum, "max_length": maximum}


def text(alphabet="abA", minimum=0, maximum=4):
    return {"type": "str", "alphabet": list(alphabet), "min_length": minimum, "max_length": maximum}


PROFILES = {
    2665: ("Find the first available meeting room", [sequence(("X", "O"), 1, 5)], []),
    2667: ("Check fuel range to the next station", [integer(1, 8), integer(1, 4), integer(1, 4)], []),
    2703: ("Sum of squares", [sequence()], []),
    2704: ("Strictly increasing after removing at most one element", [sequence(minimum=2)], []),
    2746: ("Check for a vowel at a given index", [text("aEb", maximum=3), integer(-1, 4)], []),
    2755: ("Filter elements divisible by their index", [sequence(minimum=2)], []),
    2787: ("Remove a list slice", [sequence(maximum=3), integer(0, 4), integer(0, 4)], [("lt", 1, 2)]),
    2820: ("Find differing positions in two strings", [text(maximum=3), text(maximum=3)], [("same_length", 0, 1)]),
    2831: ("Maximum sum of two elements", [sequence(minimum=2)], []),
    2834: ("Reflect a point across a center", [sequence((-1, 0, 2), 2, 2), sequence((-1, 0, 2), 2, 2)], []),
    2849: ("Find an index with equal left and right sums", [sequence(minimum=1)], [("not_all_zero", 0)]),
    2859: ("Maximum subarray sum", [sequence()], []),
    2865: ("Reverse a string", [text()], []),
    2868: ("Check for a neighbor differing by one", [sequence()], []),
    2871: ("Case-sensitive prefix check", [text(maximum=3), text(maximum=3)], []),
    2899: ("Count binary suffix flips", [text("01", 1, 6)], []),
    2910: ("Check whether all elements are within a limit", [sequence(minimum=1, maximum=3), integer(-2, 3)], []),
    2912: ("List multiples up to a limit", [integer(1, 4), integer(2, 8)], [("lt", 0, 1)]),
    2914: ("Remove elements present in the second list", [sequence((-1, 0, 2), maximum=3), sequence((-1, 0, 2), maximum=3)], []),
    2935: ("Check for exactly one vowel", [text("aEb", maximum=3)], []),
    2937: ("List integers in a closed interval", [integer(-3, 3), integer(-3, 3)], [("lt", 0, 1)]),
    2947: ("Round to the nearest multiple, unchanged on ties", [integer(1, 12), integer(3, 8)], []),
    2948: ("Stable partition by threshold", [integer(-2, 3), sequence(maximum=3)], []),
    2958: ("Count sub-cuboids", [integer(1, 4), integer(1, 4), integer(1, 4)], []),
    2961: ("Complete a sequence, returning zero for duplicates", [sequence((0, 1, 2, 3), 1, 4)], []),
    2962: ("Filter elements divisible by a number", [sequence(maximum=3), integer(1, 4)], []),
    2964: ("Sum the two smallest positive numbers", [sequence((1, 2, 3, 5), 4, 4)], []),
    2968: ("Extract the middle character or pair", [text(minimum=1, maximum=5)], []),
    2987: ("Check divisibility by two numbers", [integer(-6, 6), {"type": "int", "enum": [-3, -2, -1, 1, 2, 3]}, {"type": "int", "enum": [-3, -2, -1, 1, 2, 3]}], []),
    2991: ("Predict the final marble color", [integer(1, 6), integer(1, 6)], []),
    2996: ("Count coffees by event", [sequence(("cw", "CW", "dog", "x"), maximum=3)], []),
    3009: ("Count non-overlapping consecutive pairs", [sequence((-1, 0, 1, 2), maximum=5)], []),
    3014: ("Two-row transposition cipher", [text("ab ", maximum=5)], []),
    3019: ("Count character occurrences", [text(maximum=4), text(minimum=1, maximum=1)], []),
    3024: ("Keep four-character names", [sequence(("An", "Nam", "Anna", "Luan", "QuocAn"), maximum=3)], []),
}

VALIDATION_IDS = {2665, 2704, 2834, 2868, 2947, 2968, 3014}
TEST_IDS = {2746, 2755, 2849, 2871, 2912, 2958, 2996}


def split_for(identity):
    return "validation" if identity in VALIDATION_IDS else "test" if identity in TEST_IDS else "train"
