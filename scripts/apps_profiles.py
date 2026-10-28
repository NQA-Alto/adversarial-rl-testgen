def integer(low=-3, high=5):
    return {"type": "int", "min": low, "max": high}


def sequence(alphabet=(-2, 0, 1, 3), minimum=0, maximum=4):
    return {"type": "list", "alphabet": list(alphabet), "min_length": minimum, "max_length": maximum}


def text(alphabet="abA", minimum=0, maximum=4):
    return {"type": "str", "alphabet": list(alphabet), "min_length": minimum, "max_length": maximum}


PROFILES = {
    2665: ("Tìm phòng họp trống đầu tiên", [sequence(("X", "O"), 1, 5)], []),
    2667: ("Xe có đủ nhiên liệu đến trạm không?", [integer(1, 8), integer(1, 4), integer(1, 4)], []),
    2703: ("Tổng bình phương các phần tử", [sequence()], []),
    2704: ("Xóa tối đa một phần tử để tăng nghiêm ngặt", [sequence(minimum=2)], []),
    2746: ("Ký tự tại vị trí cho trước có phải nguyên âm?", [text("aEb", maximum=3), integer(-1, 4)], []),
    2755: ("Lọc phần tử chia hết cho chỉ số", [sequence(minimum=2)], []),
    2787: ("Giữ danh sách sau khi bỏ một lát cắt", [sequence(maximum=3), integer(0, 4), integer(0, 4)], [("lt", 1, 2)]),
    2820: ("Tìm các vị trí khác nhau của hai chuỗi", [text(maximum=3), text(maximum=3)], [("same_length", 0, 1)]),
    2831: ("Tổng lớn nhất của hai phần tử", [sequence(minimum=2)], []),
    2834: ("Điểm đối xứng qua một tâm", [sequence((-1, 0, 2), 2, 2), sequence((-1, 0, 2), 2, 2)], []),
    2849: ("Chỉ số cân bằng tổng trái và phải", [sequence(minimum=1)], [("not_all_zero", 0)]),
    2859: ("Tổng đoạn con lớn nhất", [sequence()], []),
    2865: ("Đảo ngược chuỗi", [text()], []),
    2868: ("Mỗi số có một hàng xóm chênh lệch một", [sequence()], []),
    2871: ("Kiểm tra tiền tố, có phân biệt hoa thường", [text(maximum=3), text(maximum=3)], []),
    2899: ("Số lần lật hậu tố chuỗi nhị phân", [text("01", 1, 6)], []),
    2910: ("Mọi phần tử có nhỏ hơn hoặc bằng giới hạn?", [sequence(minimum=1, maximum=3), integer(-2, 3)], []),
    2912: ("Liệt kê các bội số đến giới hạn", [integer(1, 4), integer(2, 8)], [("lt", 0, 1)]),
    2914: ("Loại mọi phần tử xuất hiện trong danh sách thứ hai", [sequence((-1, 0, 2), maximum=3), sequence((-1, 0, 2), maximum=3)], []),
    2935: ("Kiểm tra chuỗi có đúng một nguyên âm", [text("aEb", maximum=3)], []),
    2937: ("Liệt kê số nguyên trong đoạn đóng", [integer(-3, 3), integer(-3, 3)], [("lt", 0, 1)]),
    2947: ("Làm tròn đến bội gần nhất, giữ nguyên khi ở giữa", [integer(1, 12), integer(3, 8)], []),
    2948: ("Phân hoạch ổn định theo ngưỡng", [integer(-2, 3), sequence(maximum=3)], []),
    2958: ("Đếm hình hộp con", [integer(1, 4), integer(1, 4), integer(1, 4)], []),
    2961: ("Hoàn thành dãy hoặc trả về số không khi trùng", [sequence((0, 1, 2, 3), 1, 4)], []),
    2962: ("Lọc các phần tử chia hết cho một số", [sequence(maximum=3), integer(1, 4)], []),
    2964: ("Tổng hai số dương nhỏ nhất", [sequence((1, 2, 3, 5), 4, 4)], []),
    2968: ("Lấy một hoặc hai ký tự ở giữa", [text(minimum=1, maximum=5)], []),
    2987: ("Một số có chia hết cho cả hai số?", [integer(-6, 6), {"type": "int", "enum": [-3, -2, -1, 1, 2, 3]}, {"type": "int", "enum": [-3, -2, -1, 1, 2, 3]}], []),
    2991: ("Dự đoán màu viên bi cuối cùng", [integer(1, 6), integer(1, 6)], []),
    2996: ("Tính lượng cà phê theo các sự kiện", [sequence(("cw", "CW", "dog", "x"), maximum=3)], []),
    3009: ("Đếm cặp số liên tiếp không chồng lấn", [sequence((-1, 0, 1, 2), maximum=5)], []),
    3014: ("Mã hóa chuyển vị hai hàng", [text("ab ", maximum=5)], []),
    3019: ("Đếm số lần một ký tự xuất hiện", [text(maximum=4), text(minimum=1, maximum=1)], []),
    3024: ("Giữ các tên có đúng bốn ký tự", [sequence(("An", "Nam", "Anna", "Luan", "QuocAn"), maximum=3)], []),
}

VALIDATION_IDS = {2665, 2704, 2834, 2868, 2947, 2968, 3014}
TEST_IDS = {2746, 2755, 2849, 2871, 2912, 2958, 2996}


def split_for(identity):
    return "validation" if identity in VALIDATION_IDS else "test" if identity in TEST_IDS else "train"
