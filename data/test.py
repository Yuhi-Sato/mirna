import csv


def format_float(value):
    # 有効数字を15桁にフォーマット
    return format(value, ".15g")


def process_csv(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # 行の各要素に対して有効数字のフォーマットを適用
            formatted_row = [
                format_float(float(cell)) if "." in cell else cell for cell in row
            ]

            # フォーマットされた行をCSVファイルに書き込む
            writer.writerow(formatted_row)


# 使用例
input_csv_file = "patient.csv"
output_csv_file = "output.csv"
process_csv(input_csv_file, output_csv_file)
