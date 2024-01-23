import sys

def remove_comments(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as output_file:
        for line in lines:
            # 라인이 '#'으로 시작하지 않는 경우에만 쓰기
            stripped_line = line.strip()
            if not stripped_line.startswith('#'):
                output_file.write(line)

def main():
    if len(sys.argv) != 3:
        print("사용 방법: python remove_comments.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    remove_comments(input_file, output_file)
    print(f"주석이 제거된 파일이 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()
