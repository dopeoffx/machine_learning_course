def main():
    print("Hello, world!")

    a = [1, 2, 3, 4, 5, 6]

    if len(a) < 3:
        print("Not enough elements")
        return
    elif len(a) > 6:
        a = a[:5]

    a_len = len(a)

    for i in range(a_len):
        for j in range(1, a_len):
            for k in range(2, a_len):
                print(f"[{i}, {j}, {k}]")

if __name__ == "__main__":
    main()
